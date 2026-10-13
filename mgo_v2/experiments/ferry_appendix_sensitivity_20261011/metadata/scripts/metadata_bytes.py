"""Exact decode/prefill metadata payload bytes from the code's record layout (CPU only).

Decode, per rank per MoE-layer event (one all_gather_into_tensor):
  compact (mgo_v2/compact_metadata.py:11-12):  24 + B*k + 4*E
  live    (mgo_v2/live_metadata.py:16):        24 + B*k + 4*E*min(B,128)
Prefill (mgo_v2/runtime.py:40-80, probability_tail=128, padded all_gather):
  8 (count) + T*k*8 (int64 ids) + T*k*2 (bf16 weights) + max_r(tail_r)*E*4,  T=B*L_in per rank
"""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = {  # from /home/hwlee/model/*/config.json
 'Qwen3-30B-A3B': dict(k=8, E=128, moe_layers=48, hidden=2048, inter=768),      # 48 layers, mlp_only_layers=[]
 'DeepSeek-V2-Lite': dict(k=6, E=64, moe_layers=26, hidden=2048, inter=1408),   # 27 layers, first_k_dense_replace=1
}
R = 4; W = 128; BATCHES = [1, 8, 16, 32, 64, 128, 256]

def expert_bytes(m): return 3 * m['hidden'] * m['inter'] * 2   # gate/up/down, BF16

def decode_rows():
    rows = []
    for name, m in MODELS.items():
        k, E, L = m['k'], m['E'], m['moe_layers']; S = expert_bytes(m)
        for B in BATCHES:
            for path in ('compact', 'live'):
                per = 24 + B * k + 4 * E * (1 if path == 'compact' else min(B, W))
                rows_sent = (1 if path == 'compact' else R * min(B, W))
                rows_needed = min(W, R * B)
                note = ''
                if path == 'compact':
                    note = ('frozen-trace path only; gate is a 1-row float32 summary (last rank only consumed, compact_metadata.py:53); '
                            + ('B<W requires frozen_gate from trace (compact_metadata.py:26-27)' if B < W else 'B>=W: live tail mean of last rank'))
                rows.append(dict(model=name, k=k, E=E, moe_layers=L, R=R, local_batch=B, path=path,
                    ids_bytes=B * k, header_bytes=24, gate_bytes=per - 24 - B * k,
                    per_rank_send_bytes=per, gathered_total_bytes=R * per, received_from_peers_bytes=(R - 1) * per,
                    per_step_per_rank_send_bytes=per * L, per_step_gathered_bytes=R * per * L,
                    expert_weight_bytes=S, gathered_per_event_over_expert=round(R * per / S, 6),
                    gathered_per_step_over_expert=round(R * per * L / S, 4),
                    prob_rows_sent_global=(R * min(B, W) if path == 'live' else ''),
                    prob_rows_used_by_W128=(rows_needed if path == 'live' else ''),
                    prob_row_redundancy=(round(R * min(B, W) / rows_needed, 3) if path == 'live' else ''),
                    used_in_main_table=(path == 'live' and B in (16, 64)), note=note))
    return rows

def prefill_rows():
    rows = []
    for name, m in MODELS.items():
        k, E, L = m['k'], m['E'], m['moe_layers']
        for B in (16, 64):
            for Lin in (512, 1024):
                T = B * Lin; tail_max = min(W, T)   # last rank holds the whole global 128-row suffix
                per = 8 + T * k * 8 + T * k * 2 + tail_max * E * 4
                rows.append(dict(model=name, local_batch=B, input_tokens=Lin, tokens_per_rank=T,
                    per_rank_send_bytes_per_layer=per, gathered_total_bytes_per_layer=R * per,
                    per_rank_send_bytes_all_moe_layers=per * L))
    return rows

def write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def plot(rows, out):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    col = {'live': '#2a78d6', 'compact': '#eb6834'}
    for ax, name in zip(axes, MODELS):
        rs = [r for r in rows if r['model'] == name]
        for path, lab in (('live', 'live (main cells)'), ('compact', 'compact (frozen trace)')):
            xs = [r['local_batch'] for r in rs if r['path'] == path]; ys = [r['gathered_total_bytes'] for r in rs if r['path'] == path]
            ax.plot(xs, ys, color=col[path], lw=2, marker='o', ms=4, label=lab)
        S = rs[0]['expert_weight_bytes']
        ax.axhline(S, color='#52514e', lw=1, ls='--')
        ax.text(1, S * 1.25, f'one expert ({S/2**20:.1f} MiB)', fontsize=8, color='#52514e')
        ax.axvline(W, color='#9a9994', lw=1, ls=':')
        ax.text(W * 1.06, 2e3, 'tail cap\nmin(B,128)', fontsize=7.5, color='#52514e')
        for B in (16, 64):
            r = next(r for r in rs if r['path'] == 'live' and r['local_batch'] == B)
            ax.annotate(f"{r['gathered_total_bytes']/1024:.0f} KiB", (B, r['gathered_total_bytes']),
                        textcoords='offset points', xytext=(-6, 7), fontsize=7.5, ha='right', color='#0b0b0b')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(BATCHES); ax.set_xticklabels(BATCHES)
        ax.set_title(f"{name} (k={MODELS[name]['k']}, E={MODELS[name]['E']})", fontsize=9)
        ax.set_xlabel('local batch B per rank'); ax.grid(axis='y', color='#e6e5e1', lw=0.6)
    axes[0].set_ylabel('all-gathered bytes per layer event, R4')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(f'{out}.{ext}', dpi=200)

if __name__ == '__main__':
    out = Path(sys.argv[1]); d = decode_rows()
    write(out / 'METADATA_BYTES.csv', d); write(out / 'METADATA_BYTES_PREFILL.csv', prefill_rows())
    plot(d, out / 'FIG_METADATA_BYTES')
