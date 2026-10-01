"""Print the naive-vs-balanced owner-policy comparison table from two result
JSONs (coslot_owner_bench output).

Usage: python3 scripts/format_owner_bench.py <results_dir>
"""
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else open("/tmp/coslot_ownerbench_dir.txt").read().strip()
n = json.load(open(os.path.join(d, "naive.json")))
b = json.load(open(os.path.join(d, "balanced.json")))


def us(j, k):
    return j.get(k, 0) / 1e6  # us → s


def row(label, fn, fmt="{:.3f}"):
    vn, vb = fn(n), fn(b)
    print(f"  {label:34s} | {fmt.format(vn):>14} | {fmt.format(vb):>14}")


print("=" * 72)
print(f"OWNER POLICY COMPARISON  (235B EP={n['world']}, cap={n['cap_per_rank']}/rank, "
      f"bs={n['batch_per_rank']}/rank, in={n['input_len']}, out={n['output_len']})")
print(f"results: {d}")
print(f"same sample_ids: {n['sample_ids'] == b['sample_ids']}  (n={len(n['sample_ids'])})")
print("=" * 72)
print(f"  {'metric':34s} | {'naive':>14} | {'balanced':>14}")
print("  " + "-" * 68)
row("prefill latency / TTFT (s)", lambda j: j["ttft_s"])
row("prefill throughput (tok/s)", lambda j: j["prefill_throughput_tok_s"], "{:.1f}")
row("decode total latency (s)", lambda j: j["decode_total_latency_s"])
row("TPOT (ms/token)", lambda j: j["tpot_ms"], "{:.1f}")
print("  " + "-" * 68)
row("prefill a2a (s)", lambda j: us(j, "prefill_a2a_us"))
row("decode a2a (s)", lambda j: us(j, "decode_a2a_us"))
row("total a2a (s)", lambda j: us(j, "total_a2a_us"))
print("  " + "-" * 68)
row("decode fetch wait (s)", lambda j: us(j, "decode_fetch_us"))
row("decode weight-copy (s)", lambda j: us(j, "decode_weightcopy_us"))
row("decode compute/GEMM (s)", lambda j: us(j, "decode_compute_us"))
row("decode combine (s)", lambda j: us(j, "decode_combine_us"))
print("  " + "-" * 68)
row("routed tokens prefill imbal (max/mean)", lambda j: j["routed_prefill"]["max_over_mean"], "{:.4f}")
row("routed tokens decode imbal (max/mean)", lambda j: j["routed_decode"]["max_over_mean"], "{:.4f}")
print("  " + "-" * 68)

# ---- PCIe FETCH WORKLOAD (balanced's actual objective) ----
fw_n, fw_b = n.get("fetch_workload"), b.get("fetch_workload")
if fw_n and fw_b:
    print("  PCIe FETCH WORKLOAD  (balanced balances THIS, not routed tokens)")
    print("  " + "-" * 68)
    fd = lambda j: j["fetch_workload"]["decode"]["fetch_ops"]
    row("fetch_ops decode imbal (max/mean)", lambda j: fd(j)["max_over_mean"], "{:.4f}")
    row("fetch_ops decode imbal (std/mean)", lambda j: fd(j)["std_over_mean"], "{:.4f}")
    row("fetch_ops decode total", lambda j: sum(fd(j)["per_rank"]), "{:.0f}")
    row("per-layer fetch imbal mean (decode)",
        lambda j: j["fetch_workload"]["decode"]["layer_imbal_mean"], "{:.4f}")
    row("per-layer fetch imbal p95 (decode)",
        lambda j: j["fetch_workload"]["decode"]["layer_imbal_p95"], "{:.4f}")
    print("  " + "-" * 68)
    print(f"  fetch_ops decode per-rank: naive={fd(n)['per_rank']}  balanced={fd(b)['per_rank']}")
    print(f"  decode fetch_wait us/rank: naive={fw_n['decode']['fetch_wait_us_per_rank']}")
    print(f"                             bal  ={fw_b['decode']['fetch_wait_us_per_rank']}")
    print(f"  decode direct/staging:     naive direct={fw_n['decode']['direct_per_rank']} staging={fw_n['decode']['staging_per_rank']}")
    print(f"                             bal   direct={fw_b['decode']['direct_per_rank']} staging={fw_b['decode']['staging_per_rank']}")
    print(f"  naive miss expert%ep dist (decode): {fw_n['decode']['miss_expert_modulo']}")
    print("  " + "-" * 68)
    # ---- judgment Q1-Q3 ----
    n_imb, b_imb = fd(n)["max_over_mean"], fd(b)["max_over_mean"]
    n_tpot, b_tpot = n["tpot_ms"], b["tpot_ms"]
    print("  JUDGMENT (fetch-workload balancing):")
    near_uniform = n_imb < 1.05
    reduced = b_imb < n_imb - 1e-4
    print(f"    Q3 naive already balanced? {near_uniform}  "
          f"(naive fetch imbal={n_imb:.4f}; e%ep dist≈uniform → yes)")
    print(f"    Q1 balanced reduced fetch imbal? {reduced}  "
          f"(naive {n_imb:.4f} → balanced {b_imb:.4f})")
    lat_drop = b_tpot < n_tpot - 1e-3
    print(f"    Q2 imbal reduced but latency unchanged? "
          f"{reduced and not lat_drop}  (TPOT naive {n_tpot:.1f} → bal {b_tpot:.1f} ms)")
    print("=" * 72)
for lbl, key in [("routed prefill per-rank", "routed_prefill"),
                 ("routed decode per-rank", "routed_decode")]:
    print(f"  {lbl}: naive={n[key]['per_rank']}  balanced={b[key]['per_rank']}")
print("  " + "-" * 68)
row("cache hits", lambda j: j.get("cache_hits", 0), "{:.0f}")
row("cache misses", lambda j: j.get("cache_misses", 0), "{:.0f}")
row("drift max", lambda j: j.get("drift_max", -1), "{:.0f}")
print(f"  {'NaN seen':34s} | {str(n.get('nan_seen')):>14} | {str(b.get('nan_seen')):>14}")
print("=" * 72)
print(f"raw: {d}/naive.json  {d}/balanced.json")
print(f"     {d}/run_naive.log  {d}/run_balanced.log")
