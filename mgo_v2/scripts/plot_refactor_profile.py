"""Render actual rank-local CUDA/NVTX intervals after capture and reconciliation."""
import argparse
import json
from pathlib import Path


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads(args.analysis.read_text())
    assert data['status'] == 'PASS'
    preview = data['interval_decomposition']['timeline_preview']
    origin, stop = preview['window_ns']
    lanes = [(name, spans) for name, spans in preview['intervals_ns'].items() if spans]
    fig, ax = plt.subplots(figsize=(13, max(4, len(lanes) * .45)))
    for i, (name, spans) in enumerate(lanes):
        color = '#777777' if name.startswith('CPU ') else '#e69f00' if name == 'expert_H2D_DMA' else '#0072b2' if '.nccl' in name else '#009e73'
        ax.broken_barh([((a-origin)/1e6, (b-a)/1e6) for a,b in spans], (i-.32, .64), facecolors=color)
    for boundary in preview['event_boundaries_ns']:
        x = (boundary['start_ns']-origin)/1e6
        ax.axvline(x, color='#aaaaaa', linewidth=.6, linestyle=':')
    ax.set_yticks(range(len(lanes)), [name for name, _ in lanes])
    ax.invert_yaxis()
    ax.set_xlim(0, (stop-origin)/1e6)
    ax.set_xlabel('Time from first decode MoE event (ms)')
    ax.set_title(args.title)
    ax.grid(axis='x', alpha=.15)
    fig.text(.01, .01, 'Instrumented diagnostic. GPU lanes: CUDA intervals. CPU lanes: NVTX wall intervals. NCCL includes wait/spin.', fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='BF16 decode: actual GPU and CPU intervals')
    main(parser.parse_args())
