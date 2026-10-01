#!/usr/bin/env python3
"""Plot the audited matrix; requires optional matplotlib, not the runtime."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    if summary["status"] != "PASS":
        raise ValueError("only plot a complete audited matrix with matched native controls")
    rows = [row for row in summary["cells"] if row["phase"] == "matrix"]
    if len(rows) != 40:
        raise ValueError("expected R4/R8 × four local batches × five cache ratios")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    batches, ratios = [4, 8, 16, 32], [.1, .2, .3, .4, .5]
    top = max(row["output_tokens_per_second"] for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
    for ax, world in zip(axes, (4, 8)):
        grid = np.full((4, 5), np.nan)
        for row in rows:
            if row["world"] == world:
                grid[batches.index(row["local_batch"]), ratios.index(row["cache_ratio"])] = row["output_tokens_per_second"]
        if not np.isfinite(grid).all():
            raise ValueError("matrix has a missing or nonfinite cell")
        plotted = ax.imshow(grid, cmap="viridis", vmin=0, vmax=top, aspect="auto")
        ax.set_xticks(range(5), [f"{ratio:.0%}" for ratio in ratios])
        ax.set_yticks(range(4), [f"{batch} ({batch * world} global)" for batch in batches])
        ax.set_xlabel("Global expert cache ratio")
        ax.set_ylabel("Local batch size")
        ax.set_title(f"{world} GPUs")
        for y in range(4):
            for x in range(5):
                ax.text(x, y, f"{grid[y,x]:.1f}", ha="center", va="center", fontsize=12,
                        color="white" if grid[y,x] / top < .55 else "black")
    fig.colorbar(plotted, ax=axes, label="Global fixed-step output tokens / second", shrink=.8)
    fig.suptitle("Qwen3-30B-A3B: measured generation throughput\n"
                 "Coverage + Hungarian same/path · 16 steps · median of 2 repeats · cold expert cache", fontsize=13)
    fig.supxlabel("Includes work after EOS. See CSV for repeat ranges and paired quality; question subsets vary with batch.", fontsize=9)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = output / f"throughput.{suffix}"
        fig.savefig(path, dpi=180)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
