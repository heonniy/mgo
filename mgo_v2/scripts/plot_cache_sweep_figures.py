"""Publication figures from validated, already-recorded cache-sweep receipts."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "experiments"
QWEN = ROOT / "qwen_cache_ablation_20261009"
EVICTION = ROOT / "main_eviction_history_20261007"
DEEPSEEK_EVICTION = ROOT / "deepseek_eviction_hit_20261009" / "RESULTS.json"
FIG = QWEN / "figures"
SYSTEMS = ("OURS", "MoE-Infinity", "DeepSpeed", "llama.cpp")
COLORS = ("#0072B2", "#E69F00", "#D55E00", "#009E73")
CAPS = (20, 30, 40, 50)


def read(path):
    return json.loads(path.read_text())


def metric(samples):
    return {
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "samples": samples,
    }


def performance_rows():
    ours = {x["cache_percent"]: x for x in read(QWEN / "TIMING_RECHECK.json")["cases"]}
    baseline = read(QWEN / "BASELINE_SWEEP_RESULTS.json")["rows"]
    base = {(x["cache_percent"], x["system"]): x for x in baseline if x["status"] == "PASS"}
    assert len(base) == 12 and len(ours) == 4, "Wait for the completed baseline sweep"
    rows = []
    for cap in CAPS:
        for sys in SYSTEMS:
            if sys == "OURS":
                values = ours[cap]["metrics"]
                source = str(QWEN / "TIMING_RECHECK.json")
                sample_count = 2
                extract = lambda m: values[m]["samples"]
            else:
                key = {"MoE-Infinity": "infinity", "DeepSpeed": "deepspeed", "llama.cpp": "llama"}[sys]
                item = base[cap, key]
                values = item["metrics"]
                source = item["label"]
                sample_count = item["clean_target_repeats"]
                extract = lambda m: values[m]["samples"]
            # Overall output throughput includes prefill, unlike reciprocal TPOT.
            fields = {m: metric(list(extract(m))) for m in ("TTFT", "TPOT", "E2E")}
            fields["TPS"] = metric([4096.0 / t for t in extract("E2E")])
            rows.append(dict(cache_percent=cap, system=sys, sample_count=sample_count,
                             source=source, metrics=fields))
    return rows


def bars(ax, rows, measure, systems, ylabel, ymax=None):
    x = np.arange(len(CAPS), dtype=float)
    width = 0.8 / len(systems)
    for j, sys in enumerate(systems):
        vals = [next(r for r in rows if r["cache_percent"] == c and r["system"] == sys)["metrics"][measure]
                for c in CAPS]
        y = np.array([v["median"] for v in vals])
        err = np.array([[v["median"] - v["minimum"] for v in vals],
                        [v["maximum"] - v["median"] for v in vals]])
        ax.bar(x - 0.4 + width * (j + 0.5), y, width * 0.94,
               color=COLORS[SYSTEMS.index(sys)], label=sys, zorder=2,
               yerr=err, capsize=2.2, error_kw={"lw": 0.9, "ecolor": "#20242b"})
    ax.set_xticks(x, [f"C{c}" for c in CAPS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0, top=ymax)
    ax.grid(axis="y", color="#dde3e7", lw=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def performance_figure(rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9})
    fig = plt.figure(figsize=(12, 7.9), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.03])
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[1, 1])
    bars(ax0, rows, "TTFT", SYSTEMS[:3], "TTFT (s)")
    ax0.set_title("First token · GPU expert systems")
    bars(ax1, rows, "TTFT", SYSTEMS[3:], "TTFT (s)")
    ax1.set_title("First token · llama.cpp (separate scale)")
    bars(ax2, rows, "TPOT", SYSTEMS, "TPOT (s / output token)")
    ax2.set_title("Decode latency · lower is better")
    bars(ax3, rows, "TPS", SYSTEMS, "Output throughput (token/s)")
    ax3.set_title("Entire batch throughput · higher is better")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in range(4)]
    fig.legend(handles, SYSTEMS, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Qwen3-30B · ShareGPT · R4 · B16/rank · input 512 · output 64", fontsize=13)
    fig.savefig(FIG / "qwen_cache_systems.png", dpi=240, facecolor="white")
    fig.savefig(FIG / "qwen_cache_systems.pdf", facecolor="white")
    plt.close(fig)


def eviction_figure():
    policies = ("gate-score", "LFU-cumulative", "LRU-cumulative")
    palette = {"gate-score": "#0072B2", "LFU-cumulative": "#E69F00",
               "LRU-cumulative": "#D55E00"}
    markers = {"gate-score": "o", "LFU-cumulative": "s", "LRU-cumulative": "^"}
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, layout="constrained")
    for cap, folder in ((30, EVICTION), (60, EVICTION / "C60")):
        doc = read(folder / "B16" / "SUMMARY.json")
        assert doc["status"] == "PASS" and doc["prefetch"] is False
        for item in doc["policies"]:
            if item["policy"] in policies:
                rows.append(dict(model="Qwen3-30B", capacity_percent=cap,
                                 local_batch=16, policy=item["policy"],
                                 decode_hit_rate=item["decode"]["hit_rate"],
                                 hit=item["decode"]["hit"], miss=item["decode"]["miss"]))
    deepseek = read(DEEPSEEK_EVICTION)
    assert deepseek["status"] == "PASS" and deepseek["cpu_counterfactual"]
    for case in deepseek["results"]:
        for item in case["policies"]:
            if item["policy"] in policies:
                rows.append(dict(model="DeepSeek-V2-Lite", capacity_percent=case["cache_percent"],
                                 local_batch=16, policy=item["policy"],
                                 decode_hit_rate=item["decode"]["hit_rate"],
                                 hit=item["decode"]["hit"], miss=item["decode"]["miss"]))
    for ax, model in zip(axes, ("Qwen3-30B", "DeepSeek-V2-Lite")):
        for policy in policies:
            points = [x for x in rows if x["model"] == model and x["policy"] == policy]
            assert len(points) == 2
            ax.plot([x["capacity_percent"] for x in points],
                    [100 * x["decode_hit_rate"] for x in points],
                    color=palette[policy], marker=markers[policy], lw=2.2,
                    ms=5.5, label=policy.replace("-cumulative", " (history retained)"))
        ax.set_ylim(0, 100); ax.set_ylabel("Distinct-expert hit (%)")
        ax.set_xlim(25, 65)
        ax.set_title(model, loc="right")
        ax.grid(True, color="#d7dee3", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xticks([30, 60], ["C30", "C60"])
    axes[1].set_xlabel("Expert cache capacity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=8)
    fig.suptitle("MAIN decode cache · B16/rank · frozen CPU replay · prefetch OFF", fontsize=12)
    fig.savefig(FIG / "two_model_eviction_hit_rate.png", dpi=240, facecolor="white")
    fig.savefig(FIG / "two_model_eviction_hit_rate.pdf", facecolor="white")
    plt.close(fig)
    return rows


def main():
    FIG.mkdir(exist_ok=True)
    evictions = eviction_figure()
    with (FIG / "eviction_hit_rate_source.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evictions[0], lineterminator='\n')
        writer.writeheader(); writer.writerows(evictions)
    rows = performance_rows()
    performance_figure(rows)
    (FIG / "performance_source.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (FIG / "performance_source.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(["cache_percent", "system", "metric", "median", "minimum", "maximum", "n", "source"])
        for row in rows:
            for metric_name, val in row["metrics"].items():
                writer.writerow([row["cache_percent"], row["system"], metric_name,
                                 val["median"], val["minimum"], val["maximum"],
                                 row["sample_count"], row["source"]])


if __name__ == "__main__":
    main()
