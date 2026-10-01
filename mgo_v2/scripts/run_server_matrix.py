#!/usr/bin/env python3
"""Run bounded A/B/C anchors and the R4/R8 batch/cache matrix sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "offload-dir", "similarity", "affinity", "workload", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--worlds", type=int, nargs="+", default=[4, 8])
    p.add_argument("--phases", nargs="+", choices=["ablation", "matrix"], default=["ablation", "matrix"])
    args = p.parse_args()
    package = Path(__file__).resolve().parents[1]
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = {"global_cache_ratio": .3, "admission": "random", "eviction": "lru", "substitution_enabled": True}
    ablation = [{"name": "A_exact", "batch": 8, "config": dict(base, substitution_enabled=False)}]
    ablation += [{"name": "B_" + eviction, "batch": 8, "config": dict(base, eviction=eviction)}
                 for eviction in ("lru", "gate", "coverage")]
    ablation += [{"name": "C_" + admission, "batch": 8,
                  "config": dict(base, eviction="coverage", admission=admission)}
                 for admission in ("greedy_current", "greedy_path", "hungarian_current", "hungarian_same",
                                   "hungarian_same_path", "hungarian_swap")]
    matrix = [{"name": f"b{batch}_c{percent}", "batch": batch,
               "config": dict(base, global_cache_ratio=percent / 100, eviction="coverage", admission="hungarian_same_path")}
              for batch in (4, 8, 16, 32) for percent in (10, 20, 30, 40, 50)]
    conditions = {"ablation": ablation, "matrix": matrix}
    for phase, cells in conditions.items():
        (root / f"{phase}_cells.json").write_text(json.dumps(cells, indent=2) + "\n")
    started = time.time()
    runs = []
    selection = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    for phase in args.phases:
        for world in args.worlds:
            if world > len(selection):
                raise ValueError("not enough visible GPUs for requested world size")
            name = f"{phase}_r{world}"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(selection[:world]),
                       PYTHONPATH=str(package) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                       OMP_NUM_THREADS="4", MOE_EP_DISABLE_ARCHER_EVICT="1", MOE_EP_NATIVE_NUMERICS="1", MOE_EP_SLOT_VIEWS="1")
            command = [args.python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                       str(package / "examples" / "benchmark_model.py"),
                       "--model", args.model, "--offload-dir", args.offload_dir,
                       "--similarity", args.similarity, "--affinity", args.affinity, "--workload", args.workload,
                       "--cells", str(root / f"{phase}_cells.json"), "--output", str(root / name),
                       "--steps", str(args.steps), "--repeats", str(args.repeats)]
            if phase == "ablation":
                command.append("--profile")
            state = {"status": "RUNNING", "current": name, "started_unix": started, "runs": runs,
                     "command": command, "cells": len(conditions[phase]), "repeats": args.repeats}
            (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps({"started": name, "cells": state["cells"], "repeats": args.repeats}), flush=True)
            with (root / f"{name}.log").open("a") as log:
                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            runs.append({"name": name, "exit_code": code})
            print(json.dumps(runs[-1]), flush=True)
            if code:
                state.update(status="FAIL", runs=runs, failed=name)
                (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")
                raise SystemExit(code)
    (root / "status.json").write_text(json.dumps({"status": "PASS", "runs": runs,
        "elapsed_seconds": time.time() - started, "steps": args.steps, "repeats": args.repeats,
        "scope": "Short fixed-step cold-cache measurements and numeric accuracy screening; no production or population-quality claim."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
