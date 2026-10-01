#!/usr/bin/env python3
"""Sequential R8 then R4 study launcher; posthoc profiling is a separate step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--worlds", type=int, nargs="+", default=[8, 4])
    p.add_argument("--r8-extra-batches", type=int, nargs="*", choices=[16, 32], default=[16, 32])
    p.add_argument("--r4-gpus", type=int, nargs=4, default=[0, 1, 4, 5])
    p.add_argument("--resume", action="store_true", help="preserve the original binding and skip successful job receipts")
    p.add_argument("--dry-run", action="store_true", help="write manifests and execution order without launching GPUs")
    args = p.parse_args()
    assert len(set(args.r4_gpus)) == 4 and min(args.r4_gpus) >= 0
    package = Path(__file__).resolve().parents[1]
    root, inputs = Path(args.root).resolve(), Path(args.inputs).resolve()
    root.mkdir(parents=True, exist_ok=True)
    shared = ["--model", args.model, "--offload-dir", str(inputs / "expert_store"),
              "--similarity", str(inputs / "similarity.npy"), "--affinity", str(inputs / "affinity.npz"),
              "--workload", str(inputs / "screen_workload.json")]
    base = dict(global_cache_ratio=.3, substitution_enabled=True, eviction="coverage",
                same_layer_alpha=.25, path_eta=.5, seed=42)
    policies = [("C0_balanced_random", "random"), ("C1_hungarian_current", "hungarian_current"),
                ("C2_hungarian_same_path", "hungarian_same_path")]
    jobs = []
    for world in args.worlds:
        cells = [dict(name=f"b{batch}_{name}", batch=batch, config=dict(base, admission=policy))
                 for batch in ([8, 4] if world == 8 else [8]) for name, policy in policies]
        cell_path = root / f"cells_r{world}.json"
        cell_path.write_text(json.dumps(cells, indent=2) + "\n")
        jobs += [(world, f"stage_a_r{world}", "study_stage_a.py", ["--iterations", "20"]),
                 (world, f"stage_b_r{world}", "benchmark_model.py",
                  ["--cells", str(cell_path), "--steps", "65", "--repeats", "5"])]
        if world == 8 and args.r8_extra_batches:
            extra_cells = [dict(name=f"b{batch}_{name}", batch=batch, config=dict(base, admission=policy))
                           for batch in sorted(set(args.r8_extra_batches)) for name, policy in policies]
            extra_path = root / "cells_r8_extended.json"
            extra_path.write_text(json.dumps(extra_cells, indent=2) + "\n")
            jobs.append((8, "stage_b_r8_extended", "benchmark_model.py",
                         ["--cells", str(extra_path), "--steps", "65", "--repeats", "5"]))
    if args.dry_run:
        plan = [dict(world=world, name=name, script=script, arguments=shared + extra,
                     physical_gpus=args.r4_gpus if world == 4 else list(range(world)))
                for world, name, script, extra in jobs]
        (root / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps([job["name"] for job in plan]))
        return
    state = dict(status="RUNNING", started_unix=time.time(), runs=[],
                 world_order=args.worlds, fixed_decode_steps=64, generated_tokens=65,
                 r8_extra_batches=sorted(set(args.r8_extra_batches)),
                 same_support=64, same_alpha=.25, path_support=64, path_eta=.5,
                 plan_commit="1a98d10ac17557e7a9112e12ad36d07fe5d5be27",
                 source_hashes={str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in list((package / "mgo_v2").glob("*.py")) +
                                [package / "examples" / name for name in ("benchmark_model.py", "study_stage_a.py", "locality_maps.py")]})
    status = root / "status.json"
    if args.resume:
        previous = json.loads(status.read_text())
        assert previous["source_hashes"] == state["source_hashes"], "measurement source changed before resume"
        state["started_unix"] = previous["started_unix"]
        state["resumed_unix"] = time.time()
    else:
        assert not status.exists(), "existing study: use --resume after verifying prior jobs have exited"
    state["physical_gpus_by_world"] = {str(world): args.r4_gpus if world == 4 else list(range(world))
                                        for world in args.worlds}
    for world, name, script, extra in jobs:
        devices = args.r4_gpus if world == 4 else list(range(world))
        # A successful receipt is required to skip a completed job on restart.
        completed = root / f"{name}.complete.json"
        if completed.exists():
            assert args.resume, "completed job exists; use --resume"
            receipt = json.loads(completed.read_text())
            assert receipt["name"] == name and receipt["exit_code"] == 0
            if "physical_gpus" in receipt:
                assert receipt["physical_gpus"] == devices
            state["runs"].append(receipt)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, devices)),
                   PYTHONPATH=str(package), OMP_NUM_THREADS="4", MOE_EP_DISABLE_ARCHER_EVICT="1",
                   MOE_EP_NATIVE_NUMERICS="1", MOE_EP_SLOT_VIEWS="1")
        command = [args.python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                   str(package / "examples" / script), *shared, "--output", str(root / name), *extra]
        state.update(current=name, command=command, current_physical_gpus=devices)
        status.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"started": name, "unix": time.time()}), flush=True)
        began = time.time()
        with (root / f"{name}.log").open("a") as log:
            code = subprocess.call(command, cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT)
        run = dict(name=name, exit_code=code, started_unix=began, finished_unix=time.time(), physical_gpus=devices)
        state["runs"].append(run)
        print(json.dumps(run), flush=True)
        if code:
            state.update(status="FAIL")
            status.write_text(json.dumps(state, indent=2) + "\n")
            raise SystemExit(code)
        completed.write_text(json.dumps(run, indent=2) + "\n")
    state.update(status="TIMING_COMPLETE", finished_unix=time.time())
    status.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
