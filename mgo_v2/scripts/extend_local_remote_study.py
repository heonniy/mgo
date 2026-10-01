#!/usr/bin/env python3
"""Insert user-requested R8/B16,B32 cells before the paused launcher's R4 jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ("root", "inputs", "model"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    root, inputs = Path(args.root).resolve(), Path(args.inputs).resolve()
    package = Path(__file__).resolve().parents[1]
    pause = json.loads((root / "scheduling_pause.json").read_text())
    parent, active = pause["parent_pid"], pause["active_torchrun_pid"]
    path = root / "extension_status.json"
    state = dict(status="WAITING_FOR_CURRENT_R8", requested_batches=[16, 32], world=8,
                 repeats=5, decode_steps=64, parent_pid=parent, active_torchrun_pid=active,
                 requested_unix=time.time())
    path.write_text(json.dumps(state, indent=2) + "\n")
    base = dict(global_cache_ratio=.3, substitution_enabled=True, eviction="coverage",
                same_layer_alpha=.25, path_eta=.5, seed=42)
    policies = [("C0_balanced_random", "random"), ("C1_hungarian_current", "hungarian_current"),
                ("C2_hungarian_same_path", "hungarian_same_path")]
    cells = [dict(name=f"b{batch}_{name}", batch=batch, config=dict(base, admission=policy))
             for batch in (16, 32) for name, policy in policies]
    manifest = root / "cells_r8_extended.json"
    manifest.write_text(json.dumps(cells, indent=2) + "\n")
    while True:
        stat_path = Path(f"/proc/{active}/stat")
        if not stat_path.exists():
            raise RuntimeError("Original torchrun vanished before its exit status could be audited")
        fields = stat_path.read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            assert int(fields[49]) == 0, f"Original R8 torchrun failed: {fields[49]}"
            break
        time.sleep(5)
    state["original_r8_process_finished_unix"] = time.time()
    original = json.loads((root / "cells_r8.json").read_text())
    for cell in original:
        for repeat in range(5):
            for rank in range(8):
                receipt = json.loads((root / "stage_b_r8" / f"{cell['name']}-rep{repeat}-rank{rank}.json").read_text())
                assert receipt["status"] == "PASS"
    active_gpus = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    assert not active_gpus, "GPU processes remain before extension: " + active_gpus
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               str(package / "examples/benchmark_model.py"), "--model", args.model,
               "--offload-dir", str(inputs / "expert_store"), "--similarity", str(inputs / "similarity.npy"),
               "--affinity", str(inputs / "affinity.npz"), "--workload", str(inputs / "screen_workload.json"),
               "--cells", str(manifest), "--output", str(root / "stage_b_r8_extended"),
               "--steps", "65", "--repeats", "5"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", PYTHONPATH=str(package), OMP_NUM_THREADS="4",
               MOE_EP_DISABLE_ARCHER_EVICT="1", MOE_EP_NATIVE_NUMERICS="1", MOE_EP_SLOT_VIEWS="1")
    state.update(status="RUNNING", started_unix=time.time(), command=command)
    path.write_text(json.dumps(state, indent=2) + "\n")
    with (root / "stage_b_r8_extended.log").open("w") as log:
        code = subprocess.call(command, cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT)
    state.update(status="PASS" if code == 0 else "FAIL", exit_code=code, finished_unix=time.time())
    path.write_text(json.dumps(state, indent=2) + "\n")
    if code:
        raise SystemExit(code)  # leave R4 paused until the failed extension is resolved
    pause.update(status="RESUMED_AFTER_R8_EXTENSION", resumed_unix=time.time())
    (root / "scheduling_pause.json").write_text(json.dumps(pause, indent=2) + "\n")
    os.kill(parent, signal.SIGCONT)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
