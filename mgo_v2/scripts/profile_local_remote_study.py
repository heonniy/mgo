#!/usr/bin/env python3
"""Run the two preselected posthoc repeats after the complete timing audit."""
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
    for name in ("root", "inputs", "model", "nsys", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    root, inputs, output = Path(args.root).resolve(), Path(args.inputs).resolve(), Path(args.output).resolve()
    package = Path(__file__).resolve().parents[1]
    assert json.loads((root / "status.json").read_text())["status"] == "TIMING_COMPLETE"
    assert json.loads((output / "validation.json").read_text())["status"] == "PASS"
    assert not (root / "profile.nsys-rep").exists() and not (root / "profile_receipts").exists()
    cells = [cell for cell in json.loads((root / "cells_r8.json").read_text())
             if cell["batch"] == 8 and cell["config"]["admission"] in ("random", "hungarian_same_path")]
    assert len(cells) == 2
    cell_file = root / "profile_cells.json"
    cell_file.write_text(json.dumps(cells, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", PYTHONPATH=str(package),
               OMP_NUM_THREADS="4", MOE_EP_DISABLE_ARCHER_EVICT="1", MOE_EP_NATIVE_NUMERICS="1",
               MOE_EP_SLOT_VIEWS="1", MGO_STACK_DUMP_SECONDS="0")
    command = [args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--cuda-event-trace=false", "--flush-on-cudaprofilerstop=false",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--output", str(root / "profile"),
               sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               str(package / "examples/profile_model_worker.py"), "--model", args.model,
               "--offload-dir", str(inputs / "expert_store"), "--similarity", str(inputs / "similarity.npy"),
               "--affinity", str(inputs / "affinity.npz"), "--workload", str(inputs / "screen_workload.json"),
               "--cells", str(cell_file), "--output", str(root / "profile_receipts"),
               "--steps", "65", "--repeats", "1", "--profile"]
    state = dict(status="RUNNING", phase="PROFILE", started_unix=time.time(), command=command,
                 nsys_version=subprocess.check_output([args.nsys, "--version"], text=True).strip(),
                 wrapper_sha256=hashlib.sha256((package / "examples/profile_model_worker.py").read_bytes()).hexdigest())
    path = root / "profile_status.json"
    def run(command, name):
        path.write_text(json.dumps(state, indent=2) + "\n")
        with (root / f"{name}.log").open("w") as log:
            code = subprocess.call(command, cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code:
            state.update(status="FAIL", exit_code=code)
            path.write_text(json.dumps(state, indent=2) + "\n")
            raise SystemExit(code)
    run(command, "profile")
    state.update(phase="EXPORT", profile_finished_unix=time.time())
    run([args.nsys, "export", "--type=sqlite", "--output", str(root / "profile.sqlite"), str(root / "profile.nsys-rep")], "profile_export")
    state.update(phase="AUDIT", export_finished_unix=time.time())
    run([sys.executable, str(package / "scripts/summarize_study_profiles.py"), "--sqlite", str(root / "profile.sqlite"),
         "--receipts", str(root / "profile_receipts"), "--baseline", str(root / "stage_b_r8"),
         "--output", str(output)], "profile_audit")
    state.update(status="PASS", phase="COMPLETE", finished_unix=time.time())
    path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
