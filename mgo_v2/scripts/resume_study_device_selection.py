#!/usr/bin/env python3
"""Finish the active R8 extension, then resume with the owner's R4 GPUs.

Only superseded scheduling parents are stopped. The active torchrun and all
GPU workers finish normally before their exit status and receipts are checked.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "inputs", "model"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    package = Path(__file__).resolve().parents[1]
    path = root / "r4_device_change.json"
    change = json.loads(path.read_text())
    original = change["original_parent_pid"]
    extension_parent = change["extension_parent_pid"]
    active = change["extension_torchrun_pid"]
    def fields(pid):
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    assert fields(original)[0] == fields(extension_parent)[0] == "T"
    while fields(active)[0] != "Z":
        time.sleep(5)
    assert int(fields(active)[49]) == 0, "R8 extension failed; leave R4 stopped"
    extension = json.loads((root / "extension_status.json").read_text())
    pause = json.loads((root / "scheduling_pause.json").read_text())
    assert fields(pause["active_torchrun_pid"])[0] == "Z"
    assert int(fields(pause["active_torchrun_pid"])[49]) == 0
    for directory, manifest in (("stage_b_r8", "cells_r8.json"),
                                ("stage_b_r8_extended", "cells_r8_extended.json")):
        for cell in json.loads((root / manifest).read_text()):
            for repeat in range(5):
                for rank in range(8):
                    row = json.loads((root / directory / f"{cell['name']}-rep{repeat}-rank{rank}.json").read_text())
                    assert row["status"] == "PASS"
    subprocess.run([sys.executable, "scripts/summarize_local_remote_study.py", "--root", str(root),
                    "--output", str(root / "partial"), "--partial"], cwd=package, check=True)
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    now = time.time()
    starts = [json.loads(line) for line in (root / "launcher.log").read_text().splitlines() if line.startswith("{")]
    started = next(row["unix"] for row in starts if row.get("started") == "stage_b_r8")
    runs = [dict(name="stage_b_r8", exit_code=0, started_unix=started,
                 finished_unix=extension["original_r8_process_finished_unix"], physical_gpus=list(range(8))),
            dict(name="stage_b_r8_extended", exit_code=0, started_unix=extension["started_unix"],
                 finished_unix=now, physical_gpus=list(range(8)))]
    for run in runs:
        run["completion_observation"] = "Successful torchrun zombie exit status and all rank receipts; scheduler was paused."
        (root / f"{run['name']}.complete.json").write_text(json.dumps(run, indent=2) + "\n")
    extension.update(status="PASS", exit_code=0, finished_unix=now,
                     scheduler_handoff="Replacement launcher preserves R8 completion and selects R4 physical GPUs 0,1,4,5")
    (root / "extension_status.json").write_text(json.dumps(extension, indent=2) + "\n")
    for pid, expected in ((original, "scripts/run_local_remote_study.py"),
                          (extension_parent, "scripts/extend_local_remote_study.py")):
        assert expected.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
        assert fields(pid)[0] == "T"
        os.kill(pid, signal.SIGKILL)  # only obsolete schedulers; all GPU children already exited
    pause.update(status="SUPERSEDED_FOR_R4_DEVICE_SELECTION", superseded_unix=now)
    (root / "scheduling_pause.json").write_text(json.dumps(pause, indent=2) + "\n")
    command = [sys.executable, "-u", "scripts/run_local_remote_study.py", "--root", str(root),
               "--inputs", args.inputs, "--model", args.model, "--resume", "--r4-gpus",
               *map(str, change["physical_gpus"])]
    change.update(status="R4_RUNNING", resumed_unix=time.time(), command=command)
    path.write_text(json.dumps(change, indent=2) + "\n")
    with (root / "r4_resumed_launcher.log").open("w") as log:
        code = subprocess.call(command, cwd=package, stdout=log, stderr=subprocess.STDOUT)
    change.update(status="PASS" if code == 0 else "FAIL", exit_code=code, finished_unix=time.time())
    path.write_text(json.dumps(change, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
