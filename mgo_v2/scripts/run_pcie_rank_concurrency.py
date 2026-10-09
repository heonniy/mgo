"""Pinned-host H2D bandwidth for every nonempty subset of requested GPUs.

Runs only after the Qwen cache baseline sweep completes. Own model loads on
0/1/4/5 are restored afterward; other GPU processes are never terminated.
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import run_full_pinned_r4 as owner


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/pcie_rank_concurrency_20261009"
BASELINES = Path(__file__).resolve().parents[1] / "experiments/qwen_cache_ablation_20261009/BASELINE_SWEEP_RESULTS.json"
GPUS = (0, 1, 3, 4, 5, 6, 7)
MIB = 1024 * 1024
SIZES = {"Qwen expert": 9 * MIB, "DeepSeek expert": int(16.5 * MIB)}


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def gpu_processes():
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    mapping = {u.strip(): int(g.strip()) for g, u in (x.split(",") for x in uuids.splitlines())}
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    return [(mapping[u.strip()], int(p.strip())) for u, p in (line.split(",") for line in raw.splitlines())]


def gpu_free():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(g.strip()): int(float(m.strip())) for g, m in (line.split(",") for line in raw.splitlines())}


def worker(gpu, commands, replies):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        os.sched_setaffinity(0, {gpu * 24})
        import torch

        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        sizes = sorted({int(v) for v in SIZES.values()})
        buffers = {}
        for size in sizes:
            source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            source.fill_(gpu + 1)
            dest = torch.empty(size, dtype=torch.uint8, device='cuda:0')
            buffers[size] = (source, dest)
        stream = torch.cuda.Stream(device=0)
        replies.put(dict(event="initialized", gpu=gpu, cpu_core=gpu * 24))
        while True:
            command = commands.get()
            if command["event"] == "stop":
                break
            size = int(command["bytes"])
            source, dest = buffers[size]
            if command["event"] == "prepare":
                with torch.cuda.stream(stream):
                    for _ in range(4):
                        dest.copy_(source, non_blocking=True)
                stream.synchronize()
                replies.put(dict(event="ready", gpu=gpu, key=command["key"]))
            elif command["event"] == "measure":
                start_at = int(command["start_at_ns"])
                while time.perf_counter_ns() < start_at - 1_000_000:
                    time.sleep(0.0002)
                while time.perf_counter_ns() < start_at:
                    pass
                beginning = torch.cuda.Event(enable_timing=True)
                ending = torch.cuda.Event(enable_timing=True)
                host_start = time.perf_counter_ns()
                with torch.cuda.stream(stream):
                    beginning.record(stream)
                    for _ in range(command["copies"]):
                        dest.copy_(source, non_blocking=True)
                    ending.record(stream)
                ending.synchronize()
                host_end = time.perf_counter_ns()
                seconds = beginning.elapsed_time(ending) / 1000
                replies.put(dict(event="done", gpu=gpu, key=command["key"],
                                 event_seconds=seconds, host_start_ns=host_start,
                                 host_end_ns=host_end,
                                 gib_per_s=(size * command["copies"] / 2**30) / seconds))
            else:
                raise ValueError(command)
    except BaseException:
        replies.put(dict(event="error", gpu=gpu, traceback=traceback.format_exc()))


def receive(replies, desired, key=None, timeout=120):
    received = {}
    deadline = time.monotonic() + timeout
    while len(received) < len(desired):
        row = replies.get(timeout=max(0.1, deadline - time.monotonic()))
        if row["event"] == "error":
            raise RuntimeError(f"GPU{row['gpu']} worker error: {row['traceback']}")
        if row["event"] not in ("initialized", "ready", "done"):
            raise RuntimeError(row)
        if row["gpu"] not in desired or (key is not None and row.get("key") != key):
            raise RuntimeError(f"Unexpected worker receipt: {row}")
        received[row["gpu"]] = row
    return received


def run(combinations, copies, repeats, result_path):
    context = mp.get_context("spawn")
    replies = context.Queue()
    queues = {g: context.Queue() for g in GPUS}
    processes = {g: context.Process(target=worker, args=(g, queues[g], replies)) for g in GPUS}
    for process in processes.values():
        process.start()
    try:
        receive(replies, set(GPUS), timeout=180)
        previous = json.loads(result_path.read_text()) if result_path.exists() else []
        complete = {(tuple(r["gpus"]), r["bytes"], r["repeat"]) for r in previous}
        for group in combinations:
            permitted = {process.pid for process in processes.values()}
            outsiders = [(g, pid) for g, pid in gpu_processes()
                        if g in GPUS and pid not in permitted]
            if outsiders:
                raise RuntimeError(f"Another process appeared on a target GPU: {outsiders}")
            for name, size in SIZES.items():
                for repeat in range(1, repeats + 1):
                    if (group, size, repeat) in complete:
                        continue
                    if owner.host_available() < 96 * 2**30:
                        raise RuntimeError("Host free memory below 96 GiB")
                    free = gpu_free()
                    if min(free[g] for g in GPUS) < 2048:
                        raise RuntimeError(f"GPU free memory below 2 GiB: {free}")
                    key = f"{'-'.join(map(str, group))}:{size}:{repeat}"
                    for g in group:
                        queues[g].put(dict(event="prepare", key=key, bytes=size))
                    receive(replies, set(group), key)
                    at = time.perf_counter_ns() + 150_000_000
                    for g in group:
                        queues[g].put(dict(event="measure", key=key, bytes=size,
                                           copies=copies, start_at_ns=at))
                    outcomes = receive(replies, set(group), key)
                    starts = [row["host_start_ns"] for row in outcomes.values()]
                    ends = [row["host_end_ns"] for row in outcomes.values()]
                    wall_seconds = (max(ends) - min(starts)) / 1e9
                    row = dict(gpus=list(group), concurrent_ranks=len(group),
                               payload=name, bytes=size, repeat=repeat, copies=copies,
                               rank_results={str(g): outcomes[g] for g in group},
                               start_skew_ms=(max(starts) - min(starts)) / 1e6,
                               wall_seconds=wall_seconds,
                               aggregate_gib_per_s=(size * copies * len(group) / 2**30) / wall_seconds)
                    previous.append(row)
                    write(result_path, previous)
                    print(key, f"aggregate {row['aggregate_gib_per_s']:.2f} GiB/s", flush=True)
        return previous
    finally:
        for queue in queues.values():
            queue.put(dict(event="stop"))
        for process in processes.values():
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def summarize(results):
    grouped = {}
    for row in results:
        key = (tuple(row["gpus"]), row["payload"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for (group, payload), samples in sorted(grouped.items(), key=lambda x: (len(x[0][0]), x[0][0], x[0][1])):
        rank = {}
        for g in group:
            values = [r["rank_results"][str(g)]["gib_per_s"] for r in samples]
            rank[str(g)] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
        agg = [r["aggregate_gib_per_s"] for r in samples]
        rows.append(dict(gpus=list(group), concurrent_ranks=len(group), payload=payload,
                         aggregate_gib_per_s=dict(median=statistics.median(agg),
                                                  minimum=min(agg), maximum=max(agg)),
                         rank_gib_per_s=rank,
                         max_start_skew_ms=max(r["start_skew_ms"] for r in samples),
                         repeats=len(samples)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    assert args.copies >= 32 and args.repeats in (2, 3)
    baseline = json.loads(BASELINES.read_text())
    assert sum(row.get("status") == "PASS" for row in baseline["rows"]) == 12, "baseline sweep still active"
    assert owner.host_available() >= 384 * 2**30
    EXPERIMENT.mkdir(exist_ok=True)
    all_groups = [group for count in range(1, len(GPUS) + 1)
                  for group in itertools.combinations(GPUS, count)]
    assert len(all_groups) == 127
    random.Random(20261009).shuffle(all_groups)
    stopped = []
    state = dict(status="RUNNING", gpus=GPUS, combinations=len(all_groups), started=time.time(),
                 source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=owner.P.parent, text=True).strip())
    write(EXPERIMENT / "STATUS.json", state)
    try:
        stopped = owner.stop_target_idle()
        deadline = time.monotonic() + 40
        occupants = [(g, pid) for g, pid in gpu_processes() if g in GPUS]
        while occupants and time.monotonic() < deadline:
            time.sleep(1)
            occupants = [(g, pid) for g, pid in gpu_processes() if g in GPUS]
        if occupants:
            raise RuntimeError(f"Target GPUs have active processes; none terminated: {occupants}")
        free = gpu_free()
        if any(free[g] < 2048 for g in GPUS):
            raise RuntimeError(f"GPU memory preflight: {free}")
        state["gpu_free_mib"] = {g: free[g] for g in GPUS}
        state["stopped_owner_loads"] = stopped
        write(EXPERIMENT / "STATUS.json", state)
        results = run(all_groups, args.copies, args.repeats, EXPERIMENT / "RAW.json")
        rows = summarize(results)
        assert len(rows) == 254
        write(EXPERIMENT / "RESULTS.json", dict(status="PASS", rows=rows,
                                                 count=len(rows), physical_gpus=GPUS,
                                                 pinned_host_source=True, repeats=args.repeats,
                                                 copies=args.copies))
        state["status"] = "PASS"
    except BaseException as error:
        state["status"] = "FAIL"
        state["error"] = repr(error)
        raise
    finally:
        state["finished"] = time.time()
        state["restored_owner_loads"] = owner.restore_target_idle(stopped)
        write(EXPERIMENT / "STATUS.json", state)


if __name__ == "__main__":
    main()
