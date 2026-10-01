"""Structured, grep-able logging for the NUMA / cold-start init path.

When a multi-process cold-start init flakes (socket handshake race, memfd
sharing failure, archer_index flock wait, cudaHostRegister OOM, NCCL
collective timeout), the logs alone should pinpoint:
  (a) which rank / role / NUMA,
  (b) which stage,
  (c) how long it took, or *where it hung*, and
  (d) the actionable error.

Every line is single-line and tagged ``[mi-init]`` so ``grep mi-init`` over
the combined stdout/stderr of all ranks reconstructs the whole cross-process
timeline.  The key diagnostic property: a stage that logs ``BEGIN`` but never
``END``/``FAIL`` is exactly the stage the process stalled in.  Sort the grep
output by the wall-clock field to interleave ranks.

Line format::

    [mi-init 12:34:56.789 t=+  12.34s INFO pid=1234 rank=2/4 lr=0 numa=1 role=FOLLOWER] stage=numa_shm.connect msg attempt=7 waited=1.4s

Toggle: ``MOE_EP_INIT_LOG=0`` silences everything (default on).
"""
from __future__ import annotations

import contextlib
import os
import sys
import time

# Per-process reference clock.  ``t=+N`` is seconds since this module was
# imported, which happens very early (before ``import moe_infinity``), so it
# approximates time-since-process-start.  Use it to read off stage durations;
# use the wall-clock field to align lines across ranks.
_T0 = time.monotonic()
_ENABLED = os.environ.get("MOE_EP_INIT_LOG", "1") != "0"


def _ctx(**extra) -> str:
    fields = {
        "pid": os.getpid(),
        "rank": f"{os.environ.get('RANK', '?')}/{os.environ.get('WORLD_SIZE', '?')}",
        "lr": os.environ.get("LOCAL_RANK", "?"),
        "numa": os.environ.get("MOE_EP_NUMA_NODE", "?"),
    }
    # Caller-supplied fields (role, numa override, sizes, ...) win over the
    # env-derived defaults so a stage can report what it actually knows.
    fields.update({k: v for k, v in extra.items() if v is not None})
    return " ".join(f"{k}={v}" for k, v in fields.items())


def log(stage: str, msg: str = "", *, level: str = "INFO", **fields) -> None:
    """Emit one structured init-log line.  Errors go to stderr, rest stdout."""
    if not _ENABLED:
        return
    now = time.time()
    wall = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
    head = (f"[mi-init {wall} t=+{time.monotonic() - _T0:7.2f}s "
            f"{level} {_ctx(**fields)}]")
    line = f"{head} stage={stage}"
    if msg:
        line += f" {msg}"
    stream = sys.stderr if level in ("ERROR", "FATAL") else sys.stdout
    print(line, file=stream, flush=True)


def error(stage: str, msg: str = "", **fields) -> None:
    log(stage, msg, level="ERROR", **fields)


def _read_cgroup_current_bytes() -> int:
    """cgroup v2 memory.current in bytes, or -1 if unreadable."""
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return -1


def proc_mem(tag: str, **fields) -> None:
    """Log this process's locked/pinned/resident memory + cgroup current.

    Diagnostic for the pinned/NUMA-shm residue hypothesis: call right before
    a hard exit (os._exit skips atexit, so this is the only graceful hook) to
    record how much this rank is holding at teardown.
      * VmLck  — mlock'd pages (host_caching_allocator / pinned pool)
      * VmPin  — pinned pages (cudaHostRegister'd / DMA-pinned)
      * VmRSS  — resident set; VmHWM — peak resident
    A large VmLck/VmPin at exit that the cgroup doesn't shed afterward is the
    smoking gun for cross-run residue.
    """
    if not _ENABLED:
        return
    want = ("VmRSS", "VmHWM", "VmLck", "VmPin", "VmSwap")
    vals = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k = line.split(":", 1)[0]
                if k in want:
                    vals[k] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    cg = _read_cgroup_current_bytes()
    cg_gb = f"{cg / (1 << 30):.2f}GB" if cg >= 0 else "?"
    mem = " ".join(f"{k}={vals.get(k, '?')}" for k in want)
    log("proc_mem", f"{tag} {mem} cgroup.current={cg_gb}", **fields)


@contextlib.contextmanager
def stage(name: str, **fields):
    """Bracket an init step.

    Logs ``BEGIN`` on entry, then ``END elapsed=...`` on success or
    ``FAIL after ...`` (on stderr) if the body raises.  A lone ``BEGIN`` in
    the logs is the stage that hung.  ``fields`` (role, numa, sizes, ...) are
    attached to every line of the bracket.
    """
    t = time.monotonic()
    log(name, "BEGIN", **fields)
    try:
        yield
    except BaseException as e:
        log(name, f"FAIL after {time.monotonic() - t:.2f}s: "
                  f"{type(e).__name__}: {e}", level="ERROR", **fields)
        raise
    else:
        log(name, f"END elapsed={time.monotonic() - t:.2f}s", **fields)
