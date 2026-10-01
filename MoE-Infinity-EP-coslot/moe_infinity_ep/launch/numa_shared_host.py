"""NUMA-shared pinned host memory coordination.

Architecture:
  Per NUMA, one process (leader = lowest local_rank in the NUMA) creates a
  memfd-backed shared anonymous region, mmaps it, and cudaHostRegisters it.
  The other ranks on the same NUMA (followers) connect to the leader's Unix
  socket, receive the memfd via SCM_RIGHTS, mmap it (gets the same physical
  pages), and cudaHostRegister their own view.

  Both leader and follower set env vars so archer's C++ HostMemoryPool
  bump-allocates into the shared region. Leader's archer SetModuleMemoryFromDisk
  populates the bytes; follower skips the disk read.

Why memfd + Unix socket fd-passing:
  * /dev/shm has a small default size (2 GB on this box) that we cannot
    remount-resize without root.
  * memfd_create gives anonymous, RAM-backed shared memory with no tmpfs
    size limit; the size is gated only by the cgroup memory cap.
  * cudaHostRegister works on memfd-mapped regions (verified by prototype).
  * SCM_RIGHTS lets the leader pass an open fd to other processes on the
    same machine; the receiver's mmap of that fd shares physical pages with
    the leader (kernel dedup, no copy).

Verified pre-prototype that all of the above functions correctly across two
processes targeting different GPUs on the same NUMA.
"""
from __future__ import annotations

import array
import ctypes
import errno
import mmap
import os
import socket
import struct
import sys
import time
from typing import Optional

from . import init_logging as ilog

_GB = 1 << 30

# libc.memfd_create
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_LIBC.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
_LIBC.memfd_create.restype = ctypes.c_int

# libcudart for cudaHostRegister/Unregister
def _load_cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12",
                 "libcudart.so.11"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libcudart not found")

_CUDART = _load_cudart()
_CUDART.cudaHostRegister.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
]
_CUDART.cudaHostRegister.restype = ctypes.c_int
_CUDART.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_CUDART.cudaHostUnregister.restype = ctypes.c_int
_CUDART.cudaGetErrorString.argtypes = [ctypes.c_int]
_CUDART.cudaGetErrorString.restype = ctypes.c_char_p

# CUDA register flag constants (from cuda_runtime.h):
#   cudaHostRegisterDefault   = 0x00
#   cudaHostRegisterPortable  = 0x01   ← memory pinned in every CUDA context
#   cudaHostRegisterMapped    = 0x02   ← also map into device address space
#   cudaHostRegisterIoMemory  = 0x04
#   cudaHostRegisterReadOnly  = 0x08
#
# Production decision (see feedback_numa_shm_register.md): EVERY rank
# (leader AND followers) registers its own view.  cgroup overhead is
# negligible (~0.6%) and follower-skip drops H2D to the pageable slow path
# (verified: 47-96% GPU util split → 90-92% symmetric after every-rank).
# The Portable flag is kept defensively but is NOT what makes the shared
# region work — process-local cudaHostRegister calls are required.
CUDA_HOST_REGISTER_PORTABLE = 0x01
CUDA_HOST_REGISTER_MAPPED = 0x02

_PAGE_SIZE = os.sysconf("SC_PAGESIZE")


class NumaSharedRegion:
    """Holds a process's view of the per-NUMA shared pinned region."""

    def __init__(self, numa_id: int, size: int, is_leader: bool,
                 sock_path: str, mm: mmap.mmap, addr: int, fd: int):
        self.numa_id = numa_id
        self.size = size
        self.is_leader = is_leader
        self.sock_path = sock_path
        self.mm = mm
        self.addr = addr
        self.fd = fd
        self._registered = False

    def cuda_register(self) -> None:
        # Every rank registers its own view (see feedback memory).  CUDA's
        # registration table is process-local, so without this call a
        # process's H2D copies fall back to pageable bounce-buffer.
        #
        # This is a prime cold-start failure point: pinning a 200GB+ region
        # can take seconds and can fail (OOM / RLIMIT_MEMLOCK), and a leader
        # that dies here strands every follower on the socket.  Time it and
        # log the outcome so a slow/failed register is unmistakable.
        flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED
        _role = "LEADER" if self.is_leader else "FOLLOWER"
        ilog.log("numa_shm.cuda_register", "BEGIN", role=_role,
                 numa=self.numa_id, addr=hex(self.addr),
                 size_gb=f"{self.size / _GB:.2f}", flags=f"0x{flags:x}")
        _t = time.monotonic()
        code = _CUDART.cudaHostRegister(self.addr, self.size, flags)
        if code != 0:
            msg = _CUDART.cudaGetErrorString(code).decode()
            ilog.error("numa_shm.cuda_register",
                       f"cudaHostRegister FAILED after "
                       f"{time.monotonic() - _t:.2f}s: code={code} ({msg}). "
                       f"Likely cgroup/RLIMIT_MEMLOCK pressure on a "
                       f"{self.size / _GB:.1f}GB region — followers will now "
                       f"strand on the socket. Check cgroup memory.max and "
                       f"`ulimit -l`.",
                       role=_role, numa=self.numa_id)
            raise RuntimeError(
                f"cudaHostRegister(NUMA {self.numa_id}, size={self.size}, "
                f"flags=0x{flags:x}) returned {code}: {msg}"
            )
        self._registered = True
        ilog.log("numa_shm.cuda_register",
                 f"END elapsed={time.monotonic() - _t:.2f}s", role=_role,
                 numa=self.numa_id)

    def export_to_env(self) -> None:
        os.environ["MOE_INFINITY_SHM_BASE_PTR"] = str(self.addr)
        os.environ["MOE_INFINITY_SHM_SIZE"] = str(self.size)
        os.environ["MOE_INFINITY_NUMA_FOLLOWER"] = (
            "0" if self.is_leader else "1"
        )

    def close(self) -> None:
        # NOTE: this is registered via atexit in entry.py, but the launch
        # scripts (scripts/m*.py) terminate with os._exit(0), which SKIPS
        # atexit. So in the normal exit path this method does NOT run — the
        # absence of these logs is itself the diagnostic signal that graceful
        # cudaHostUnregister/munmap never happened and we rely entirely on
        # kernel reclaim at process death.
        _role = "LEADER" if self.is_leader else "FOLLOWER"
        ilog.log("numa_shm.close", "BEGIN graceful cleanup", role=_role,
                 numa=self.numa_id, addr=hex(self.addr),
                 size_gb=f"{self.size / _GB:.2f}", fd=self.fd)
        if self._registered:
            code = _CUDART.cudaHostUnregister(self.addr)
            if code != 0:
                msg = _CUDART.cudaGetErrorString(code).decode()
                ilog.error("numa_shm.close",
                           f"cudaHostUnregister returned {code} ({msg})",
                           role=_role, numa=self.numa_id)
            else:
                ilog.log("numa_shm.close", "cudaHostUnregister ok",
                         role=_role, numa=self.numa_id)
            self._registered = False
        try:
            self.mm.close()
        except Exception as e:
            ilog.error("numa_shm.close", f"mm.close failed: {e!r}",
                       role=_role, numa=self.numa_id)
        try:
            os.close(self.fd)
        except Exception as e:
            ilog.error("numa_shm.close", f"os.close(fd) failed: {e!r}",
                       role=_role, numa=self.numa_id)
        ilog.log("numa_shm.close", "END", role=_role, numa=self.numa_id)


def _round_up_page(n: int) -> int:
    return (n + _PAGE_SIZE - 1) & ~(_PAGE_SIZE - 1)


def _memfd_create(name: str, size: int) -> int:
    fd = _LIBC.memfd_create(name.encode(), 0)
    if fd < 0:
        err = ctypes.get_errno()
        ilog.error("numa_shm.memfd_create",
                   f"memfd_create({name}) failed: {os.strerror(err)}")
        raise OSError(err, f"memfd_create({name}): {os.strerror(err)}")
    # ftruncate only reserves address space; physical pages are allocated
    # lazily on first write (leader's disk populate) — an OOM here is rare,
    # but a failure to size the memfd is fatal to the whole NUMA group.
    try:
        os.ftruncate(fd, size)
    except OSError as e:
        ilog.error("numa_shm.memfd_create",
                   f"ftruncate(fd={fd}, size={size}) failed: {e!r}")
        raise
    ilog.log("numa_shm.memfd_create", "ok", fd=fd,
             size_gb=f"{size / _GB:.2f}")
    return fd


def _mmap_fd(fd: int, size: int):
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    addr = ctypes.addressof(ctypes.c_byte.from_buffer(mm))
    return mm, addr


def _send_fd(conn: socket.socket, fd: int) -> None:
    conn.sendmsg(
        [b"x"],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))],
    )


def _recv_fd(conn: socket.socket, timeout_s: float = 60.0) -> int:
    # M14 fix (2026-05-27): apply socket-level timeout. Previously recvmsg
    # was a blocking call with no timeout — if the leader process died
    # AFTER accept() but BEFORE sendmsg() (e.g. cudaHostRegister OOM on the
    # leader), follower would block here indefinitely until NCCL watchdog
    # fired ~120 minutes later. Now follower sees a clean socket.timeout
    # → RuntimeError, surfacing the leader failure.
    conn.settimeout(timeout_s)
    ilog.log("numa_shm.recv_fd", "BEGIN waiting for SCM_RIGHTS fd",
             timeout_s=timeout_s)
    _t = time.monotonic()
    try:
        _msg, ancdata, _flags, _addr = conn.recvmsg(
            1, socket.CMSG_LEN(4)
        )
    except socket.timeout as e:
        ilog.error("numa_shm.recv_fd",
                   f"recvmsg TIMED OUT after {timeout_s}s — leader accepted "
                   f"but never sent the fd. Leader likely died mid-setup "
                   f"(cudaHostRegister OOM is the usual cause). Check the "
                   f"leader rank's mi-init log for a numa_shm.* FAIL.")
        raise RuntimeError(
            f"_recv_fd: recvmsg timed out after {timeout_s}s — leader may "
            f"have died after accept(). Check leader's log for "
            f"setup_numa_shared errors."
        ) from e
    except (ConnectionResetError, BrokenPipeError) as e:
        ilog.error("numa_shm.recv_fd",
                   f"leader closed connection unexpectedly: {e!r} — leader "
                   f"died during shm setup; check its mi-init log.")
        raise RuntimeError(
            f"_recv_fd: leader closed connection unexpectedly: {e!r}.  "
            f"Likely leader died during shm setup; check its log."
        ) from e
    arr = array.array("i")
    for level, ctype, cdata in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            arr.frombytes(cdata[: arr.itemsize])
            break
    if len(arr) == 0:
        ilog.error("numa_shm.recv_fd",
                   "recvmsg returned no fd via SCM_RIGHTS (ancillary data "
                   "missing) — kernel/socket-buffer issue.")
        raise RuntimeError("recvmsg returned no fd via SCM_RIGHTS")
    ilog.log("numa_shm.recv_fd",
             f"END received fd={arr[0]} elapsed={time.monotonic() - _t:.2f}s")
    return arr[0]


def setup_numa_shared(numa_id: int, size: int, is_leader: bool,
                      sock_path: str, follower_count: int,
                      timeout_s: float = 120.0) -> NumaSharedRegion:
    """Set up the NUMA-shared region for this process.

    Leader: creates memfd, listens on sock_path, sends fd to each follower
            that connects.
    Follower: connects to sock_path, receives fd.

    Both: mmap + cudaHostRegister + export env.

    Deterministic-allocation invariant (DO NOT regress):
        Both leader and follower must invoke archer's sparse_node allocation
        sequence in the same order so the per-process atomic bump_offset
        in memory_pool.cpp lands each tensor at the same shm offset.  Any
        path that lets one process call AllocateMemory in a different order
        than the other corrupts the shared region.  Today this is enforced
        by archer's deterministic sparse_nodes iteration during
        SetModuleMemoryFromDisk; if a future change introduces parallel
        allocation, switch bump_offset to a memfd-resident atomic.
    """
    size = _round_up_page(size)
    _role = "LEADER" if is_leader else "FOLLOWER"
    ilog.log("numa_shm.setup", "BEGIN", role=_role, numa=numa_id,
             size_gb=f"{size / _GB:.2f}", sock=sock_path,
             followers=follower_count if is_leader else "n/a")
    _setup_t = time.monotonic()

    if is_leader:
        fd = _memfd_create(f"moe_numa{numa_id}", size)
        # Listen for followers. Background thread accepts and sends fd.
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(sock_path)
        sock.listen(follower_count)
        ilog.log("numa_shm.listen", f"listening, expecting {follower_count} "
                 f"follower(s)", role=_role, numa=numa_id, sock=sock_path)

        # Accept-thread state is surfaced to the leader so a stuck follower
        # surfaces an error in the leader's logs instead of dying silently.
        accept_state = {"remaining": follower_count, "errors": []}

        def _accept_loop():
            sock.settimeout(timeout_s)
            while accept_state["remaining"] > 0:
                try:
                    conn, _ = sock.accept()
                    _send_fd(conn, fd)
                    conn.close()
                    accept_state["remaining"] -= 1
                    ilog.log("numa_shm.accept",
                             f"follower connected, sent fd "
                             f"({follower_count - accept_state['remaining']}"
                             f"/{follower_count}), "
                             f"{accept_state['remaining']} remaining",
                             role=_role, numa=numa_id)
                except socket.timeout:
                    accept_state["errors"].append(
                        f"timeout after {timeout_s}s with "
                        f"{accept_state['remaining']} follower(s) still missing"
                    )
                    ilog.error("numa_shm.accept",
                               f"accept TIMED OUT after {timeout_s}s — "
                               f"{accept_state['remaining']}/{follower_count} "
                               f"follower(s) never connected. They likely "
                               f"died before reaching setup_numa_shared "
                               f"(check their mi-init logs).",
                               role=_role, numa=numa_id)
                    break
                except Exception as e:
                    accept_state["errors"].append(f"accept: {e}")
                    ilog.error("numa_shm.accept",
                               f"accept loop crashed: {e!r}",
                               role=_role, numa=numa_id)
                    break
            try:
                sock.close()
            except Exception:
                pass
            if accept_state["errors"]:
                print(f"[numa_shared_host] leader accept ended with errors: "
                      f"{accept_state['errors']}", flush=True)

        import threading
        threading.Thread(target=_accept_loop, daemon=True).start()

        mm, addr = _mmap_fd(fd, size)
        ilog.log("numa_shm.mmap", f"mmap'd memfd at {hex(addr)}",
                 role=_role, numa=numa_id)
    else:
        # Wait for leader to bring up socket. The leader may still be in its
        # own memfd/listen setup (or slow to schedule), so retry until the
        # deadline. Progress is logged every ~2s so a long wait is visible
        # rather than looking like a hang.
        deadline = time.time() + timeout_s
        last_err = None
        conn = None
        _connect_t = time.monotonic()
        _attempt = 0
        _next_report = 2.0
        ilog.log("numa_shm.connect", "BEGIN connecting to leader socket",
                 role=_role, numa=numa_id, sock=sock_path, timeout_s=timeout_s)
        while time.time() < deadline:
            _attempt += 1
            try:
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.connect(sock_path)
                break
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                conn = None
                _waited = time.monotonic() - _connect_t
                if _waited >= _next_report:
                    ilog.log("numa_shm.connect",
                             f"still waiting for leader socket "
                             f"({type(e).__name__})",
                             role=_role, numa=numa_id, attempt=_attempt,
                             waited=f"{_waited:.1f}s")
                    _next_report += 2.0
                time.sleep(0.2)
        if conn is None:
            ilog.error("numa_shm.connect",
                       f"could not connect to leader socket {sock_path} "
                       f"within {timeout_s}s ({_attempt} attempts): "
                       f"{last_err!r}. Leader never created the socket — it "
                       f"likely crashed in its own setup (check leader's "
                       f"mi-init log).",
                       role=_role, numa=numa_id)
            raise RuntimeError(
                f"follower: could not connect to leader socket "
                f"{sock_path}: {last_err}"
            )
        ilog.log("numa_shm.connect",
                 f"END connected after {_attempt} attempt(s), "
                 f"waited={time.monotonic() - _connect_t:.2f}s",
                 role=_role, numa=numa_id)
        fd = _recv_fd(conn, timeout_s=timeout_s)
        conn.close()
        mm, addr = _mmap_fd(fd, size)
        ilog.log("numa_shm.mmap", f"mmap'd received memfd at {hex(addr)}",
                 role=_role, numa=numa_id)

    region = NumaSharedRegion(
        numa_id=numa_id, size=size, is_leader=is_leader,
        sock_path=sock_path, mm=mm, addr=addr, fd=fd,
    )
    # === DO NOT roll back to leader-only register. ==========================
    # Every rank (leader AND followers) MUST call cudaHostRegister on the
    # same memfd-backed region. This is a deliberate, measurement-backed
    # decision recorded in feedback_numa_shm_register.md (auto-memory).
    #
    # Original misdiagnosis (v3/v4/v5, 2026-05-26): we believed cgroup
    # memory.max was being exceeded because each rank's cudaHostRegister
    # counted as a separate N× shm charge. So we made the call leader-only,
    # which silently dropped follower H2D copies onto the pageable bounce-
    # buffer slow path — measurable as asymmetric GPU util (96% leader vs
    # ~50% follower) in v6/v7 production runs.
    #
    # Verified later the same day with scripts/test_cgroup_count.py at
    # production scale (444 GB shm × 4 ranks):
    #   leader-only register : cgroup current = 452.95 GB
    #   all-4-ranks register : cgroup current = 455.54 GB
    #   delta                : +2.59 GB total (~0.86 GB/extra rank, 0.6%)
    # cgroup v2 charges shared anonymous pages O(1) per physical region;
    # subsequent register calls only add a small per-process CUDA metadata
    # cost. The real v3/v4/v5 OOM came from sparse-load transient buffers
    # + per-process anonymous workload, NOT from cudaHostRegister.
    #
    # Production v9 (2026-05-26) confirmed: every-rank register restores
    # symmetric GPU util (90-92% on all four GPUs) without OOM.
    #
    # Companion verification scripts (kept for documentation):
    #   /home/work/hyewon.lee/실험/main_exp/scripts/test_pinned_share.py
    #   /home/work/hyewon.lee/실험/main_exp/scripts/test_cgroup_count.py
    region.cuda_register()
    region.export_to_env()
    # 2026-05-27: comprehensive startup log — every detail needed to
    # diagnose a follower-handshake failure or memfd-sharing issue.
    print(
        f"[numa_shared_host] pid={os.getpid()} numa={numa_id} "
        f"role={'LEADER' if is_leader else 'FOLLOWER'} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
        f"global_rank={os.environ.get('RANK', '?')} "
        f"addr={hex(addr)} size={size} bytes ({size/_GB:.2f} GB) "
        f"sock={sock_path} "
        f"followers_expected={follower_count if is_leader else 'n/a'} "
        f"cuda_register=ok env=MOE_INFINITY_SHM_BASE_PTR,SHM_SIZE,NUMA_FOLLOWER",
        flush=True,
    )
    ilog.log("numa_shm.setup",
             f"END region ready at {hex(addr)} "
             f"elapsed={time.monotonic() - _setup_t:.2f}s",
             role=_role, numa=numa_id, size_gb=f"{size / _GB:.2f}")

    # Optional cross-process verification — confirms leader & follower really
    # share the same physical pages.  Gated on env to avoid 1-page write+read
    # overhead in production.  Procedure:
    #   1. Leader writes a known 16-byte signature at offset 0 + size//2.
    #   2. Both ranks sleep briefly to let leader's write propagate.
    #   3. Each rank reads both offsets and checks the signature.
    # Mismatch on follower means the memfd fd-passing failed silently OR
    # the mmap didn't share pages (e.g. MAP_PRIVATE by mistake).
    if os.environ.get("MOE_EP_VERIFY_NUMA_SHM", "0") == "1":
        try:
            _verify_shm_sharing(region, is_leader)
        except Exception as e:
            print(f"[numa_shared_host] verification FAILED pid={os.getpid()} "
                  f"role={'leader' if is_leader else 'follower'}: {e}",
                  flush=True)
            raise

    return region


_VERIFY_SIG_LEADER = b"NUMA-SHM-LEADER!"   # 16 bytes
_VERIFY_OFFSETS = (0, 1 << 20)              # offset 0 + 1 MiB


def _verify_shm_sharing(region: "NumaSharedRegion", is_leader: bool) -> None:
    """Cross-process check that leader's writes are visible to followers.

    Side effect: writes 16 bytes at offsets 0 and 1 MiB.  Both offsets are
    BEFORE archer's sparse load starts allocating (since sparse_nodes
    iteration begins later), so these bytes will be overwritten by the
    real disk data — verification leaves no residue.
    """
    import ctypes as _ct
    base = _ct.cast(region.addr, _ct.POINTER(_ct.c_ubyte))
    if is_leader:
        for off in _VERIFY_OFFSETS:
            for i, b in enumerate(_VERIFY_SIG_LEADER):
                base[off + i] = b
        print(f"[numa_shared_host:verify] LEADER pid={os.getpid()} "
              f"wrote signature at offsets {_VERIFY_OFFSETS}", flush=True)
    else:
        # Follower waits briefly for leader's write to be visible.
        # Cache coherence on x86 makes this near-instant, but give some
        # slack so the verify is not racy.
        time.sleep(0.5)
        for off in _VERIFY_OFFSETS:
            got = bytes(bytearray(base[off + i] for i in range(
                len(_VERIFY_SIG_LEADER))))
            if got != _VERIFY_SIG_LEADER:
                raise RuntimeError(
                    f"FOLLOWER pid={os.getpid()} read at offset {off}: "
                    f"got {got!r}, expected {_VERIFY_SIG_LEADER!r}.  "
                    f"shm-shared memfd handshake produced a NON-SHARED "
                    f"mapping (process saw private pages).  Check "
                    f"setup_numa_shared follower path: SCM_RIGHTS fd "
                    f"transfer + mmap MAP_SHARED.")
        print(f"[numa_shared_host:verify] FOLLOWER pid={os.getpid()} "
              f"read leader signature at offsets {_VERIFY_OFFSETS} ✓",
              flush=True)


def compute_sparse_bytes_for_model(num_layers: int, num_experts: int,
                                   hidden: int, intermediate: int,
                                   dtype_bytes: int = 2) -> int:
    """Total bytes archer's sparse cache will hold across all experts."""
    return num_layers * num_experts * 3 * intermediate * hidden * dtype_bytes
