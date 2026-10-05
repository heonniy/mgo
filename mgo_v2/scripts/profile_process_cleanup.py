"""Track owned diagnostic descendants even when Nsight creates new sessions."""
import os
import signal
import time
from pathlib import Path


def identity(pid):
    path = Path('/proc') / str(pid)
    try:
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def track(root, known):
    pending = [root]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stamp = identity(pid)
        if stamp is None:
            continue
        known[pid] = stamp
        try:
            for task in (Path('/proc') / str(pid) / 'task').iterdir():
                pending.extend(map(int, (task / 'children').read_text().split()))
        except OSError:
            pass


def terminate(known):
    def alive():
        return [pid for pid, stamp in known.items() if identity(pid) == stamp]
    remaining = alive()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in remaining:
            if identity(pid) == known[pid]:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while remaining and time.monotonic() < deadline:
            time.sleep(.1)
            remaining = alive()
    if remaining:
        raise RuntimeError(f'Diagnostic descendants remain alive: {remaining}')
