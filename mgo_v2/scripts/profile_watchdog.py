"""Cooperative diagnostic stack snapshots; never installed in primary timing."""
import json
import sys
import threading
import time


def start_watchdog(path, interval=60):
    stop = threading.Event()
    def monitor():
        while not stop.wait(interval):
            rows = []
            for tid, frame in sys._current_frames().items():
                stack = []
                for _ in range(32):
                    if frame is None:
                        break
                    code = frame.f_code
                    stack.append(dict(file=code.co_filename, function=code.co_name, line=frame.f_lineno))
                    # Do not traverse synthetic torch.compile trampoline frames.
                    if code.co_filename == '<shim>' or code.co_name == '<interpreter trampoline>':
                        break
                    frame = frame.f_back
                rows.append(dict(thread=tid, stack=stack))
            with path.open('a') as output:
                output.write(json.dumps(dict(unix=time.time(), threads=rows,
                                             scope='Cooperative Python snapshots require the GIL; absence is not proof of native progress.'))+'\n')
    threading.Thread(target=monitor, name='profile-stack-watchdog', daemon=True).start()
    return stop
