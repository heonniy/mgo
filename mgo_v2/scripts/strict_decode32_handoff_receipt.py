import os,signal,time,json
from pathlib import Path
r=Path('/home/hwlee/mgo-results/strict_laca_headroom_20261006');pid=2363666
while True:
 p=r/'physical/S1_R8_B64_L512_7/status.json'
 if p.exists() and json.loads(p.read_text())['status']=='PASS':
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  assert b'run_strict_headroom_physical.py' in cmd
  os.kill(pid,signal.SIGSTOP)
  # The final S1 torchrun already exited before status PASS is written.
  assert all(json.loads(x.read_text())['status']=='PASS' for x in (r/'physical').glob('S1_*/status.json'))
  assert len(list((r/'physical').glob('S1_*/status.json')))==64
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  (r/'decode32_handoff.json').write_text(json.dumps(dict(status='S1_COMPLETE_S2_REPLACED',reason='Owner requested decode32 TPOT/E2E in final validation',old_driver_pid=pid,unix=time.time()),indent=2))
  break
 time.sleep(.1)
