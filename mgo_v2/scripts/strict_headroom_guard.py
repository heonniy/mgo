"""Resource checks outside primary timing windows."""
import subprocess,time
from strict_headroom_common import ROOT
import run_timing_stability as h

def guard(gpus,pid=None,initial=False):
 row=h.sample(pid);row['gpus']=[g for g in row['gpus'] if g['gpu'] in gpus]
 ids={u.strip():int(i) for i,u in [x.split(',') for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()]}
 apps=[x.split(',') for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()]
 relevant={int(p) for u,p in apps if ids[u.strip()] in gpus}
 row['foreign_pids']=[p for p in row['foreign_pids'] if p in relevant]
 assert not row['foreign_pids'],f'foreign GPU processes: {row["foreign_pids"]}'
 assert row['host_available_bytes']>=(768 if initial else 256)*2**30,'host memory guard'
 assert len(row['gpus'])==len(gpus)
 assert all(g['free_mib']>(76000 if initial else 8192) and g['temperature_c']<(65 if initial else 85) for g in row['gpus']),'GPU memory/temperature guard'
 assert not (ROOT/'STOP').exists(),'owner STOP'
 return row

def cooldown(gpus):
 end=time.monotonic()+180
 while any(g['temperature_c']>=65 for g in h.snapshot()['gpus'] if g['gpu'] in gpus) and time.monotonic()<end:time.sleep(5)
 return guard(gpus,initial=True)
