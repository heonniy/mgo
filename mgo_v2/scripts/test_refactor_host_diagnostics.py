"""Check nested/thread separation and GC lifecycle for diagnostic accounting."""
import gc
import threading
from refactor_host_diagnostics import HostDiagnostics

def main():
 d=HostDiagnostics(lambda:96)
 with d.phase('disabled'):pass
 assert not d.rows
 d.start()
 with d.phase('parent'):
  with d.phase('child'):sum(range(200))
 def worker():
  with d.phase('worker'):sum(range(200))
 t=threading.Thread(target=worker);t.start();t.join()
 gc.collect()
 result=d.finish();rows={r['phase']:r for r in result['phases']}
 assert set(rows)=={'parent','child','worker'}
 for metric in ['wall','cpu']:
  assert rows['parent'][f'exclusive_{metric}_ns']==rows['parent'][f'inclusive_{metric}_ns']-rows['child'][f'inclusive_{metric}_ns']
 assert rows['worker']['tid']!=rows['parent']['tid']
 assert all(r['observed_step']==1 and r['calls']==1 for r in rows.values())
 assert result['gc'] and d.gc_callback not in gc.callbacks and not d.enabled
 print('PASS nested exclusive accounting, separate thread ranges, step scope, GC callback cleanup')
if __name__=='__main__':main()
