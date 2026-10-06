"""One clean LA/Near policy pair, GPUs0,1,4,5 only."""
from pathlib import Path
import run_r4_br_near_h0 as base
if __name__=='__main__':
 base.ROOT=Path('/home/hwlee/mgo-results/r4_la_near_h0_20261007')
 base.PACKET=base.P/'experiments/r4_la_near_h0_20261007'
 base.WORKER=base.P/'examples/r4_la_near_h0_worker.py'
 base.POLICY_LABEL='LA Near';base.TIME_LIMIT=1800
 base.main()
