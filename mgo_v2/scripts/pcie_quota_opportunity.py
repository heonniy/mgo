"""Count-only quota opportunity on recorded serving misses; no speed prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pcie_host import write

def analyze(root,out):
    report=dict(status='PASS',scope='Count-only same-M counterfactual, not measured H2D or TPOT speedup',arms={})
    for arm in ('R-NEAR','G-NEAR'):
        path=root/arm/'policy_trace_repeat1_rank0.npy';trace=np.load(path)
        assert trace.shape==(3072,61) and np.array_equal(trace[:,52],np.arange(3072))
        m=trace[48:,19].astype(np.int64)
        r=2*(m//4)+np.minimum(m%4,2);g=(m+1)//2
        records=[]
        for value in np.unique(m):
            critical_r=int(2*(value//4)+min(value%4,2));critical_g=int((value+1)//2)
            records.append(dict(M=int(value),events=int((m==value).sum()),
                         rank_order_critical_group_fetches=critical_r,balanced_critical_group_fetches=critical_g,
                         count_only_reduction_percent=(critical_r-critical_g)/max(1,critical_r)*100))
        report['arms'][arm]=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 decode_events=len(m),mean_M=float(m.mean()),M_mod4_equals2_fraction=float((m%4==2).mean()),
                 count_only_average_critical_group_reduction_percent=float(np.mean((r-g)/np.maximum(1,r)))*100,
                 miss_histogram_and_count_opportunity=records)
    report['notes']=['Both quotas receive each observed event\'s SAME M. Future cache differences between R/G are not assigned to this proxy.',
       'M=6: critical-group fetch count 4 to 3, 25% fewer; M=54: 28 to 27, 3.57% fewer.',
       'Group-level critical count differs only when M mod 4 = 2. Equal counts do not imply equal physical bandwidth.',
       'Use measured H2D completion and full TPOT for performance conclusions; do not multiply this proxy by phase share as measured speedup.']
    write(out,report);print(json.dumps(dict(out=str(out),arms={a:r['count_only_average_critical_group_reduction_percent'] for a,r in report['arms'].items()})))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.out)
