"""Validate cross-rank physical phase intervals from a separate diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
from pcie_host import write


def validate(root):
    groups=[]
    for repeat in (0,1):
        ranks=[json.loads((root/f'phases_repeat{repeat}_rank{r}.json').read_text()) for r in range(4)]
        count=len(ranks[0]);assert count and all(len(rows)==count for rows in ranks)
        quotas=[np.load(root/f'quota_repeat{repeat}_rank{r}.npy') for r in range(4)]
        policy=[np.load(root/f'policy_trace_repeat{repeat}_rank{r}.npy') for r in range(4)]
        for rank in range(1,4):
            np.testing.assert_array_equal(quotas[0],quotas[rank]);np.testing.assert_array_equal(policy[0],policy[rank])
        for event in range(count):
            rows=[ranks[r][event] for r in range(4)]
            assert all(row['event']==event and row['rank']==r for r,row in enumerate(rows))
            for row in rows:
                stages=['metadata_plan_complete','forward_a2a','forward_global_complete','h2d_only','h2d_global_complete','expert_compute','compute_global_complete','return_a2a','return_global_complete']
                assert all(row[name][0]<=row[name][1] for name in stages)
                assert all(row[a][1]<=row[b][0] for a,b in zip(stages,stages[1:]))
            assert max(row['forward_a2a'][1] for row in rows)<=min(row['h2d_only'][0] for row in rows),'G2G/H2D overlap'
            assert max(row['h2d_only'][1] for row in rows)<=min(row['expert_compute'][0] for row in rows),'H2D/compute overlap'
            assert max(row['expert_compute'][1] for row in rows)<=min(row['return_a2a'][0] for row in rows),'compute/return overlap'
            if event:
                assert max(ranks[r][event-1]['return_a2a'][1] for r in range(4))<=min(row['metadata_plan_complete'][0] for row in rows)
        pinned=[json.loads((root/f'pinned_rank{r}.json').read_text()) for r in range(4)]
        for left,right in ((0,1),(2,3)):
            assert pinned[left]['inode']==pinned[right]['inode'] and pinned[left]['device']==pinned[right]['device']
        assert all(p['bytes']==54*2**30 and p['pinned'] for p in pinned)
        results=[json.loads((root/f'repeat{repeat}_rank{r}.json').read_text()) for r in range(4)]
        assert all(r['finite_logits'] and r['validation']['status']=='PASS' and r['validation']['physical_slots']==1843 for r in results)
        assert all(r['validation']['scheduler']['background_copies']==0 for r in results)
        groups.append(dict(repeat=repeat,events=count,output_tokens=results[0]['output_tokens'],
                           forward_before_h2d=True,h2d_before_compute=True,compute_before_return=True,
                           plan_and_owner_rank_parity=True,quota_rank_parity=True))
    result=dict(status='PASS',diagnostic_root=str(root),scope='Cross-rank host intervals each end after CUDA physical synchronization; separate Gloo rendezvous; no NCCL during H2D',repeats=groups)
    write(root/'phase_validation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);validate(p.parse_args().root)
