"""Validate and join diagnostic-only native assignments to exact live traces."""
import csv
import json
from pathlib import Path
import numpy as np


def assignments(root,trace):
    root=Path(root);events=len(trace)
    with np.load(root/'actual_fetches_repeat-1.npz') as capture:
        fetches=capture['fetches']
        assert int(capture['events'])==events
        assert capture['columns'].tolist()==['event','rank','key','logical_main_slot','victim','replica','physical_slot']
    assert fetches.ndim==2 and fetches.shape[1]==7
    assert np.all(fetches[:,0]>=0) and np.all(fetches[:,0]<events)
    assert np.all(np.diff(fetches[:,0])>=0),'Assignments not ordered by event'
    assert np.all(fetches[:,1]>=0) and np.all(fetches[:,1]<4)
    assert np.all(fetches[:,5]==0)
    assert len(np.unique(fetches[:,0]*6144+fetches[:,2]))==len(fetches)
    counts=np.zeros((events,4),np.int64)
    np.add.at(counts,(fetches[:,0],fetches[:,1]),1)
    np.testing.assert_array_equal(counts,trace[:,48:52])
    assert np.all(fetches[:,2]//128==fetches[:,0]%48)
    capacities=np.array([459,459,459,458],np.int64)
    assert np.all(fetches[:,3]>=0) and np.all(fetches[:,3]<capacities[fetches[:,1]])
    assert np.all(fetches[:,6]>=0) and np.all(fetches[:,6]<capacities[fetches[:,1]]+2)
    with (root/'assigned_experts.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['event','layer','phase','rank','physical_gpu','expert_id','key','logical_main_slot','physical_slot','victim','replica'])
        for event,rank,key,slot,victim,rep,physical in fetches:
            writer.writerow([event,event%48,'prefill' if event<48 else 'decode',rank,[0,1,4,5][rank],key%128,key,slot,physical,victim,rep])
    offsets=np.r_[0,np.cumsum(counts.sum(axis=1))]
    details=[json.dumps(fetches[offsets[i]:offsets[i+1],1:].tolist(),separators=(',',':')) for i in range(events)]
    path=root/'quota_event_trace.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=list(reader.fieldnames);rows=list(reader)
    fields=[f for f in fields if f not in ('assigned_experts','assignment_provenance')]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields+['assigned_experts','assignment_provenance']);writer.writeheader()
        for row in rows:
            row['assigned_experts']=details[int(row['event'])]
            row['assignment_provenance']='actual separate diagnostic; same tokens/cache/full trace as this repeat; [rank,key,logical_slot,victim,replica,physical_slot]'
            writer.writerow(row)
    with (root/'phase_timing.jsonl').open('w') as stream:
        for rank in range(4):
            rows=json.loads((root/f'phases_repeat-1_rank{rank}.json').read_text());assert len(rows)==events
            for row in rows:
                stream.write(json.dumps(dict(row,physical_gpu=[0,1,4,5][rank],diagnostic=True))+'\n')
    return dict(status='PASS',events=events,fetches=len(fetches),every_rank_quota_exact=True,
                no_replication=True,legal_slots=True,actual_diagnostic_assignments=True,
                scope='Diagnostic copies excluded from primaries; exact per-arm token/cache/full policy trace must be verified before joining',
                diagnostic_overhead='Assignment array copies contribute to PLAN residual')
