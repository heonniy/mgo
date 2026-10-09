"""Diagnostic-only current-event inputs/state for fixed-miss Stage-II replay."""
import json
from pathlib import Path
import numpy as np


class DecisionCapture:
    def __init__(self,policy):
        self.policy=policy;self.events=[];self.inputs=[];self.states=[]

    def append(self,event,selected,weights,origins,gate):
        p=self.policy
        assert not p.substitution and np.all(p.birth==-1) and not np.any(p.lost)
        self.events.append(event)
        self.inputs.append(tuple(np.array(x,copy=True) for x in (selected,weights,origins,gate)))
        self.states.append(tuple(np.array(getattr(p,name),copy=True) for name in ('slots','last','gates','seen')))

    def save(self,path):
        offsets=np.r_[0,np.cumsum([len(x[0]) for x in self.inputs])].astype(np.int64)
        arrays=dict(events=np.array(self.events,np.int64),offsets=offsets,capacities=self.policy.capacities.copy())
        for i,name in enumerate(('selected','weights','origins')):arrays[name]=np.concatenate([x[i] for x in self.inputs])
        arrays['gate']=np.stack([x[3] for x in self.inputs])
        for i,name in enumerate(('slots','last','gates','seen')):arrays[name]=np.stack([x[i] for x in self.states])
        np.savez_compressed(path,**arrays)
        Path(str(path)+'.json').write_text(json.dumps(dict(status='PASS',events=len(self.events),source_policy=self.policy.policy,
          semantics='Current inputs and pre-admission cache state only; replay restores the same state for every placement arm; not an independent serving result',
          replica_state='OFF: lost false, birth -1, reuses zero; owner/primary uniquely derived from slots'),indent=2)+'\n')
