"""Pure CPU reuse accounting and an opt-in future-scoring cache observer."""
from bisect import bisect_right
from collections import Counter
import numpy as np
from replica_pareto_cpu import ReplicaReplay,ROW_BYTES,EXPERT_BYTES,traffic
HORIZONS=(1,2,4,'remaining')
THRESHOLDS=(0,65536,262144,1048576)

def marginal_pairs(origins,selected,destinations):
    """One candidate at a time; dispatch saved iff its old destination vanishes."""
    records={}
    for origin,experts,dests in zip(origins,selected,destinations):
        origin=int(origin);occupancy=Counter(map(int,dests))
        for expert,dest in zip(experts,dests):
            expert=int(expert);dest=int(dest)
            row=records.setdefault((expert,origin),dict(demand_routes=0,remote_routes=0,marginal_bytes=0))
            row['demand_routes']+=1
            if dest!=origin:
                row['remote_routes']+=1
                row['marginal_bytes']+=ROW_BYTES*(1+(occupancy[dest]==1))
    return records

def end_step(step,horizon):return 8 if horizon=='remaining' else min(8,step+horizon)
def next_distance(values,step):
    return next((j-step for j in range(step+1,9) if values[j]>0),None)
def burst(values,step):
    n=0
    for j in range(step,9):
        if values[j]<=0:break
        n+=1
    return n

def future_stats(timeline,step,horizon):
    indices=range(step+1,end_step(step,horizon)+1)
    return dict(future_peer_bytes_saved=sum(timeline['marginal_bytes'][i] for i in indices),
                reuse_steps=sum(timeline['remote_routes'][i]>0 for i in indices),
                reuse_routes=sum(timeline['remote_routes'][i] for i in indices),
                demand_steps=sum(timeline['demand_routes'][i]>0 for i in indices))

def future_demand(demand_indices,key,event,horizon):
    values=demand_indices.get(key,());pos=bisect_right(values,event)
    limit=431 if horizon=='remaining' else min(431,event+48*horizon)
    return pos<len(values) and values[pos]<=limit

def victim_penalty(replay,choice,event,horizon,demand_indices):
    victim=choice[1]
    return int(victim is not None and len(replay.owners[victim])==1 and future_demand(demand_indices,victim,event,horizon))

def adjusted_score(future_bytes,penalty,threshold,mode='fetch_normalized'):
    if mode=='fetch_normalized':score=future_bytes//(1+penalty)
    elif mode=='subtract_expert_bytes':score=future_bytes-penalty*EXPERT_BYTES
    else:raise ValueError(mode)
    return score if score>threshold else 0

class SelectiveReplay(ReplicaReplay):
    """All placement/LRU/traffic logic stays in ReplicaReplay; observe copy lives."""
    def __init__(self,capacities,timelines,demand_indices,horizon,threshold,penalty_mode):
        super().__init__(capacities,1)
        self.timelines=timelines;self.demand_indices=demand_indices;self.horizon=horizon;self.threshold=threshold;self.penalty_mode=penalty_mode
        self.event_index=-1;self.live={};self.lifetimes=[];self.pending_lost=set();self.victim_reloads=0;self.accepted_victim_penalties=0
    def score(self,replay,layer,expert,rank,choice,current_saving):
        # Identical F prefill; diagnostic replicas start at decode step 1.
        if self.event_index<48:return 0
        line=self.timelines.get((layer,expert,rank))
        if line is None:return 0
        score=future_stats(line,self.event_index//48,self.horizon)['future_peer_bytes_saved']
        penalty=victim_penalty(self,choice,self.event_index,self.horizon,self.demand_indices)
        return adjusted_score(score,penalty,self.threshold,self.penalty_mode)
    def finish_copy(self,rank,key,censored=False):
        record=self.live.pop((rank,key),None)
        if record:
            record.update(end_event=self.event_index,lifetime_layer_events=self.event_index-record['birth_event'],right_censored=censored)
            self.lifetimes.append(record)
    def place(self,rank,key,choice):
        replica=bool(self.owners.get(key));victim=choice[1]
        if victim is not None:
            self.finish_copy(rank,victim)
            if replica and len(self.owners[victim])==1:
                self.pending_lost.add(victim)
                self.accepted_victim_penalties+=victim_penalty(self,choice,self.event_index,self.horizon,self.demand_indices)
        if not replica and key in self.pending_lost:
            self.victim_reloads+=1;self.pending_lost.remove(key)
        super().place(rank,key,choice)
        if replica:self.live[rank,key]=dict(rank=rank,layer=key[0],expert=key[1],birth_event=self.event_index,reused=False,reuse_events=0)
    def step(self,event_index,layer,origins,selected):
        self.event_index=event_index
        result=super().event(layer,origins,selected,candidate_score=self.score)
        served={(int(dst),(layer,int(e))) for origin,es,ds in zip(origins,selected,result[1]) for e,dst in zip(es,ds) if int(dst)==int(origin)}
        for token in served:
            record=self.live.get(token)
            if record and event_index>record['birth_event']:
                record['reused']=True;record['reuse_events']+=1
        return result
    def finish(self):
        self.event_index=432
        for rank,key in list(self.live):self.finish_copy(rank,key,True)
        return self.lifetimes
