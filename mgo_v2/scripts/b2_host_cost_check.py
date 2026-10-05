"""Test owner-dependent host work counts; no fit to old policy TPOT.

BR prefix steps0-3 calibrate per-call host cost by exact expert row count;
steps4-7 of BR and FCA are held out. Then report old-capture delta predictions
as an exploratory check, not an automatic oracle authorization.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from prepare_critical_microbench import ROOT,PACKET,write
from critical_stage_b import sha
B2=ROOT/'b2'
def main():
 sources={};checks=[];models={};oldchecks=[];support={}
 ret_path=PACKET/'B2_RETURN_ATTRIBUTION.json';sources[str(ret_path)]=sha(ret_path);ret=json.loads(ret_path.read_text())['rows']
 for cache in ('C30','C60'):
  allcells={}
  for policy in ('BR','FCA'):
   ev=[]
   for rank in range(4):
    p=B2/cache/f'B2_SHARED_RETRY1_V3_OPT_PF_OVERLAP_{policy}_B128_H8'/f'b2_rank{rank}_analysis.json';sources[str(p)]=sha(p);ev.extend(json.loads(p.read_text())['events'])
   allcells[policy]=ev
  train=[r for r in allcells['BR'] if r['step']<4];byrows={};noncompiled=0;count=0
  for r in train:
   calls=r['expert_calls'];ph=r['phases'];noncompiled+=ph['expert_loop']['host_union_ms']-ph['expert_compiled_kernel']['host_union_ms']-ph.get('expert_ready_wait',{}).get('host_union_ms',0)
   count+=len(calls)
   for call in calls:byrows.setdefault(call['rows'],[]).append((call['host_end_raw_ns']-call['host_start_raw_ns'])/1e6)
  ns=sorted(byrows);cost=[float(np.median(byrows[n])) for n in ns];overhead=noncompiled/count
  models[cache]=dict(rows=ns,compiled_call_median_host_ms=cost,noncompiled_host_ms_per_expert=overhead,calibration='BR steps0-3 only; same-cache scope; no policy TPOT fitted',interpolation='linear row interpolation for unseen n; no coefficient retuning')
  lut=np.interp(np.arange(513),ns,cost)+overhead
  def pred(es):return float(lut[[n for e,n in es]].sum())
  for policy in ('BR','FCA'):
   held=[r for r in allcells[policy] if r['step']>=4];p=[pred([(c['expert'],c['rows']) for c in r['expert_calls']]) for r in held];o=[r['phases']['expert_loop']['host_union_ms']-r['phases'].get('expert_ready_wait',{}).get('host_union_ms',0) for r in held]
   checks.append(dict(cache=cache,policy=policy,heldout_rank_events=len(held),spearman=float(spearmanr(p,o).statistic),aggregate_error=float(abs(sum(p)-sum(o))/sum(o)),median_event_error=float(np.median(abs(np.array(p)-o)/o))))
  costs={}
  for policy in ('BR','FCA'):
   es={rank:sorted([e for e in allcells[policy] if e['rank']==rank and e['step']>=4],key=lambda e:e['event']) for rank in range(4)}
   ar=np.array([[pred([(z['expert'],z['rows']) for z in e['expert_calls']]) for e in es[rank]] for rank in range(4)])
   predicted=float((ar.max(0)-ar).sum()/4/4);observed=sum(z['arrival_wait_hat_ms'] for z in ret if z['cache']==cache and z['policy']==policy and z['event']>=240)/4/4
   costs[policy]=dict(predicted_wait_ms_per_step=predicted,observed_wait_ms_per_step=observed)
  delta=costs['FCA']['observed_wait_ms_per_step']-costs['BR']['observed_wait_ms_per_step'];pred_delta=costs['FCA']['predicted_wait_ms_per_step']-costs['BR']['predicted_wait_ms_per_step']
  support[cache]=dict(cells=costs,delta_fraction=pred_delta/delta,delta_relative_error=abs(pred_delta-delta)/abs(delta))
  predictions={};observations={}
  for policy in ('BR','OLD_CA','FCA','LA_CA'):
   p=ROOT/'stage_b'/f'{cache}_{policy}.json';sources[str(p)]=sha(p);rows=json.loads(p.read_text())['rows'];predicted=[max(pred(es) for es in r['executing_expert_rows_by_rank'])+r['pred_forward_ms']+r['pred_return_base_ms'] for r in rows];observed=[r['obs_critical_ms'] for r in rows];predictions[policy]=np.array(predicted);observations[policy]=np.array(observed)
  for policy in ('OLD_CA','FCA','LA_CA'):
   p=predictions[policy]-predictions['BR'];o=observations[policy]-observations['BR'];den=abs(o.sum())
   oldchecks.append(dict(cache=cache,policy=policy,predicted_delta_ms_per_step=float(p.sum()/64),observed_delta_ms_per_step=float(o.sum()/64),delta_spearman=float(spearmanr(p,o).statistic),aggregate_delta_relative_error=float(abs(p.sum()-o.sum())/den) if den else None))
 report=dict(status='DIAGNOSTIC_PREDICTABILITY_CHECK',models=models,prefix_heldout=checks,old_capture_exploratory_delta=oldchecks,source_sha256=sources,heldout_host_count_arrival_support=support,revised_delta_gate='FAIL: exploratory old-policy deltas fail<=25% error/correlation gates; not an authorized oracle cost model',interpretation='Owner-dependent expert-call count and row shapes are known before executing a layer. A stable measured host-call cost may be class-A work volume; variable host scheduling remains B. This finite check does not establish a server-independent constant or authorize Stage C. Exploratory old delta scores do not include a fitted H2D/TPOT coefficient.')
 write(PACKET/'B2_HOST_COST_PREDICTABILITY.json',report);print(json.dumps(dict(prefix=checks,old_delta=oldchecks),indent=2))
if __name__=='__main__':main()
