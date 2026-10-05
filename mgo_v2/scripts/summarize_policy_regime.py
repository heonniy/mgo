"""Audit all raw repeats before reporting within-capacity policy gains."""
import json,statistics,hashlib
from pathlib import Path
from run_refactor_measure import PACKET
from policy_regime_paths import ROOT,PACKET,ENVIRONMENT
POLICIES=('BR','OLD_CA','FCA','LA_CA')
def main():
 assert json.loads((ROOT/'status.json').read_text())['status']=='TIMING_COMPLETE'
 rows=[];sources={};identities=[]
 def read(p):
  raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 for setting in ('C30','C60'):
  base=ROOT/setting/f'POLICY_REGIME_{setting}_B128_H64';state=read(base/'status.json');result=read(base/'result.json')
  assert state['status']==result['status']=='PASS';assert state['group']['gpus']==[0,1,4,5]
  identities.append(state['common_stack']);samples=result['pairs'];assert len(samples) in (2,3)
  for repeat,sample in enumerate(samples,1):
   assert set(sample)==set(POLICIES)
   for p in POLICIES:
    ranks=[read(base/f'{p}_r{repeat}_measure_rank{r}.json') for r in range(4)]
    for r,x in enumerate(ranks):
     assert x['status']=='PASS' and x['rank']==r and x['repeat']==repeat and x['no_compile_in_measure']
     assert x['case']['policy']==p and x['case']['horizon']==64 and x['case']['runtime_arm']=='V3_OPT_PF_OVERLAP'
     placement=x['thread_placement'];team=placement['fixed_team']
     assert team['stage_mask']==[team['staging_cpu']] and team['helper_mask']==[team['helper_cpu']]
     assert len(set(placement['main']+[team['staging_cpu'],team['helper_cpu']]))==3
    for metric in ('TPOT','E2E_wall'):assert max(x[metric] for x in ranks)==sample[p][metric]
  reducer=statistics.mean if len(samples)==2 else statistics.median
  by_policy={}
  for p in POLICIES:
   metrics={}
   for metric in ('TPOT','E2E_wall'):
    values=[s[p][metric] for s in samples]
    metrics[metric]=dict(value=reducer(values),mean=statistics.mean(values),median=statistics.median(values),range=[min(values),max(values)],samples=values,spread_relative=(max(values)-min(values))/statistics.mean(values))
   by_policy[p]=dict(metrics=metrics,excluded_repeats=[],estimator='mean' if len(samples)==2 else 'median',noise_evidence='No independent invalidation established; retain all observations.')
  for p in POLICIES:
   by_policy[p]['TPOT_gain']=1-by_policy[p]['metrics']['TPOT']['value']/by_policy['BR']['metrics']['TPOT']['value']
   if p!='BR':by_policy[p]['paired_gate']=result['gates'][p]
  rows.append(dict(setting=setting,policies=by_policy,unstable=result['unstable'],raw_pairs=samples))
 for key in ('code','inputs','environment','affinity_sha256','predictor_sha256','precision'):assert identities[0][key]==identities[1][key],key
 report=dict(status='TIMING_AUDITED',rows=rows,source_sha256=sources,common_inputs_and_stack='PASS',scope='R4 0/1/4/5, local B128, decode64, V3 P2/T2, BF16. Cache groups separate; no excluded repeats or latency-based noise selection. Mechanism attribution is separate and pending until profile audit completes.')
 out=PACKET/'POLICY_REGIME_TIMING_RESULTS.json';out.write_text(json.dumps(report,indent=2)+'\n')
 for row in rows:
  print(row['setting'],'unstable',row['unstable'])
  for p,x in row['policies'].items():print(p,x['metrics']['TPOT']['value'],x['TPOT_gain'])
if __name__=='__main__':main()
