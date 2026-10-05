"""Audit explicit reused/new paths before ranking the bounded stable search."""
import json,hashlib
from pathlib import Path
from prepare_refactor_arms import ARMS,PACKET
from refactor_fingerprint import assert_equivalent
from summarize_refactor_pairs import evaluate

def main():
 manifest=json.loads((PACKET/'STABLE_SEARCH_MANIFEST.json').read_text())
 records={};identities={};sources={};receipts=0
 def read(path):
  raw=path.read_bytes();sources[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 for arm in ARMS:
  records[arm]={}
  for batch in (128,256):
   root=Path(manifest['arms'][arm][str(batch)]);state=read(root/'status.json');result=read(root/'result.json')
   assert state['status']==result['status']=='PASS'
   g=state['group'];assert (g['world'],g['gpus'],g['batch'],g['horizon'])==(4,[0,1,4,5],batch,64)
   assert g['cases']==result['cases']
   for case in g['cases']:
    for key in ('unique_combine','async_metadata_inputs','isolated_cpu_threads','fixed_staging_team'):assert case[key] is True
    assert case['staging_backend']=='torch'
   for repeat,pair in enumerate(result['pairs'],1):
    for policy in ('BR','LA'):
     xs=[]
     for rank in range(4):
      x=read(root/f'{policy}_r{repeat}_measure_rank{rank}.json');receipts+=1;xs.append(x)
      assert x['status']=='PASS' and x['rank']==rank and x['repeat']==repeat and x['no_compile_in_measure'] is True
      assert x['case']==next(c for c in g['cases'] if c['policy']==policy)
      assert x['unique_combine_layers']==64*48 and x['async_metadata_inputs'] is True
      placement=x['thread_placement'];team=placement['fixed_team'];assert placement['isolated'] is True
      assert team['stage_mask']==placement['staging']==[team['staging_cpu']] and team['helper_mask']==[team['helper_cpu']]
      assert len(set(placement['main']+[team['staging_cpu'],team['helper_cpu']]))==3
     for metric in ('E2E_wall','TPOT'):assert max(x[metric] for x in xs)==pair[policy][metric]
   records[arm][batch]=result;identities[(arm,batch)]=state['common_stack']
 assert_equivalent(identities)
 report=evaluate(records,64,4);report.update(source_hashes=sources,validated_rank_receipts=receipts,common_stack_equivalence='PASS',reused_paths=manifest['reused'],selection_rule='Existing frozen evaluator: maximize minimum paired LA gain across B128/B256 among stable, non-dominated arms. Reject arms with no-better absolute LA TPOT in either batch and >2% worse in at least one than another stable arm. No tuning after observing LA.',fixed_options=dict(staging_backend='torch',unique_combine=True,async_metadata_inputs=True,isolated_cpu_threads=True,fixed_staging_team=True),scope='Optimal only among these three planned arms, at frozen BR-tuned P/trigger. Profiling and secondary CA/default adoption remain separate. Stable two-pair outcomes are not repeated for significance.')
 out=PACKET/'STABLE_THREE_ARM_RESULTS.json';out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('status','winner','improvement_supported')},indent=2))
 for x in report['rows']:
  print(x['arm'],'stable',x['stable'],'score',x['score'],'dominated_by',x['dominated_by'])
if __name__=='__main__':main()
