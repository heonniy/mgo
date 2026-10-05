"""Recompute paired evidence; reject missing arms, noise and gain-only domination."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from adaptive_timing import paired_decision
from prepare_refactor_arms import ARMS
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'

def evaluate(records,horizon=256,world=8):
 rows=[]
 for arm in ARMS:
  batches={}
  for batch in (128,256):
   d=records[arm][batch];assert d['status']=='PASS' and d['baseline']=='BR' and d['candidate']=='LA'
   assert d['physical_arena_count']==1
   cases=d['cases'];assert [c['policy'] for c in cases]==['BR','LA']
   assert all(c['runtime_arm']==arm and c['horizon']==horizon and c['partial_precision']=='bf16' for c in cases)
   for k in ('P','trigger','horizon','overlap','partial_precision'):assert cases[0][k]==cases[1][k]
   assert cases[0]['overlap']==(arm==ARMS[2]);assert (cases[0]['P']==0)==(arm==ARMS[0])
   gate=paired_decision(d['pairs']);assert gate['complete'],'incomplete adaptive repetitions'
   batches[str(batch)]=dict(gate=gate,samples=d['pairs'],LA_TPOT=statistics.mean(p['LA']['TPOT'] for p in d['pairs']),BR_TPOT=statistics.mean(p['BR']['TPOT'] for p in d['pairs']))
  rows.append(dict(arm=arm,batches=batches,stable=all(not b['gate']['unstable'] for b in batches.values()),positive_supported=all(b['gate']['gain']['TPOT']['positive_supported'] for b in batches.values()),score=min(b['gate']['gain']['TPOT']['estimate'] for b in batches.values()),mean_gain=statistics.mean(b['gate']['gain']['TPOT']['estimate'] for b in batches.values())))
 # Prefetch and trigger choices cannot change after seeing LA timings.
 tuning={(records[a][b]['cases'][0]['P'],records[a][b]['cases'][0]['trigger']) for a in ARMS[1:] for b in (128,256)}
 assert len(tuning)==1,'V2/V3 or batch-specific retuning'
 for row in rows:
  row['dominated_by']=[]
  for other in rows:
   if other is row or not other['stable']:continue
   ratios=[other['batches'][b]['LA_TPOT']/row['batches'][b]['LA_TPOT'] for b in ('128','256')]
   if max(ratios)<=1 and min(ratios)<.98:row['dominated_by'].append(other['arm'])
  row['eligible']=row['stable'] and not row['dominated_by']
 eligible=[r for r in rows if r['eligible']]
 ranked=sorted(eligible,key=lambda r:(-r['score'],-r['mean_gain'],sum(b['LA_TPOT'] for b in r['batches'].values())))
 return dict(world=world,horizon=horizon,status='TIMING_CANDIDATE' if ranked else 'NO_STABLE_CANDIDATE',winner=ranked[0]['arm'] if ranked else None,improvement_supported=bool(ranked and ranked[0]['positive_supported']),rows=rows,selection_scope='BF16 common-stack V1/V2/V3 only; final selection pending profiling and secondary CA',gain_estimator='1 - exp(mean(log(TPOT_LA/TPOT_BR))); paired 95% t intervals; every valid sample retained')

def main(a):
 records={};sources={};identities={}
 for arm in ARMS:
  records[arm]={}
  for batch in (128,256):
   path=a.root/f'{a.stage_prefix}_{arm}_B{batch}_H{a.horizon}/result.json';raw=path.read_bytes();records[arm][batch]=json.loads(raw);sources[str(path)]=hashlib.sha256(raw).hexdigest()
   state_path=path.with_name('status.json');state_raw=state_path.read_bytes();state=json.loads(state_raw);assert state['status']=='PASS';assert state['group'].get('world',8)==a.world;identities[(arm,batch)]=state['common_stack'];sources[str(state_path)]=hashlib.sha256(state_raw).hexdigest()
 from refactor_fingerprint import assert_equivalent
 assert_equivalent(identities)
 out=evaluate(records,a.horizon,a.world);out['source_hashes']=sources;out['common_stack_equivalence']='PASS: source/input/environment fingerprints match across arms'
 (PACKET/(a.output_prefix+'THREE_ARM_RESULTS.json')).write_text(json.dumps(out,indent=2)+'\n')
 lines=['# BF16 three-arm paired timing','',f'Status: {out["status"]}. Timing candidate: {out["winner"]}. Positive LA gain supported in both batches: {out["improvement_supported"]}.','',out['selection_scope'], '', '| Arm | Batch | BR TPOT (s) | LA TPOT (s) | Paired LA gain | 95% interval | Repeats | Eligible |','|---|---:|---:|---:|---:|---|---:|---|']
 for r in out['rows']:
  for batch,b in r['batches'].items():
   g=b['gate']['gain']['TPOT'];lo,hi=g['CI95']
   lines.append(f'| {r["arm"]} | {batch} | {b["BR_TPOT"]:.6f} | {b["LA_TPOT"]:.6f} | {g["estimate"]:.2%} | [{lo:.2%}, {hi:.2%}] | {len(b["samples"])} | {r["eligible"]} |')
 lines+=['', '| Arm | Batch | BR E2E (s) | LA E2E (s) | Paired E2E gain | 95% interval |', '|---|---:|---:|---:|---:|---|']
 for r in out['rows']:
  for batch,b in r['batches'].items():
   g=b['gate']['gain']['E2E_wall'];lo,hi=g['CI95'];br=statistics.mean(p['BR']['E2E_wall'] for p in b['samples']);la=statistics.mean(p['LA']['E2E_wall'] for p in b['samples'])
   lines.append(f'| {r["arm"]} | {batch} | {br:.6f} | {la:.6f} | {g["estimate"]:.2%} | [{lo:.2%}, {hi:.2%}] |')
 lines+=['', 'BR/LA absolute times are arithmetic means. Relative gains use paired log ratios and therefore need not equal the ratio of the displayed absolute means. JSON retains E2E, all raw pairs, full ranges, stability decisions, and exclusions.']
 (PACKET/(a.output_prefix+'THREE_ARM_RESULTS.md')).write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--horizon',type=int,default=256);p.add_argument('--world',type=int,default=8);p.add_argument('--output-prefix',default='');p.add_argument('--stage-prefix',default='M15');main(p.parse_args())
