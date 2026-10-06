"""Final selected-seed TTFT, wall TPOT and E2E with separate phase accounting."""
from strict_headroom_common import *
import statistics

def main():
 rows=[]
 for d in sorted((ROOT/'physical').glob('S2D32_*')):
  state=json.loads((d/'status.json').read_text());assert state['status']=='PASS';x=state['result'];world=x['spec']['world'];diag={};copies={}
  for policy in ('BR','LA_CA_NEAR','LA'):
   ranks=[json.loads((d/f'{policy}_diagnostic_rank{r}.json').read_text()) for r in range(world)]
   assert all(z['status']=='PASS' and len(z['phases'])==1584 for z in ranks)
   diag[policy]={}
   for phase in ('prefill','decode'):
    diag[policy][phase]={m:statistics.mean(sum(e[m] for e in r['phases'] if e['phase']==phase) for r in ranks) for m in ['controller_wall_ms','routing_admission_layout_ms','h2d_local_ms','h2d_barrier_ms','forward_cuda_ms','expert_cuda_ms','compute_local_complete_ms','compute_barrier_ms','return_cuda_ms']}
   copies[policy]={phase:[r['validation'][phase+'_H2D_copies'] for r in ranks] for phase in ('prefill','decode')}
  rows.append(dict(**x,diagnostic_mean_rank_ms=diag,copies_by_rank=copies,classification={p:'STRICT_PLACEMENT' if sum(copies[p]['decode'])==sum(copies['BR']['decode']) else 'PLACEMENT_STATE_EFFECT' for p in ['LA_CA_NEAR','LA']},ranges={p:{m:[min(y[m] for y in v),max(y[m] for y in v)] for m in ['TTFT','TPOT','E2E']} for p,v in x['samples'].items()}))
 assert len(rows)==8
 write(PACKET/'DECODE32_RESULTS.json',dict(status='PASS',rows=rows,scope='S1 TTFT-selected BR-adversarial triples. Same policy for prefill/decode; cache carryover means TPOT is not an isolated decode-policy effect. Common frozen BR token/route trace. Prefill first output plus32 decode outputs. No latency sample deletion. Diagnostic phase times are separate-pass rank means, not additive global critical path.'))
 lines=['# Strict selected-seed TTFT + decode32','','Fresh final validation with the same policy in prefill and decode. BR-generated routing and teacher tokens are common to all policies.32 decode forward steps follow the first prefill output (33 outputs total). TTFT, TPOT and E2E are wall times; per-metric maximum rank durations are aggregated. No independent maximum-headroom claim for TPOT: seeds were selected using TTFT.','','| R | B | Input | Policy | TTFT s | TPOT s/token | E2E s | TTFT gain | TPOT gain | E2E gain | Any unstable |','|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
 for x in rows:
  for p in ('BR','LA_CA_NEAR','LA'):
   e=x['estimates'][p];g=x['gains'].get(p,dict(TTFT=0,TPOT=0,E2E=0));lines.append(f"| {x['spec']['world']} | {x['spec']['batch']} | {x['spec']['context']} | {p} | {e['TTFT']:.4f} | {e['TPOT']:.4f} | {e['E2E']:.4f} | {100*g['TTFT']:+.2f}% | {100*g['TPOT']:+.2f}% | {100*g['E2E']:+.2f}% | {any(x['unstable'][p].values())} |")
 lines+=['','Two repeats initially; at most one third if the maximum first-two difference across policies/metrics is >2% and <=5%. If any exceeds5%, flag instability without further repeats. Full ranges and samples remain in JSON. Decode copy differences are classified as placement-state effects, not pure placement. Same BR baseline is paired with both candidates in a common three-policy process, with order reversal between repeats.','']
 (PACKET/'DECODE32_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
