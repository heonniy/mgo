"""Freeze robust seed-median stress winners, top three, and neutral references."""
import json,csv,statistics
from ca_stress_common import *
def table(name,rows):
 with (PACKET/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 retained=json.loads((ROOT/'retained.json').read_text());assert len(retained)==512
 groups={};all_proofs=[];replays=0
 pool_sizes={name:json.loads((ROOT/(name+'_requests.json')).read_text())['pool_size'] for name in ['MATH','ShareGPT']}
 for job in retained:
  key=f"{job['dataset']}_R{job['world']}_B{job['batch']}_s{job['sample_seed']}_d{job['dp_seed']}";path=ROOT/'candidates'/(key+'.json');data=json.loads(path.read_text());assert data['status']=='PASS'
  assert data['manifest_sha256']==digest(data['manifest']);m=data['manifest'];assert len(set(m['request_ids']))==m['world']*m['batch'] and all(len(r)==m['batch'] for r in m['ranks'])
  all_proofs.append(dict(path=str(path),sha256=sha(path)))
  for cell in data['results']:
   ca=cell['CA'];br=cell['BR'];assert [r['seed'] for r in br]==SEEDS
   for result in [ca,*br]:assert result['quota_max_minus_min']<=1 and result['full']['peak_resident_copies']<=result['cache_capacity'];replays+=1
   peer_ca=ca['full']['peer_bytes'];absolute=[r['full']['peer_bytes']-peer_ca for r in br];relative=[g/r['full']['peer_bytes'] for g,r in zip(absolute,br)]
   best=max(range(8),key=lambda i:(absolute[i],relative[i],-br[i]['seed']))
   row=dict(dataset=m['dataset'],R=m['world'],local_B=m['batch'],cache_percent=cell['cache'],candidate_pool_size=pool_sizes[m['dataset']],global_N=m['world']*m['batch'],sample_seed=m['sample_seed'],dp_seed=m['dp_seed'],manifest_sha256=data['manifest_sha256'],request_subset_sha256=data['request_subset_sha256'],median_gain_abs_bytes=statistics.median(absolute),median_gain_rel=statistics.median(relative),min_gain_abs_bytes=min(absolute),max_gain_abs_bytes=max(absolute),min_gain_rel=min(relative),max_gain_rel=max(relative),best_observed_br_seed=br[best]['seed'],best_gain_abs_bytes=absolute[best],best_gain_rel=relative[best],BR_peer_bytes_median=statistics.median(r['full']['peer_bytes'] for r in br),CA_peer_bytes=peer_ca,BR_H2D_bytes_median=statistics.median(r['full']['H2D_bytes'] for r in br),CA_H2D_bytes=ca['full']['H2D_bytes'],BR_exact_global_hit_rate_median=statistics.median(r['full']['exact_global_hits_fraction'] for r in br),CA_exact_global_hit_rate=ca['full']['exact_global_hits_fraction'],BR_exact_local_hit_rate_median=statistics.median(r['full']['exact_local_hits_fraction'] for r in br),CA_exact_local_hit_rate=ca['full']['exact_local_hits_fraction'],max_event_quota_imbalance=max(r['quota_max_minus_min'] for r in [ca,*br]),CA_final_state_sha256=ca['final_state_sha256'],BR_final_state_hashes=json.dumps({r['seed']:r['final_state_sha256'] for r in br},sort_keys=True),candidate_receipt=str(path),candidate_receipt_sha256=sha(path))
   groups.setdefault((m['dataset'],m['world'],m['batch'],cell['cache']),[]).append((row,m))
 assert len(groups)==64 and replays==18432
 winners=[];top3=[];neutral=[]
 for key,candidates in sorted(groups.items()):
  assert len(candidates)==32
  candidates.sort(key=lambda item:(-item[0]['median_gain_abs_bytes'],-item[0]['median_gain_rel'],item[0]['CA_peer_bytes'],item[0]['sample_seed'],item[0]['dp_seed']))
  row,manifest=candidates[0];winners.append(row)
  name=f'{key[0]}_R{key[1]}_B{key[2]}_c{key[3]}'
  write(PACKET/'selected_manifests'/(name+'.json'),dict(**manifest,cache_percent=key[3],manifest_sha256=row['manifest_sha256'],pool_manifest_sha256=sha(ROOT/(key[0]+'_requests.json')),interpretation='optimized communication stress, not dataset average'))
  for position,(r,m) in enumerate(candidates[:3],1):top3.append(dict(position=position,**r))
  old={policy:SOURCE/'cells'/f'{key[0]}_h256_R{key[1]}_B{key[2]}_c{key[3]}_gate_s0_{policy}_seed42.json' for policy in ['BR','CA']};baseline={p:json.loads(path.read_text()) for p,path in old.items()};a=baseline['BR']['full'];b=baseline['CA']['full']
  neutral.append(dict(dataset=key[0],R=key[1],local_B=key[2],cache_percent=key[3],BR_seed=42,gain_abs_bytes=a['peer_bytes']-b['peer_bytes'],gain_rel=(a['peer_bytes']-b['peer_bytes'])/a['peer_bytes'],BR_source_sha256=sha(old['BR']),CA_source_sha256=sha(old['CA']),reference='existing unoptimized original512-prefix workload; one BR seed, not matched eight-seed statistic'))
 table('primary_stress.csv',winners);table('top3_candidates.csv',top3);table('neutral_reference.csv',neutral)
 proof=dict(status='PASS',cells=64,proxy_sample_DP_pairs=16*256*256,retained_candidates=512,exact_policy_replays=replays,BR_seeds=SEEDS,selection='max median absolute gain; then median relative gain; then lower CA peer bytes; deterministic seed tie order',substitution=False,eviction='Gate W128',decode=256,per_event_quota_imbalance_max=max(r['max_event_quota_imbalance'] for r in winners),source_receipts=all_proofs,policy_timing_runs=0)
 write(PACKET/'validation.json',proof)
 lines=['# CA communication-stress search','','These are optimized real-request stress examples, not dataset-average results.','Exact frozen-route resource accounting only: no physical E2E or quality claim.','Primary sample/DP choices maximize median absolute peer savings across eight','predeclared BR seeds. Best observed BR seed is reported separately.','','| Dataset | R | Local B | Cache | Sample / DP | Median peer reduction | Best observed BR seed |','|---|---:|---:|---:|---|---:|---:|']
 for row in winners:lines.append(f"| {row['dataset']} | {row['R']} | {row['local_B']} | {row['cache_percent']}% | {row['sample_seed']} / {row['dp_seed']} | {row['median_gain_rel']:.2%} | {row['best_observed_br_seed']} |")
 lines+=['','Full H2D, hit rates, byte gains, seed ranges, quota checks and state hashes are','in primary_stress.csv. top3_candidates.csv preserves selection alternatives.','selected_manifests/ pins exact request membership and balanced rank order.','neutral_reference.csv supplies the prior unoptimized random-prefix controls;','their seed42 statistic is not an eight-seed median. Larger candidate pools and','optimized sample/rank selection intentionally favor stress cases.','No further physical timing or replica-policy experiment follows automatically.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
