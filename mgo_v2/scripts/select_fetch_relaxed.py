"""Rerank existing exact64 receipts without repeating the completed search."""
from fetch_relaxed_common import *
def main():
 rows=[];sources=[];checked=0
 for path in sorted((OLD_ROOT/'B').glob('*.json')):
  d=json.loads(path.read_text());assert d['status']=='PASS';ca=d['CA'];job=d['job'];sources.append(dict(path=str(path),sha256=sha(path)))
  for br in d['BR']:
   checked+=1;a,b=br['full'],ca['full']
   if not eligible(a,b):continue
   assert a['max_event_rank_fetch_imbalance']<=1 and b['max_event_rank_fetch_imbalance']<=1
   peer=a['peer_bytes']-b['peer_bytes'];critical=br['Critical_total']-ca['Critical_total']
   rows.append(dict(world=job['world'],sample_seed=job['sample_seed'],dp_seed=job['dp_seed'],BR_seed=br['seed'],peer_gain_bytes=peer,peer_gain_relative=peer/a['peer_bytes'],critical_gain_bytes=critical,critical_gain_relative=critical/br['Critical_total'],reload_equal=a['reload_fetches']==b['reload_fetches'],fetch_difference_relative=abs(a['total_fetches']-b['total_fetches'])/a['total_fetches'],H2D_difference_relative=abs(a['H2D_bytes']-b['H2D_bytes'])/a['H2D_bytes'],candidate_path=str(path),candidate_sha256=sha(path),request_ids=d['request_ids'],route_sha256=d['route_sha256'],gate_sha256=d['gate_sha256'],CA=ca,BR=br))
 winners=[];top=[]
 for world in [4,8]:
  group=[r for r in rows if r['world']==world];assert group,f'No eligible R{world}'
  for objective in ['Peer-best','Critical-best']:
   def key(r):
    score=(r['peer_gain_bytes'],r['peer_gain_relative'],r['critical_gain_bytes'],r['reload_equal']) if objective=='Peer-best' else (r['critical_gain_bytes'],r['critical_gain_relative'],r['peer_gain_bytes'],r['reload_equal'])
    return (*score,-r['sample_seed'],-r['dp_seed'],-r['BR_seed'])
   ranked=sorted(group,key=key,reverse=True);winners.append(dict(objective=objective,**ranked[0]));top.append(dict(world=world,objective=objective,rows=ranked[:10]))
 write(PACKET/'winners.json',winners);write(PACKET/'top10.json',top)
 write(PACKET/'selection_validation.json',dict(status='PASS',pairs_reused=checked,new_replays=0,new_trace_capture=False,horizon=64,eligible_pairs=len(rows),eligible_per_R={w:sum(r['world']==w for r in rows) for w in [4,8]},relative_difference_denominator='BR',tolerance=.001,sources=sources))
 for w in winners:print(w['world'],w['objective'],w['sample_seed'],w['dp_seed'],w['BR_seed'],'fetch difference',w['fetch_difference_relative'],'peer gain',w['peer_gain_relative'],flush=True)
if __name__=='__main__':main()
