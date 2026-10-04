"""Build common-stack BF16 arms from immutable BR-only tuning evidence."""
import argparse,hashlib,json
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'
ARMS=('V1_OPT_NOPF_BARRIER','V2_OPT_PF_BARRIER','V3_OPT_PF_OVERLAP')
def groups_for(selection,horizon,batches):
 assert selection['status']=='PASS' and selection['frozen']
 assert selection['policy_used']=='BR only' and selection['immutable_for_LA']
 assert selection['horizon']==256 and selection['common_return_precision']=='bf16'
 chosen=selection['chosen'];assert chosen['P'] in (1,2,4) and chosen['trigger'] in ('T0','T1','T2')
 groups=[]
 for arm in ARMS:
  for batch in batches:
   cases=[dict(label=policy,policy=policy,P=0 if arm==ARMS[0] else chosen['P'],trigger=chosen['trigger'],horizon=horizon,overlap=arm==ARMS[2],partial_precision='bf16',runtime_arm=arm) for policy in ('BR','LA')]
   groups.append(dict(batch=batch,horizon=horizon,paired=True,runtime_arm=arm,cases=cases))
 return groups

def main(a):
 raw=a.selection.read_bytes();selection=json.loads(raw)
 for arm in ARMS:
  groups=[g for g in groups_for(selection,8 if a.smoke else 256,[128] if a.smoke else [128,256]) if g['runtime_arm']==arm]
  target=PACKET/f'{"M14_SMOKE" if a.smoke else "M15"}_{arm}_CONFIG.json'
  target.write_text(json.dumps(groups,indent=2)+'\n')
 receipt=dict(status='PREPARED',physically_validated=False,selection_path=str(a.selection),selection_sha256=hashlib.sha256(raw).hexdigest(),precision='bf16',horizon=8 if a.smoke else 256,arms=ARMS,policy_order='alternate BR/LA and LA/BR per repeat',arena_count=1)
 (PACKET/f'{"M14_SMOKE" if a.smoke else "M15"}_ARMS_PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,default=PACKET/'M13_FROZEN_PREFETCH.json');p.add_argument('--smoke',action='store_true');main(p.parse_args())
