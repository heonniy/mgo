"""Fixed-512 TTFT packet paths and atomic receipts."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1]
PACKET=P/'experiments/prefill_decode_policy_headroom_20261006'
ROOT=Path('/home/hwlee/mgo-results/prefill_decode_policy_headroom_20261006')
MODEL='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
GPUS=[0,1,4,5]
POLICIES=['BR','CA','OLD_CA','LA']
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
