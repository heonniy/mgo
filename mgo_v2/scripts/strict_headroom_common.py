"""Shared paths/helpers for strict LA+CA maximum-headroom study."""
from pathlib import Path
import hashlib,json,numpy as np

P=Path(__file__).resolve().parents[1]
PACKET=P/'experiments/strict_laca_headroom_20261006'
ROOT=Path('/home/hwlee/mgo-results/strict_laca_headroom_20261006')
OLD_ROOT=Path('/home/hwlee/mgo-results/prefill_decode_policy_headroom_20261006')
MODEL='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
R4_GPUS=[0,1,4,5]
CONTEXTS=(256,512)
WORLDS=(4,8)
BATCHES=(16,64)

def write(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(row,indent=2)+'\n')
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def pool_dir(context):
    context=int(context)
    if context==512:return OLD_ROOT/'pool'
    if context==256:return ROOT/'pool_L256'
    raise ValueError(context)

def request_rows():
    row=json.loads((OLD_ROOT/'requests.json').read_text())
    assert row['status']=='PASS' and len(row['requests'])==2048
    return row['requests']

def sample_ids(world,batch,sample_seed):
    n=int(world)*int(batch)
    return np.random.default_rng(int(sample_seed)).permutation(2048)[:n]

def ranked_ids(world,batch,sample_seed,dp_seed):
    ids=sample_ids(world,batch,sample_seed)
    ids=np.random.default_rng(int(dp_seed)).permutation(ids)
    return ids.reshape(int(world),int(batch))
