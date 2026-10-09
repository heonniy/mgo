"""Verify pinned HF revisions and prepare the CPU expert store before GPU jobs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
PKG=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PKG),str(PKG/'scripts')]
from pcie_host import ROOT,write
from huggingface_hub import HfApi

MODEL=Path('/data2/esjung/models/Qwen3-30B-A3B-Instruct-2507')
REV='0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    api=HfApi();meta=api.model_info('Qwen/Qwen3-30B-A3B-Instruct-2507',revision=REV,files_metadata=True)
    assert meta.sha==REV
    expected={s.rfilename:s for s in meta.siblings if s.rfilename.endswith('.safetensors')}
    required=list(expected)+['model.safetensors.index.json','config.json','tokenizer.json','tokenizer_config.json','generation_config.json']
    deadline=time.monotonic()+3600
    while not all((MODEL/p).exists() and (p not in expected or (MODEL/p).stat().st_size==expected[p].size) for p in required):
        if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP file')
        if time.monotonic()>deadline:raise TimeoutError('Checkpoint acquisition exceeded one hour')
        time.sleep(5)
    shards={}
    for name,sibling in expected.items():
        actual=sha(MODEL/name)
        assert actual==sibling.lfs.sha256,(name,actual,sibling.lfs.sha256)
        shards[name]=dict(bytes=(MODEL/name).stat().st_size,sha256=actual)
        print('Checkpoint shard SHA256 PASS',name,flush=True)
    files={p:sha(MODEL/p) for p in required if p not in expected}
    workload=json.loads(Path('/data2/esjung/datasets/frozen_pcie_topology_20261009/WORKLOADS.json').read_text())
    for spec in workload['cells']:
        for phase in ('target','warmup'):
            assert sha(Path(spec[phase]['path']))==spec[phase]['sha256']
    dataset=Path('/data2/esjung/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json')
    dataset_sha=sha(dataset)
    assert dataset_sha=='35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
    from mgo_v2.model_loader import prepare_checkpoint_store
    manifest=prepare_checkpoint_store(str(MODEL),'/data2/esjung/models/Qwen3-expert-store')
    assert manifest['bytes']==54*2**30 and manifest['tensors']==48*128*3
    receipt=dict(status='PASS',repository='Qwen/Qwen3-30B-A3B-Instruct-2507',revision=REV,path=str(MODEL),
                 shards=shards,files=files,expert_store=manifest,dataset_path=str(dataset),dataset_sha256=dataset_sha,
                 dataset_revision='192ab2185289094fc556ec8ce5ce1e8e587154ca',workloads=workload,
                 git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PKG.parent,text=True).strip())
    write(ROOT/'checkpoint_preflight.json',receipt)
    print('PASS checkpoint revision, all shard hashes, frozen manifests, dataset and 54-GiB expert store',flush=True)


if __name__=='__main__':main()
