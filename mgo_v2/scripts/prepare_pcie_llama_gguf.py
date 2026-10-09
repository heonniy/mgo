"""Convert only the frozen local model and verify actual BF16 expert storage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from pcie_host import ROOT,write

DATA=Path('/data2/esjung');SOURCE=DATA/'tools/llama.cpp'
MODEL=DATA/'models/Qwen3-30B-A3B-Instruct-2507'
GGUF=DATA/'models/Qwen3-30B-A3B-Instruct-2507-BF16.gguf'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    converter=SOURCE/'convert_hf_to_gguf.py'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()=='3109914090564b4c5280f30896369d55b86bbdbf'
    a.out.mkdir(parents=True,exist_ok=False)
    pending=a.out/'Qwen3-BF16.pending.gguf'
    command=[sys.executable,str(converter),str(MODEL),'--outtype','bf16','--use-temp-file','--outfile',str(pending)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
             PYTHONPATH=str(SOURCE/'gguf-py'),TMPDIR=str(a.out),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if GGUF.exists():
        receipt=json.loads((ROOT/'GGUF_CONVERSION.json').read_text())
        assert receipt['status']=='PASS' and sha(GGUF)==receipt['gguf_sha256']
    else:subprocess.run(command,env=env,check=True)
    sys.path.insert(0,str(SOURCE/'gguf-py'))
    import gguf
    actual=GGUF if GGUF.exists() else pending
    reader=gguf.GGUFReader(str(actual),'r')
    experts=[t for t in reader.tensors if '_exps.weight' in t.name]
    assert len(experts)==48*3
    assert all(t.tensor_type==gguf.GGMLQuantizationType.BF16 for t in experts)
    assert sum(t.n_bytes for t in experts)==54*2**30
    rows=[dict(name=t.name,type=t.tensor_type.name,shape=t.shape.tolist(),bytes=t.n_bytes) for t in experts]
    del reader,experts
    digest=sha(actual)
    if actual==pending:pending.rename(GGUF)
    result=dict(status='PASS',command=command,converter_sha256=sha(converter),gguf=str(GGUF),gguf_sha256=digest,
        gguf_bytes=GGUF.stat().st_size,expert_tensors=144,expert_bytes=54*2**30,expert_storage='BF16',experts=rows,
        model_config_sha256=sha(MODEL/'config.json'),model_index_sha256=sha(MODEL/'model.safetensors.index.json'),
        conda_prefix=sys.prefix,scope='Local frozen model conversion; normalization/auxiliary tensors may use native FP32, expert weights are actual BF16, no quantized expert substitution')
    write(ROOT/'GGUF_CONVERSION.json',result);write(a.out/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
