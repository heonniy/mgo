"""CPU-only private CUDA12.1/BF16 Infinity build with native pinned telemetry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from pcie_host import ROOT,write
from build_pcie_llama import prepare_toolkit,TOOLKIT

DATA=Path('/data2/esjung');SOURCE=DATA/'tools/MoE-Infinity'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    assert commit=='9f819a6d43e043bded6e0692e5e58793e1623364'
    cutlass=DATA/'tools/cutlass-v3.5.1'
    assert subprocess.check_output(['git','describe','--tags','--exact-match'],cwd=cutlass,text=True).strip()=='v3.5.1'
    prepare_toolkit()
    paths=[SOURCE,DATA/'tools/moe-store']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME=str(TOOLKIT),
        PATH=str(DATA/'envs/cuda121/bin')+':'+os.environ['PATH'],
        CUTLASS_DIR=str(cutlass),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='1',
        MOE_ENABLE_SM90='0',MOE_ENABLE_SM120='0',MOE_ENABLE_V4_FP4='0',NVTX_DISABLE='1',
        CPATH=str(TOOLKIT/'include'),LIBRARY_PATH=str(TOOLKIT/'lib'),
        PYTHONPATH=':'.join(map(str,paths)),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    nvcc=subprocess.check_output([str(TOOLKIT/'bin/nvcc'),'--version'],env=env,text=True)
    assert 'release 12.1' in nvcc
    argv=[sys.executable,'setup.py','build_ext','--inplace']
    subprocess.run(argv,cwd=SOURCE,env=env,check=True)
    # Import checks are CPU-only; actual native allocation/progress is smoke.
    check="""import torch
import moe_infinity._store as store
assert not torch.cuda.is_initialized()
names=[name for name in dir(store) if isinstance(getattr(store,name),type) and hasattr(getattr(store,name),'get_host_pinned_memory_stats')]
assert names,'Native pinned telemetry binding missing'
print(names)
"""
    check_result=subprocess.check_output([sys.executable,'-c',check],cwd=SOURCE,env=env,text=True)
    binaries=list((SOURCE/'moe_infinity').glob('*.so'));assert binaries
    sources={str(p.relative_to(SOURCE)):sha(p) for prefix in ('core','extensions','moe_infinity')
             for p in (SOURCE/prefix).rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cc','.cpp','.cu','.py')}
    write(ROOT/'INFINITY_BUILD.json',dict(status='PASS',source_commit=commit,setup_sha256=sha(SOURCE/'setup.py'),
        command=argv,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_HOME','CUTLASS_DIR','TORCH_CUDA_ARCH_LIST','MAX_JOBS','MOE_ENABLE_SM90','MOE_ENABLE_SM120','MOE_ENABLE_V4_FP4','NVTX_DISABLE')},
        explicit_setup_cuda_architecture='sm80 BF16, forward-compatible with Ada sm89; Blackwell FP4 disabled',
        conda_prefix=sys.prefix,nvcc=nvcc,cpu_import_check=check_result,
        native_binaries={str(p):sha(p) for p in binaries},source_sha256=sources,
        scope='Build/import/telemetry availability only; physical smoke/full64 measurement still required'))


if __name__=='__main__':main()
