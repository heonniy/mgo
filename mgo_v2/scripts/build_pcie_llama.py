"""Build the frozen native BF16 llama baseline without initializing CUDA."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from pcie_host import ROOT, write

DATA=Path('/data2/esjung')
PKG=Path(__file__).resolve().parents[1]
SOURCE=DATA/'tools/llama.cpp'
CUDA=DATA/'envs/cuda121'
TOOLKIT=DATA/'tools/cuda121-toolkit'


def link(source,target):
    if not target.exists() and not target.is_symlink():target.symlink_to(source)


def prepare_toolkit():
    # Add wheel-supplied cuBLAS/cuSPARSE to a private toolkit view. Neither
    # Conda's prefix nor the Torch installation is modified by these links.
    TOOLKIT.mkdir(exist_ok=True)
    for name in ('bin','nvvm'):link(CUDA/name,TOOLKIT/name)
    for name in ('include','lib'):(TOOLKIT/name).mkdir(exist_ok=True)
    link(TOOLKIT/'lib',TOOLKIT/'lib64')
    vendors=DATA/'envs/mgo-pcie/lib/python3.11/site-packages/nvidia'
    for root in [CUDA]+sorted(vendors.iterdir()):
        for name in ('include','lib'):
            directory=root/name
            if not directory.exists():continue
            for item in directory.iterdir():
                target=TOOLKIT/name/item.name;link(item,target)
                if '.so.' in item.name:link(item,TOOLKIT/name/(item.name.split('.so.')[0]+'.so'))
    driver=Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
    assert driver.exists();link(driver,TOOLKIT/'lib/libcuda.so')
    assert all((TOOLKIT/'lib'/name).exists() for name in ('libcudadevrt.a','libcudart_static.a','libcublas.so','libcublasLt.so','libcusparse.so'))
    write(ROOT/'CUDA_TOOLKIT_VIEW.json',dict(root=str(TOOLKIT),conda_cuda=str(CUDA),torch_vendors=str(vendors),
          gpu_initialization=False,links={str(p.relative_to(TOOLKIT)):str(p.readlink()) for p in TOOLKIT.rglob('*') if p.is_symlink()}))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Build must have no visible GPUs'
    expected='3109914090564b4c5280f30896369d55b86bbdbf'
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();assert actual==expected
    prepare_toolkit()
    build=SOURCE/'build';src=PKG/'examples/headline_llama_sync.cpp';binary=build/'bin/headline-llama-sync'
    cmake=shutil.which('cmake');assert cmake
    configure=[cmake,'-S',str(SOURCE),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
               '-DGGML_CUDA=ON','-DGGML_CUDA_GRAPHS=ON','-DCMAKE_CUDA_ARCHITECTURES=89',
               '-DCMAKE_CUDA_COMPILER='+str(CUDA/'bin/nvcc'),'-DCUDAToolkit_ROOT='+str(TOOLKIT),
               '-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=OFF']
    compile_backend=[cmake,'--build',str(build),'--target','llama','--config','Release','--parallel','4']
    environment=dict(os.environ,CUDA_HOME=str(TOOLKIT),CUDACXX=str(CUDA/'bin/nvcc'),
                     CPATH=str(TOOLKIT/'include'),LIBRARY_PATH=str(TOOLKIT/'lib'))
    subprocess.run(configure,check=True,env=environment);subprocess.run(compile_backend,check=True,env=environment)
    compile_wrapper=['g++','-std=c++17','-O2',str(src),'-I'+str(SOURCE/'include'),
                     '-I'+str(SOURCE/'ggml/include'),'-I'+str(SOURCE/'vendor/nlohmann'),
                     '-L'+str(build/'bin'),'-Wl,-rpath,'+str(build/'bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(binary)]
    subprocess.run(compile_wrapper,check=True,env=environment)
    flags={}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if line.startswith(('GGML_CUDA_GRAPHS:','GGML_CUDA:','GGML_NATIVE:','GGML_OPENMP:','CMAKE_CUDA_ARCHITECTURES:')):
            key,value=line.split('=',1);flags[key]=value
    assert flags['GGML_CUDA_GRAPHS:BOOL']=='ON' and flags['GGML_CUDA:BOOL']=='ON'
    write(ROOT/'LLAMA_BUILD.json',dict(configure_command=configure,backend_build_command=compile_backend,command=compile_wrapper,
          source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
          llama_commit=actual,cmake_flags=flags,cuda_graph_support=True,baseline_policy_eligible=True,
          headline_cuda_graph_runtime='OFF via GGML_CUDA_DISABLE_GRAPHS=1',llama_graph_reuse=False,
          headline_llama_graph_reuse='OFF via LLAMA_GRAPH_REUSE_DISABLE=1',headline_expert_placement='balanced3',
          thread_policy='32 physical CPU cores across both NUMA nodes',build_visible_gpus='',cuda_toolkit=str(TOOLKIT)))
    print(binary,flush=True)


if __name__=='__main__':main()
