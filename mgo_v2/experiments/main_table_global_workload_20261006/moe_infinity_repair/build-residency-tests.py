import subprocess,sysconfig,os
from pathlib import Path
import torch
from torch.utils.cpp_extension import include_paths,library_paths
root=Path('/home/hwlee/mgo-tools/headline-r4/MoE-Infinity');tools=root.parent;gt=tools/'googletest/googletest'
incs=[root/'core',root/'core/include',tools/'cutlass/include',tools/'cutlass/tools/util/include',gt/'include',gt,Path('/usr/local/cuda/include'),tools/'NVTX/c/include',Path('/home/hwlee/anaconda3/envs/qwen3-gemm/include'),Path(sysconfig.get_paths()['include']),*map(Path,include_paths())]
libs=library_paths()+['/usr/local/cuda/lib64',sysconfig.get_config_var('LIBDIR')]
cmd=['g++','-std=c++20','-O0','-pthread','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]+['-I'+str(x) for x in incs]
cmd += [str(root/'tests/cpp/unit/prefetch/test_expert_residency.cpp'),str(root/'tests/cpp/unit/prefetch/test_expert_residency_variants.cpp'),str(root/'core/prefetch/expert_residency.cpp'),str(gt/'src/gtest-all.cc'),str(gt/'src/gtest_main.cc'),str(root/'moe_infinity/_store.cpython-312-x86_64-linux-gnu.so')]
cmd += ['-L'+x for x in libs]+['-Wl,-rpath,'+x for x in libs]+['-ltorch','-ltorch_cpu','-ltorch_python','-lc10','-lcudart','-lpython3.12','-o',str(tools/'test_eam_residency')]
subprocess.run(cmd,check=True,cwd=root)
subprocess.run([str(tools/'test_eam_residency')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
