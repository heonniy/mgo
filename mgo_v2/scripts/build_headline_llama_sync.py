import subprocess,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1]
root=Path('/home/hwlee/mgo-tools/headline-r4/llama.cpp')
build_dir=root/'build'
src=p/'examples/headline_llama_sync.cpp'
binary=build_dir/'bin/headline-llama-sync'

# Compile CUDA Graph support once. The revised headline baseline disables CUDA
# Graph capture/replay at runtime with GGML_CUDA_DISABLE_GRAPHS=1. Keeping the
# support compiled in lets the single-cell A/B use the exact same binary.
configure_cmd=['cmake','-S',str(root),'-B',str(build_dir),'-DGGML_CUDA_GRAPHS=ON']
backend_build_cmd=['cmake','--build',str(build_dir),'--config','Release','--parallel','16']
subprocess.run(configure_cmd,check=True)
subprocess.run(backend_build_cmd,check=True)

cache=build_dir/'CMakeCache.txt';flags={}
if cache.exists():
 for line in cache.read_text(errors='replace').splitlines():
  if line.startswith(('GGML_CUDA_GRAPHS:','GGML_CUDA:','GGML_NATIVE:','GGML_OPENMP:')):
   k,v=line.split('=',1);flags[k]=v
assert flags.get('GGML_CUDA_GRAPHS:BOOL')=='ON',flags
assert flags.get('GGML_CUDA:BOOL')=='ON',flags

cmd=['g++','-std=c++17','-O2',str(src),
     '-I'+str(root/'include'),'-I'+str(root/'ggml/include'),'-I'+str(root/'vendor/nlohmann'),
     '-L'+str(build_dir/'bin'),'-Wl,-rpath,'+str(build_dir/'bin'),
     '-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(binary)]
subprocess.run(cmd,check=True)

receipt=dict(
 configure_command=configure_cmd,backend_build_command=backend_build_cmd,command=cmd,
 source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
 llama_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 cmake_flags=flags,cuda_graph_support=True,
 headline_cuda_graph_runtime='OFF via GGML_CUDA_DISABLE_GRAPHS=1',
 llama_graph_reuse=True,baseline_policy_eligible=True,
 thread_policy='headline fixed 32/32 with deterministic affinity'
)
(p/'experiments/main_table_global_workload_20261006/expanded_matrix/LLAMA_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(binary)
