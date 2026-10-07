import subprocess,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1];root=Path('/home/hwlee/mgo-tools/headline-r4/llama.cpp');src=p/'examples/headline_llama_sync.cpp';binary=root/'build/bin/headline-llama-sync'
cmd=['g++','-std=c++17','-O2',str(src),'-I'+str(root/'include'),'-I'+str(root/'ggml/include'),'-I'+str(root/'vendor/nlohmann'),'-L'+str(root/'build/bin'),'-Wl,-rpath,'+str(root/'build/bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(binary)]
subprocess.run(cmd,check=True)
cache=root/'build/CMakeCache.txt';flags={}
if cache.exists():
 for line in cache.read_text(errors='replace').splitlines():
  if line.startswith(('GGML_CUDA_GRAPHS:','GGML_CUDA:','GGML_NATIVE:','GGML_OPENMP:')):
   k,v=line.split('=',1);flags[k]=v
receipt=dict(command=cmd,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),llama_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),cmake_flags=flags,thread_policy='runtime explicit; headline 32/32, audit 16/16 32/32 64/64')
(p/'experiments/main_table_global_workload_20261006/expanded_matrix/LLAMA_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(binary)
