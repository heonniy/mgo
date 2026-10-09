"""Build DeepSeek synchronous runner against the pinned llama.cpp backend."""
import hashlib
import json
import subprocess
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
ROOT = Path('/home/hwlee/mgo-tools/headline-r4/llama.cpp')
BUILD = ROOT / 'build'
SOURCE = PKG / 'examples/headline_llama_deepseek_sync.cpp'
BINARY = BUILD / 'bin/headline-llama-deepseek-sync'
RECEIPT = PKG / 'experiments/deepseek_cache_ablation_20261009/LLAMA_BUILD.json'


def main():
    flags = {}
    for line in (BUILD / 'CMakeCache.txt').read_text(errors='replace').splitlines():
        if line.startswith(('GGML_CUDA_GRAPHS:', 'GGML_CUDA:', 'GGML_NATIVE:', 'GGML_OPENMP:')):
            key, value = line.split('=', 1)
            flags[key] = value
    assert flags.get('GGML_CUDA_GRAPHS:BOOL') == 'ON' and flags.get('GGML_CUDA:BOOL') == 'ON', flags
    cmd = ['g++', '-std=c++17', '-O2', str(SOURCE),
           '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'ggml/include'),
           '-I' + str(ROOT / 'vendor/nlohmann'), '-L' + str(BUILD / 'bin'),
           '-Wl,-rpath,' + str(BUILD / 'bin'), '-lllama', '-lggml', '-lggml-base', '-lggml-cpu',
           '-o', str(BINARY)]
    subprocess.run(cmd, check=True)
    receipt = dict(status='PASS', command=cmd, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
                   llama_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   cmake_flags=flags, cuda_graph_support=True, baseline_policy_eligible=True,
                   expert_placement=['balanced4', 'balanced8', 'balanced12'],
                   cpu_threads=32, cuda_graphs_runtime='OFF', graph_reuse='OFF')
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'PASS {BINARY}')


if __name__ == '__main__':
    main()
