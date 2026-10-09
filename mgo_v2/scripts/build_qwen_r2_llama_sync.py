"""Build the separate R2 balanced llama.cpp comparison executable."""

import hashlib
import json
import subprocess
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
ROOT = Path('/home/hwlee/mgo-tools/headline-r4/llama.cpp')
SOURCE = PKG / 'examples/headline_llama_sync_r2.cpp'
BINARY = ROOT / 'build/bin/headline-llama-sync-r2'
RECEIPT = PKG / 'experiments/qwen_r2_r8_main_table_20261010/LLAMA_R2_BUILD.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    flags = {}
    for line in (ROOT / 'build/CMakeCache.txt').read_text().splitlines():
        if line.startswith(('GGML_CUDA_GRAPHS:', 'GGML_CUDA:')):
            key, value = line.split('=', 1)
            flags[key] = value
    assert flags['GGML_CUDA_GRAPHS:BOOL'] == 'ON' and flags['GGML_CUDA:BOOL'] == 'ON'
    command = ['g++', '-std=c++17', '-O2', str(SOURCE),
               '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'ggml/include'),
               '-I' + str(ROOT / 'vendor/nlohmann'),
               '-L' + str(ROOT / 'build/bin'),
               '-Wl,-rpath,' + str(ROOT / 'build/bin'),
               '-lllama', '-lggml', '-lggml-base', '-lggml-cpu', '-o', str(BINARY)]
    subprocess.run(command, check=True)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(dict(source_sha256=digest(SOURCE),
                                       binary_sha256=digest(BINARY),
                                       cmake_flags=flags, command=command,
                                       cuda_graph_support=True,
                                       baseline_policy_eligible=True,
                                       expert_placement='balanced7',
                                       physical_gpus=[0, 1],
                                       llama_commit=subprocess.check_output(
                                           ['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                           text=True).strip()), indent=2) + '\n')
    print(BINARY)


if __name__ == '__main__':
    main()
