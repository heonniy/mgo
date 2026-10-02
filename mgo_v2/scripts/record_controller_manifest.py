#!/usr/bin/env python3
"""Freeze separate controller-stage sources and verify the original C0 core."""
import argparse
import hashlib
import json
import subprocess
import time
from run_controller_overhead import PACKAGE,ROOT,OUT,ORIGINAL,INPUTS,make_cell


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--original-commit',required=True);args=parser.parse_args()
    if (OUT/'measurement_manifest.json').exists():raise RuntimeError('preserve the existing freeze; record an explicit amendment instead')
    original=json.loads((ORIGINAL/'measurement_manifest.json').read_text())
    for name,expected in original['inputs'].items():assert sha(INPUTS/name)==expected,(name,'frozen input changed')
    core={name:expected for name,expected in original['source_sha256'].items()
          if name.startswith('mgo_v2/') or name in ('examples/benchmark_model.py','examples/rank_oracle_worker.py','scripts/run_rank_oracle_study.py')}
    for name,expected in core.items():assert sha(PACKAGE/name)==expected,(name,'frozen C0 source changed')
    paths=list((PACKAGE/'mgo_v2').glob('*.py'))+[PACKAGE/'examples/benchmark_model.py',PACKAGE/'examples/controller_overhead_worker.py',PACKAGE/'examples/rank_oracle_worker.py']
    paths += [PACKAGE/'scripts'/name for name in ('run_controller_overhead.py','replay_controller_equivalence.py',
              'analyze_controller_profiles.py','summarize_controller_overhead.py','record_controller_manifest.py','analyze_rank_oracle_profiles.py')]
    paths.append(PACKAGE/'tests/test_controller_optimized.py')
    manifest=dict(created_unix=time.time(),original_packet_commit=args.original_commit,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip(),
        frozen_C0_core_sha256=core,source_sha256={str(p.relative_to(PACKAGE)):sha(p) for p in sorted(paths)},
        original_measurement_manifest_sha256=sha(ORIGINAL/'measurement_manifest.json'),
        original_result_validation_sha256=sha(ORIGINAL/'validation.json'),
        scope_amendment_sha256=sha(ORIGINAL/'scope_amendment.json'),protocol_sha256=sha(OUT/'PROTOCOL.md'),
        inputs={name:sha(INPUTS/name) for name in original['inputs']},checkpoint=original['checkpoint'],
        extension_sha256=original['extension_sha256'],physical_gpus=[0,1,4,5],local_batch=8,
        primary_generations=9,repeats=1,forwards=65,expanded_batches=False,
        order=[make_cell(p,v) for v in ('C0','C1','C2') for p in ('P0','P1','O0')],
        posthoc_profiles=[make_cell('P1',v,True) for v in ('C1','C2')],profile_forwards=9,
        timing_scope='Fresh physical/logical cache per generation; compact evidence capture shared by C0/C1/C2; hashing/serialization/parity outside timer; no diagnostic subcomponent timers or profiler in primary timing.',
        primary_limit='Single sample per policy/controller with C0 then C1 then C2 order. Descriptive measurements only; no stable or small-speedup claim.',
        diagnostics='Full captured-route CPU C0/C1/follower differential gate, separate exclusive component timers; short P1 GPU profile after all primary generations.',
        raw_root=str(ROOT),python=original['python'],nsys_version=original['nsys_version'])
    (OUT/'measurement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Controller manifest recorded; original C0 core unchanged.')

if __name__=='__main__':main()
