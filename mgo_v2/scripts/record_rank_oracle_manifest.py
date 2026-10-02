#!/usr/bin/env python3
"""Freeze source, checkpoint, inputs, schedule and runtime versions before timing."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from run_rank_oracle_study import ROOT,OUT,PACKAGE,INPUTS,NSYS,write


def main():
    paths=list((PACKAGE/'mgo_v2').glob('*.py')) + [PACKAGE/'examples/benchmark_model.py',PACKAGE/'examples/rank_oracle_worker.py']
    paths+=list((PACKAGE/'scripts').glob('*rank_oracle*.py'))
    source={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    ext,=(PACKAGE.parent/'MoE-Infinity-EP-archer-coslot/moe_infinity').glob('_store*.so')
    manifest=dict(plan_commit='dccc93cb9c1ed3ecbd6a72f276c52600e2d43035',
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip(),
        source_sha256=source,extension_sha256=hashlib.sha256(ext.read_bytes()).hexdigest(),
        checkpoint=json.loads((INPUTS/'expert_store/manifest.json').read_text())['identity'],
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [INPUTS/'similarity.npy',INPUTS/'affinity.npz',INPUTS/'screen_workload.json']},
        physical_gpus=[0,1,4,5],world=4,local_batches=[4,8,16],decode_forwards=64,prefill_forwards=1,
        primary_generations=54,profiles=3,order_permutations=list(itertools.permutations(('P0','P1','O0'))),
        model_load_protocol='One load per local batch; empty expert cache/controller/history each policy generation',
        validation_evidence='CPU evidence retained equally for each policy; hashes, JSON serialization and token checks outside generation timer',
        oracle_timing='Exact solves in separate planning runs; replay demand checks online; trace loaded before timer',
        profile_tool=str(NSYS),nsys_version=subprocess.check_output([str(NSYS),'--version'],text=True).strip(),
        python=sys.executable,created_unix=time.time(),raw_root=str(ROOT))
    write(OUT/'measurement_manifest.json',manifest)

if __name__=='__main__': main()
