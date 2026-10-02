#!/usr/bin/env python3
"""Seal provenance and hash receipts after all diagnostic and analysis gates."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from summarize_trajectory_study import OUT,ROOT,PACKAGE,read


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    assert read(OUT/'validation.json')['status']=='PASS'
    assert read(ROOT/'status.json')['status']=='PHYSICAL_COMPLETE'
    assert read(ROOT/'replay_status.json')['status']=='COMPLETE'
    assert read(OUT/'figure_validation.json')['status']=='PASS'
    assert (OUT/'RESULTS.md').exists() and (OUT/'state_examples.md').exists()
    extension,=(PACKAGE.parent/'MoE-Infinity-EP-archer-coslot/moe_infinity').glob('_store*.so')
    extension_hash=sha(extension)
    assert extension_hash=='c5efdf74c571dbc7906eca4aea1293f3f3349c9383377633bf777c471a9cd85a'
    unchanged=['controller.py','admission.py','eviction.py','runtime.py','cache.py','executor.py','substitution.py','communicator.py']
    for name in unchanged:
        original=subprocess.check_output(['git','show',f'e61758e:mgo_v2/mgo_v2/{name}'],cwd=PACKAGE)
        assert original==(PACKAGE/'mgo_v2'/name).read_bytes()
    raw=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.name not in ('raw_hashes.json',):
            raw.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    (OUT/'raw_hash_receipts.json').write_text(json.dumps(raw,indent=2)+'\n')
    manifest=read(ROOT/'status.json')
    manifest.update(status='COMPLETE',finished_unix=time.time(),validation=read(OUT/'validation.json'),
        replay_status=read(ROOT/'replay_status.json'),native_extension_sha256=extension_hash,
        baseline_unchanged_runtime_files=unchanged,
        analysis_source_sha256={str(p.relative_to(PACKAGE)):sha(p) for p in sorted((PACKAGE/'scripts').glob('*trajectory*.py'))},
        raw_root=str(ROOT),raw_file_count=len(raw),raw_bytes=sum(r['bytes'] for r in raw))
    (OUT/'measurement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(ROOT/'cpu-tests-final.log',OUT/'cpu-tests-final.log')
    receipts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'}
    (OUT/'artifact_hashes.json').write_text(json.dumps(dict(status='PASS',files=receipts),indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',raw_files=len(raw),raw_gib=sum(r['bytes'] for r in raw)/2**30,portable_files=len(receipts))))

if __name__=='__main__':main()
