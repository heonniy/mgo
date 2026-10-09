"""Archive a validated completed optimization cohort; one local commit per arm."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from pcie_host import write

REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'mgo_v2/experiments/pcie_topology_ablation_20261009/tpot_optimization'

def archive(source,destination):
    destination.mkdir(parents=True,exist_ok=False);inventory={}
    for path in sorted(source.iterdir()):
        if not path.is_file():continue
        data=path.read_bytes();entry=dict(source=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if path.suffix in ('.json','.npy','.csv','.pdf','.png'):
            compressed=len(data)>128*1024 and path.suffix not in ('.pdf','.png')
            target=destination/(path.name+'.gz' if compressed else path.name)
            payload=gzip.compress(data,mtime=0) if compressed else data;target.write_bytes(payload)
            if compressed:assert gzip.decompress(payload)==data
            entry.update(artifact=target.name,lossless_gzip=compressed,artifact_sha256=hashlib.sha256(payload).hexdigest())
        inventory[path.name]=entry
    write(destination/'RAW_ARTIFACTS.json',inventory)
    (destination/'ARCHIVE.md').write_text('Files above 128 KiB are stored as deterministic lossless gzip. RAW_ARTIFACTS.json records both original and stored SHA256. Decompress .gz files to their original names to use the validators. Other raw files remain at the recorded data2 paths.\n')

def commit(destination,message):
    subprocess.run(['git','add',str(destination.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m',message],cwd=REPO,check=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()

def run(root,report,do_commit):
    assert json.loads((root/'status.json').read_text())['status']=='PASS'
    cohort=json.loads((root/'result.json').read_text());assert cohort['status']=='PASS' and cohort['sequence']=='optimize'
    validated=json.loads((report/'RESULTS.json').read_text());assert validated['status']=='PASS' and validated['root']==str(root)
    assert set(validated['arms'])==set(cohort['arms'])
    if do_commit:
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip(),'Unrelated staged changes exist'
    commits={}
    for arm in cohort['arms']:
        source=root/arm
        assert json.loads((source/'phase_validation.json').read_text())['status']=='PASS'
        archive(source,DEST/arm)
        if do_commit:commits[arm]=commit(DEST/arm,f'Complete {arm} full64 TPOT optimization primaries and isolated timers-only breakdown')
    cohort_dir=DEST/'cohort';archive(root,cohort_dir)
    write(cohort_dir/'ARM_COMMITS.json',commits)
    archive(report,DEST/'comparison')
    if do_commit:
        subprocess.run(['git','add',str(cohort_dir.relative_to(REPO)),str((DEST/'comparison').relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m',
                        'Report common R/G TPOT optimization gains, metadata parity, live traffic and endpoint-complete breakdown'],cwd=REPO,check=True)
    print(json.dumps(dict(status='PASS',destination=str(DEST),arm_commits=commits)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--commit',action='store_true');a=p.parse_args();run(a.root,a.report,a.commit)
