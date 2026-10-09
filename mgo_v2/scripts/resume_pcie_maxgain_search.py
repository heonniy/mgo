"""Resume after input-identity repair, retaining original measurements verbatim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from pcie_host import ROOT,write
from pcie_maxgain_select import sha

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
DEST=PKG/'experiments/pcie_topology_ablation_20261009/maxgain64'


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);done=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
             PYTHONPATH=':'.join([str(PKG),str(PKG/'scripts'),str(PKG/'examples')]))
    def step(label,argv):
        if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP observed')
        write(a.out/'status.json',dict(status='RUNNING',step=label,pid=os.getpid(),completed=done))
        with (a.out/(label+'.log')).open('w') as log:
            child=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(a.out/'step.json',dict(step=label,pid=child.pid,argv=argv,started=time.time()))
            code=child.wait()
        if code:raise RuntimeError(f'{label} exited {code}; source attempts preserved')
        done.append(label);write(a.out/'completed.json',done);print(json.dumps(dict(completed=label)),flush=True)
    def command(name,*args):return [sys.executable,str(PKG/'scripts'/name),*map(str,args)]
    def commit(path,message):
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add','--',str(path.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m',message],cwd=REPO,check=True)
    try:
        write(a.out/'status.json',dict(status='RUNNING',step='wait_repaired_nominees',pid=os.getpid(),completed=[]))
        started=time.monotonic()
        while True:
            if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP observed')
            status=json.loads((a.repair/'status.json').read_text())
            if status['status']=='FAIL':raise RuntimeError('Repaired nomination failed; inspect source failure before resuming')
            if status['status']=='PASS':
                try:os.kill(status['pid'],0)
                except ProcessLookupError:break
            if time.monotonic()-started>10860:raise TimeoutError('Repaired nomination wait exceeded its supervisor bound')
            time.sleep(1)
        step('archive_repaired_nominees',command('pcie_maxgain_report.py','--root',a.repair,'--archive','--commit','--archive-label','nomination_repair'))
        spec=json.loads(a.spec.read_text());assert spec['status']=='FROZEN'
        original=json.loads((a.original/'SEARCH_SPEC.json').read_text());old={c['candidate']:c for c in original['candidates']}
        changed=[c['candidate'] for c in spec['candidates'] if c['sha256']!=old[c['candidate']]['sha256']]
        assert changed==['family_1','family_4'] and len(spec['candidates'])==33
        assert json.loads((a.original/'status.json').read_text())['status']=='PASS'
        assert json.loads((a.repair/'result.json').read_text())['candidates']==changed
        launches=[json.loads((root/'launch.json').read_text()) for root in (a.original,a.repair)]
        runtime_paths=['mgo_v2/mgo_v2/'+name for name in ('decode_runtime.py','pcie_grouped_decode.py','pcie_grouped_probe.py','grouped_expert.py','pcie_native_metadata.py')]
        runtime_paths+=['mgo_v2/scripts/'+name for name in ('native_pcie_controller.cpp','native_pcie_metadata.cpp')]
        for path in runtime_paths:assert launches[0]['source_sha256'][path]==launches[1]['source_sha256'][path],path
        for name in ('MGO_MODEL_PATH','MGO_EXPERT_STORE'):assert launches[0]['environment'][name]==launches[1]['environment'][name]
        for arm in ('R-NEAR','G-NEAR'):
            for rank in range(4):
                samples=[json.loads((root/'_warmup'/arm/f'repeat0_rank{rank}.json').read_text()) for root in (a.original,a.repair)]
                assert samples[0]['tokens']==samples[1]['tokens'] and samples[0]['validation']['state_hash']==samples[1]['validation']['state_hash']
        composite=a.out/'composite_nomination';composite.mkdir();provenance={};token_hashes=set()
        for c in spec['candidates']:
            assert sha(Path(c['path']))==c['sha256']
            import hashlib,numpy as np
            rows=json.loads(Path(c['path']).read_text())['requests']
            h=hashlib.sha256(np.asarray([r['input_ids'] for r in rows],np.uint32).tobytes()).hexdigest()
            assert h not in token_hashes;token_hashes.add(h)
            source=a.repair if c['candidate'] in changed else a.original
            (composite/c['candidate']).symlink_to(source/c['candidate'],target_is_directory=True)
            provenance[c['candidate']]=dict(source_job=str(source),manifest_sha256=c['sha256'],ordered_input_sha256=h)
        (composite/'_warmup').symlink_to(a.repair/'_warmup',target_is_directory=True)
        write(composite/'SEARCH_SPEC.json',spec)
        write(composite/'result.json',dict(status='PASS',search_stage='nomination',candidates=[c['candidate'] for c in spec['candidates']],
              primary_repeats=0,output_tokens=16,selection_stage=True,composite=True,search_spec=str(a.spec),search_spec_sha256=sha(a.spec)))
        write(composite/'status.json',dict(status='PASS',composite=True,fully_completed_source_jobs=[str(a.original),str(a.repair)]))
        write(composite/'SOURCE_PROVENANCE.json',dict(status='PASS',cases=provenance,unique_ordered_inputs=33,
              active_runtime_source_hashes_equal=runtime_paths,warmup_full_tokens_cache_exact=True,
              scope='Original31 unchanged valid nominations plus two preregistered repaired nominees; no averaged or combined serving timings'))
        selection=a.out/'nomination_selection'
        step('repaired_nomination_selection',command('pcie_maxgain_select.py','--stage','nomination','--spec',a.spec,'--root',composite,'--out',selection))
        saved=DEST/'nomination_selection_repaired';saved.mkdir(parents=True,exist_ok=False)
        for name in ('CANDIDATES.json','SELECTION.json'):shutil.copyfile(selection/name,saved/name)
        shutil.copyfile(composite/'SOURCE_PROVENANCE.json',saved/'SOURCE_PROVENANCE.json')
        commit(saved,'Validate 33 unique best64 nominations with exact C++ replay and freeze corrected top8 before screen primaries')
        frozen=selection/'CANDIDATES.json'
        for stage in ('screen','final'):
            root=a.out/stage
            step(stage,command('run_pcie_ours_job.py','--arm','R-NEAR','--out',root,'--search-stage',stage,'--search-spec',frozen,'--timeout','10800'))
            if stage=='screen':
                chosen=a.out/'screen_selection'
                step('screen_selection',command('pcie_maxgain_select.py','--stage','screen','--spec',frozen,'--root',root,'--out',chosen))
                frozen=chosen/'CANDIDATES.json';saved=DEST/'screen_selection';saved.mkdir()
                for name in ('CANDIDATES.json','SELECTION.json'):shutil.copyfile(chosen/name,saved/name)
                commit(saved,'Freeze best64 top3 by full64 R/G screen TPOT before counterordered final repetitions')
            step(stage+'_archive',command('pcie_maxgain_report.py','--root',root,'--archive','--commit'))
        report=json.loads((a.out/'final/maxgain_validation.json').read_text());assert report['status']=='PASS'
        write(a.out/'result.json',dict(status='PASS',best_observed=report['best_observed'],scope=report['scope'],completed=done,original_full_plan_goal_still_active=True))
        write(a.out/'status.json',dict(status='PASS',pid=os.getpid(),completed=done))
        print(json.dumps(dict(status='PASS',best_observed=report['best_observed'])),flush=True)
    except BaseException as exc:
        write(a.out/'failure.json',dict(status='FAIL',cause=repr(exc),completed=done,unix=time.time()))
        write(a.out/'status.json',dict(status='FAIL',pid=os.getpid(),cause=repr(exc),completed=done));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--repair',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
