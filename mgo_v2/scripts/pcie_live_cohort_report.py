"""Validate the complete common seven-policy cohort and commit every arm."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from pcie_host import write
from pcie_maxgain_select import verify_generation
from pcie_policy_report import summarize_arm,expected_quota
from pcie_tpot_optimization_report import partition
from pcie_assignment_report import assignments
from validate_pcie_phases import validate

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
DEST=PKG/'experiments/pcie_topology_ablation_20261009/stage2_grouped'
ARMS=['R-NEAR','G-NEAR','G-BR','G-CA','G-NUMA-CA','R-BR','R-CA']
COMPARISONS=[('R-NEAR','G-NEAR'),('G-BR','G-CA'),('G-CA','G-NUMA-CA'),
             ('G-NUMA-CA','G-NEAR'),('G-BR','G-NEAR'),('R-BR','G-BR'),('R-CA','G-CA')]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(root):
    cohort=json.loads((root/'result.json').read_text())
    assert cohort['status']=='PASS' and cohort['sequence']=='stage2_grouped' and cohort['arms']==ARMS
    assert json.loads((root/'status.json').read_text())['status']=='PASS'
    config=json.loads((root/'config.json').read_text())
    assert config['sequence']=='stage2_grouped' and not config['overlap']
    assert config['physical_gpus']==[0,1,4,5] and config['cache_slots']==1843
    costs=json.loads((root/'peer_costs_frozen.json').read_text());assert sha(Path(costs['path']))==costs['sha256']
    spec=next(c for c in json.loads(Path('/data2/esjung/datasets/frozen_pcie_topology_20261009/WORKLOADS.json').read_text())['cells'] if c['cell']=='R4_C30_B16_L512_O64')
    manifests={phase:json.loads(Path(spec[phase]['path']).read_text())['requests'] for phase in ('warmup','target')}
    recorded=json.loads((root/'WORKLOAD_MANIFESTS.json').read_text())
    for phase in ('warmup','target'):
        assert sha(Path(spec[phase]['path']))==spec[phase]['sha256']==recorded[phase]['sha256']
        assert len(manifests[phase])==64 and all(len(x['input_ids'])==512 for x in manifests[phase])
    assert not ({x['request_id'] for x in manifests['warmup']}&{x['request_id'] for x in manifests['target']})
    report=dict(status='PASS',root=str(root),arms={},comparisons=[],peer_costs=costs,
                primary_repeats=cohort['primary_repeats'],unique_pinned_bytes=108*2**30,
                scope='Independent live greedy generations on the original frozen headline64; fixed-snapshot same-miss evidence is separate',
                runtime='Common grouped decode, native C++ metadata/controller and native prefill; globally serialized G2G/H2D/compute/return',
                rss_note='Post-generation RSS per rank includes shared source mappings; summing rank RSS double-counts physical source pages')
    token_arrays={}
    for arm in ARMS:
        path=root/arm
        for rank in range(4):
            prefill=json.loads((path/f'prefill_validation_rank{rank}.json').read_text())
            assert prefill['status']=='PASS' and prefill['state']['status']=='PASS'
            assert len(prefill['per_layer_numerics'])==48
            assert all(np.isfinite(c['relative_l2']) and c['relative_l2']<=.01 for c in prefill['per_layer_numerics'])
            warm=json.loads((path/f'grouped_repeat0_rank{rank}.json').read_text())
            assert warm['all_ready'] and not warm['expert_h2d_compute_overlap'] and warm['decode_calls']==3024
            assert len(warm['checks'])==48 and all(c['finite'] and c['relative_l2']<=.01 for c in warm['checks'])
            meta=json.loads((path/f'metadata_repeat0_rank{rank}.json').read_text())
            assert meta['status']=='PASS' and meta['wire_checks']==48 and meta['calls']==3024
            receipt=json.loads((path/f'repeat0_rank{rank}.json').read_text())
            assert receipt['finite_logits'] and receipt['expert_cache_start']=='empty'
            assert receipt['request_ids']==[r['request_id'] for r in manifests['warmup'][rank*16:(rank+1)*16]]
        trace,ranks=verify_generation(path,1,64)
        for event,row in enumerate(trace):np.testing.assert_array_equal(row[48:52],expected_quota(int(row[19]),event,arm.startswith('G-')))
        for repeat in list(range(2,cohort['primary_repeats']+1))+[-1]:
            other,rr=verify_generation(path,repeat,64);np.testing.assert_array_equal(trace,other)
            assert [r['tokens'] for r in ranks]==[r['tokens'] for r in rr]
            assert [r['validation']['state_hash'] for r in ranks]==[r['validation']['state_hash'] for r in rr]
        for repeat in list(range(1,cohort['primary_repeats']+1))+[-1]:
            for rank in range(4):
                receipt=json.loads((path/f'repeat{repeat}_rank{rank}.json').read_text())
                assert receipt['host_rss_bytes']>0 and receipt['pinned_host_bytes']==54*2**30
                assert receipt['pcie_g2g_first_serial'] and receipt['prefetch_off'] and receipt['native_prefill']
                assert receipt['profiled']==(repeat==-1)
                assert receipt['request_ids']==[r['request_id'] for r in manifests['target'][rank*16:(rank+1)*16]]
        validate(path,(-1,))
        entry=dict(live=summarize_arm(path),diagnostic=partition(path))
        entry['assignments']=assignments(path,trace)
        entry['host_rss_bytes']=[[json.loads((path/f'repeat{repeat}_rank{rank}.json').read_text())['host_rss_bytes'] for rank in range(4)] for repeat in range(1,cohort['primary_repeats']+1)]
        report['arms'][arm]=entry
        token_arrays[arm]=np.concatenate([np.asarray(r['tokens']) for r in ranks])
        assert token_arrays[arm].shape==(64,64)
    for left,right in COMPARISONS:
        l=report['arms'][left]['live']['primary']['TPOT']['median'];r=report['arms'][right]['live']['primary']['TPOT']['median']
        same=token_arrays[left]==token_arrays[right]
        report['comparisons'].append(dict(left=left,right=right,left_TPOT_median=l,right_TPOT_median=r,
             right_reduction_percent=100*(1-r/l),full_token_agreement=float(same.mean()),first_token_agreement=float(same[:,0].mean()),
             matching_prefix_tokens=[int(np.flatnonzero(~row)[0]) if not row.all() else 64 for row in same]))
    fields=['arm','primary_repeats','TTFT_median_s','TPOT_median_s','E2E_median_s','TPOT_mean_s','TPOT_sd_s','TPOT_min_s','TPOT_max_s','H2D_GiB','mean_M','M_mod4_2_fraction']
    rows=[]
    for arm,entry in report['arms'].items():
        p=entry['live']['primary'];d=entry['live']['live_decode'][0]
        rows.append([arm,cohort['primary_repeats'],p['TTFT']['median'],p['TPOT']['median'],p['E2E']['median'],p['TPOT']['mean'],p['TPOT']['sd'],p['TPOT']['min'],p['TPOT']['max'],d['H2D_bytes']/2**30,d['miss_mean'],d['M_mod_4_equals_2_fraction']])
    with (root/'main_table_ours.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(fields);writer.writerows(rows)
    tex=['\\begin{tabular}{lrrr}','Policy & TTFT (s) & TPOT (s) & E2E (s) \\\\','\\hline']
    tex.extend(f'{row[0]} & {row[2]:.6f} & {row[3]:.6f} & {row[4]:.6f} \\\\' for row in rows)
    tex.append('\\end{tabular}');(root/'main_table_ours.tex').write_text('\n'.join(tex)+'\n')
    lines=['# Seven-policy common grouped live cohort','',report['scope']+'.','',report['runtime']+'.','',
           '| Policy | TTFT (s) | TPOT (s) | E2E (s) | TPOT SD (s) | H2D (GiB) |',
           '|---|---:|---:|---:|---:|---:|']
    lines.extend(f'| {r[0]} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[6]:.6f} | {r[9]:.3f} |' for r in rows)
    lines.extend(['','All primary repetitions are retained. Diagnostics and warmups do not enter these statistics.',
                  '',report['rss_note']+'.','',
                  'Actual assigned experts and quota rows are joined only after diagnostic tokens/cache/full traces match all primaries. Nested CPU spans are not summed as separate phases; diagnostic frontier partitions cover the complete diagnostic TPOT.',
                  '', 'Independent greedy trajectories can change miss sets, cache states and quota values. Same-miss placement evidence is in stage2_fixed_miss; this table measures live generation.',
                  '', 'MoE-Infinity, DeepSpeed and llama.cpp results are not included in this OURS-only table. Their required new-host measurements remain pending.'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(root/'live_cohort_validation.json',report)
    return report


def archive(root,report,do_commit):
    assert not DEST.exists(),'Archive already exists; inspect preserved state before resuming'
    if do_commit:assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
    def copy(source,target):
        target.mkdir(parents=True,exist_ok=False);inventory={}
        for path in sorted(source.iterdir()):
            if not path.is_file():continue
            dest=target/path.name;digest=sha(path)
            compressed=path.stat().st_size>128*1024 and path.suffix in ('.json','.jsonl','.npy','.csv')
            if compressed:
                dest=dest.with_name(dest.name+'.gz')
                with path.open('rb') as src,dest.open('wb') as dst:
                    with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
                with gzip.open(dest,'rb') as stream:
                    h=hashlib.sha256()
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                assert h.hexdigest()==digest
            else:shutil.copyfile(path,dest)
            inventory[path.name]=dict(source_path=str(path),source_sha256=digest,stored_path=dest.name,stored_sha256=sha(dest),bytes=path.stat().st_size)
        write(target/'SOURCE_INVENTORY.json',dict(status='PASS',files=inventory))
    def commit(path,message):
        if do_commit:
            subprocess.run(['git','add','--',str(path.relative_to(REPO))],cwd=REPO,check=True)
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m',message],cwd=REPO,check=True)
    commits={}
    for arm in ARMS:
        copy(root/arm,DEST/arm);commit(DEST/arm,f'Complete common grouped live experiment {arm} with full64 repeats and isolated assignment breakdown')
        if do_commit:commits[arm]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    copy(root,DEST/'cohort');write(DEST/'cohort/ARM_COMMITS.json',commits)
    commit(DEST/'cohort','Report all seven grouped live policies, measured peer costs and controlled versus live trajectory distinctions')
    print(json.dumps(dict(status='PASS',destination=str(DEST),arm_commits=commits)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',action='store_true');p.add_argument('--commit',action='store_true')
    a=p.parse_args();report=run(a.root)
    if a.archive:archive(a.root,report,a.commit)
