"""Validate search evidence, report finalist medians and archive each experiment."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from pcie_host import write
from pcie_maxgain_select import verify_generation
from pcie_policy_report import summarize_arm,expected_quota
from pcie_tpot_optimization_report import partition
from validate_pcie_phases import validate

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
DEST=PKG/'experiments/pcie_topology_ablation_20261009/maxgain64'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def validate_stage(root):
    cohort=json.loads((root/'result.json').read_text());assert cohort['status']=='PASS'
    assert json.loads((root/'status.json').read_text())['status']=='PASS'
    spec=json.loads((root/'SEARCH_SPEC.json').read_text());stage=cohort['search_stage']
    assert spec['status']=='FROZEN' and cohort['candidates']==[c['candidate'] for c in spec['candidates']]
    assert sha(Path(cohort['search_spec']))==cohort['search_spec_sha256']
    for arm in ('R-NEAR','G-NEAR'):
        warm=root/'_warmup'/arm
        for r in range(4):
            group=json.loads((warm/f'grouped_repeat0_rank{r}.json').read_text())
            assert group['decode_calls']==3024 and group['all_ready'] and not group['expert_h2d_compute_overlap']
            assert len(group['checks'])==48 and all(c['finite'] and c['relative_l2']<=.01 for c in group['checks'])
            receipt=json.loads((warm/f'repeat0_rank{r}.json').read_text())
            assert receipt['metadata_wire_checks']==48 and receipt['finite_logits']
    report=dict(status='PASS',stage=stage,root=str(root),arms={},selection_is_not_final_estimate=stage!='final')
    for c in spec['candidates']:
        for arm in (('G-NEAR',) if stage=='nomination' else ('R-NEAR','G-NEAR')):
            path=root/c['candidate']/arm;n=16 if stage=='nomination' else 64
            trace,ranks=verify_generation(path,1,n)
            for event,row in enumerate(trace):np.testing.assert_array_equal(row[48:52],expected_quota(int(row[19]),event,arm=='G-NEAR'))
            entry=dict(status='PASS',candidate_manifest=c['path'],candidate_manifest_sha256=c['sha256'])
            assert sha(Path(c['path']))==c['sha256']
            if stage=='final':
                validate(path,(-1,));entry['live']=summarize_arm(path);entry['diagnostic']=partition(path)
                with np.load(path/'actual_fetches_repeat-1.npz') as capture:
                    fetches=capture['fetches'];assert int(capture['events'])==3072
                assert fetches.shape[1]==7 and np.all(fetches[:,5]==0)
                assert len(np.unique(fetches[:,0]*6144+fetches[:,2]))==len(fetches),'Duplicate mandatory expert within an event'
                counts=np.zeros((3072,4),np.int64)
                np.add.at(counts,(fetches[:,0],fetches[:,1]),1)
                np.testing.assert_array_equal(counts,trace[:,48:52])
                assert np.all(fetches[:,2]//128==fetches[:,0]%48)
                capacities=np.array([459,459,459,458],np.int64)
                assert np.all(fetches[:,3]>=0) and np.all(fetches[:,3]<capacities[fetches[:,1]])
                assert np.all(fetches[:,6]>=0) and np.all(fetches[:,6]<capacities[fetches[:,1]]+2)
                with (path/'assigned_experts.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['event','layer','phase','rank','physical_gpu','expert_id','key','logical_main_slot','physical_slot','victim','replica'])
                    for event,rank,key,slot,victim,rep,physical in fetches:
                        writer.writerow([event,event%48,'prefill' if event<48 else 'decode',rank,[0,1,4,5][rank],key%128,key,slot,physical,victim,rep])
                with (path/'phase_timing.jsonl').open('w') as stream:
                    for rank in range(4):
                        for event in json.loads((path/f'phases_repeat-1_rank{rank}.json').read_text()):
                            stream.write(json.dumps(dict(event,physical_gpu=[0,1,4,5][rank],diagnostic=True))+'\n')
                entry['actual_assignment_validation']=dict(status='PASS',events=3072,fetches=len(fetches),
                     every_rank_quota_exact=True,legal_main_and_physical_slots=True,no_replication=True,
                     scope='Actual diagnostic controller fetches; diagnostic tokens/cache/full policy trace match unprofiled repeats',
                     additional_diagnostic_overhead='Copying assignment arrays is diagnostic-only and included in PLAN residual')
                for repeat in list(range(2,cohort['primary_repeats']+1))+[-1]:
                    other,rr=verify_generation(path,repeat,64);np.testing.assert_array_equal(trace,other)
                    assert [r['tokens'] for r in ranks]==[r['tokens'] for r in rr]
                    assert [r['validation']['state_hash'] for r in ranks]==[r['validation']['state_hash'] for r in rr]
                # Identical per-arm trajectories let a separate diagnostic
                # supply assignment detail without logging inside primaries.
                offsets=np.r_[0,np.cumsum(counts.sum(axis=1))]
                assignments=[json.dumps(fetches[offsets[event]:offsets[event+1],1:].tolist(),separators=(',',':')) for event in range(3072)]
                quota_path=path/'quota_event_trace.csv'
                with quota_path.open(newline='') as stream:
                    reader=csv.DictReader(stream);fields=list(reader.fieldnames);quota_rows=list(reader)
                fields=[f for f in fields if f not in ('assigned_experts','assignment_provenance')]
                with quota_path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields+['assigned_experts','assignment_provenance']);writer.writeheader()
                    for row in quota_rows:
                        row['assigned_experts']=assignments[int(row['event'])]
                        row['assignment_provenance']='actual separate repeat-1 diagnostic; full tokens/cache/policy trace match; [rank,key,logical_slot,victim,replica,physical_slot]'
                        writer.writerow(row)
            report['arms'][c['candidate']+'/'+arm]=entry
    if stage=='final':
        comparisons=[]
        for c in spec['candidates']:
            r=report['arms'][c['candidate']+'/R-NEAR']['live']['primary']['TPOT']
            g=report['arms'][c['candidate']+'/G-NEAR']['live']['primary']['TPOT']
            token_arrays={}
            for arm in ('R-NEAR','G-NEAR'):
                token_arrays[arm]=np.concatenate([np.asarray(json.loads((root/c['candidate']/arm/f'repeat1_rank{rank}.json').read_text())['tokens']) for rank in range(4)])
            same=token_arrays['R-NEAR']==token_arrays['G-NEAR'];assert same.shape==(64,64)
            config=json.loads(Path('/data2/esjung/models/Qwen3-30B-A3B-Instruct-2507/config.json').read_text())
            eos=config['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
            eos_stats={}
            for arm,array in token_arrays.items():
                hits=np.isin(array,eos);first=[int(np.flatnonzero(row)[0])+1 if row.any() else None for row in hits]
                eos_stats[arm]=dict(eos_token_ids=eos,requests_with_eos=int(hits.any(axis=1).sum()),
                    total_eos_tokens=int(hits.sum()),first_eos_token_positions=first,
                    early_stop=False,all_requests_generate_exactly64=True)
            requests=json.loads(Path(c['path']).read_text())['requests']
            prefix_hashes=[hashlib.sha256(np.asarray(row['input_ids'],np.uint32).tobytes()).hexdigest() for row in requests]
            families=Counter(row['conversation_key'] for row in requests if row.get('conversation_key') is not None)
            unknown=sum(row.get('conversation_key') is None for row in requests)
            provenance=dict(distinct_source_rows=len({row['source_row'] for row in requests}),
                    distinct_input_token_prefixes=len(set(prefix_hashes)),
                    ordered_input_tokens_uint32_sha256=hashlib.sha256(np.asarray([row['input_ids'] for row in requests],np.uint32).tobytes()).hexdigest(),
                    known_conversation_families=len(families),unknown_family_requests=unknown,
                    conversation_family_multiplicities=dict(families),source_dataset_sha256=spec['source_dataset_sha256'],
                    note='Distinct dataset split records and inputs can come from the same original conversation; they are not necessarily independent conversations')
            assert provenance['distinct_source_rows']==64 and provenance['distinct_input_token_prefixes']==64
            if c.get('ordered_input_tokens_uint32_sha256'):assert provenance['ordered_input_tokens_uint32_sha256']==c['ordered_input_tokens_uint32_sha256']
            if c.get('conversation_families') is not None:assert len(families)==c['conversation_families'] and not unknown
            comparisons.append(dict(candidate=c['candidate'],R=r,G=g,gain_percent=(1-g['median']/r['median'])*100,
                    request_ids=c['request_ids'],manifest_path=c['path'],manifest_sha256=c['sha256'],
                    input_provenance=provenance,
                    R_G_full_token_agreement=float(same.mean()),R_G_first_token_agreement=float(same[:,0].mean()),
                    matching_prefix_tokens=[int(np.flatnonzero(~row)[0]) if not row.all() else 64 for row in same],eos=eos_stats))
        comparisons.sort(key=lambda c:(-c['gain_percent'],c['candidate']))
        report.update(comparisons=comparisons,best_observed=comparisons[0],
             scope='Best observed among the preregistered screened candidates; no global-optimum or corpus-average claim',
             trajectory_note='Independent live greedy R/G generations may change routing, miss counts and cache trajectories; same-input repetitions within each arm must match exactly')
        with (root/'maxgain_table.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['candidate','R_median_s','G_median_s','gain_percent','R_sd','G_sd','R_min','R_max','G_min','G_max'])
            for row in comparisons:w.writerow([row['candidate'],row['R']['median'],row['G']['median'],row['gain_percent'],row['R']['sd'],row['G']['sd'],row['R']['min'],row['R']['max'],row['G']['min'],row['G']['max']])
        lines=['# Best observed real ShareGPT batch64','',report['scope']+'.','',
            '| Candidate | R TPOT (s) | G TPOT (s) | Reduction |','|---|---:|---:|---:|']
        for row in comparisons:lines.append(f"| {row['candidate']} | {row['R']['median']:.6f} | {row['G']['median']:.6f} | {row['gain_percent']:.3f}% |")
        best=comparisons[0]
        lines.extend(['',f"Winner token manifest: `{best['manifest_path']}`; SHA256 `{best['manifest_sha256']}`.",
             '',f"Winner provenance: {best['input_provenance']['distinct_source_rows']} distinct dataset source rows and {best['input_provenance']['distinct_input_token_prefixes']} distinct input prefixes, from {best['input_provenance']['known_conversation_families']} known original conversation families ({best['input_provenance']['unknown_family_requests']} rows with unknown family).",
             '',best['input_provenance']['note']+'. Exact family multiplicities and dataset hashes are retained in maxgain_validation.json.',
             '',report['trajectory_note']+'.',
             '', 'R/G full token agreement, matching prefixes and EOS positions are retained in maxgain_validation.json. Every request runs all 64 output steps even if EOS occurs; any resulting routing/cache differences are part of live serving evidence, not proof of a same-trace quota-only speedup.',
             '', 'Nomination and single-pair screen timing selected the finalists and do not enter the final median. All candidates and screen pairs remain reported. Finalist batches were frozen before counterordered final repetitions; selecting the largest final gain still has selection bias.',
             '',f"Request source IDs (64): {best['request_ids']}"])
        best_arms={arm:report['arms'][best['candidate']+'/'+arm] for arm in ('R-NEAR','G-NEAR')}
        lines.extend(['','## Winner live traffic and separate diagnostic breakdown','',
             '| Policy | Mean M | M mod 4 = 2 | H2D GiB | Group A/B H2D GiB | Reload fetches |','|---|---:|---:|---:|---:|---:|'])
        for arm,entry in best_arms.items():
            live=entry['live']['live_decode'][0]
            groups='/'.join(f'{v/2**30:.3f}' for v in live['group_H2D_bytes'])
            lines.append(f"| {arm} | {live['miss_mean']:.3f} | {live['M_mod_4_equals_2_fraction']*100:.2f}% | {live['H2D_bytes']/2**30:.3f} | {groups} | {live['reload_fetches']} |")
        lines.extend(['','The live H2D and cache totals include changed greedy trajectories; identical miss sets are not assumed.',
             '', '| Policy | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |','|---|---:|---:|---:|---:|---:|---:|'])
        for arm,entry in best_arms.items():
            phases=entry['diagnostic']['seconds_per_token']
            values=' | '.join(f'{phases[key]:.6f}' for key in ('metadata_plan','forward','h2d','compute','return','attention_router_other'))
            lines.append(f'| {arm} | {values} |')
        lines.extend(['','Phase values are seconds per token from separate timers-only diagnostics, including rendezvous and diagnostic assignment-copy overhead. Their endpoint sum equals diagnostic TPOT; they are excluded from unprofiled primary statistics. Differences between these separate diagnostics do not establish causal contributions to the primary R/G gap.',
             '', 'R/G per-arm output agreement, actual expert assignments, all-rank stage completions, nested native controller/metadata call costs and all repetitions remain inspectable in the archived validation/analysis/CSV/JSONL artifacts.'])
        (root/'MAX_GAIN_RESULTS.md').write_text('\n'.join(lines)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
        x=np.arange(len(comparisons));width=.34
        for j,(arm,color) in enumerate((('R','#0072B2'),('G','#D55E00'))):
            values=[c[arm] for c in comparisons];med=np.array([v['median'] for v in values])
            ax.bar(x+(j-.5)*width,med,width,label=arm+'-NEAR',color=color)
            ax.errorbar(x+(j-.5)*width,med,yerr=np.array([[v['median']-v['min'] for v in values],
                 [v['max']-v['median'] for v in values]]),fmt='none',ecolor='#333333',capsize=3)
        ax.set_xticks(x,[c['candidate']+'\n'+f"{c['gain_percent']:.2f}% reduction" for c in comparisons])
        ax.set_ylabel('Unprofiled median TPOT (s/token)');ax.set_ylim(bottom=0);ax.legend()
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title('Best observed real ShareGPT batches of 64\nFinal repetitions; bars show min–max range')
        fig.savefig(root/'maxgain64.pdf');fig.savefig(root/'maxgain64.png',dpi=160);plt.close(fig)
    write(root/'maxgain_validation.json',report);return report


def archive(root,report,commit,label=None):
    stage=report['stage'];label=label or stage
    assert Path(label).name==label and label not in ('.','..')
    destination=DEST/label;destination.mkdir(parents=True,exist_ok=False)
    if commit:assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip(),'Unrelated staged files present'
    def copy_tree(source,target,external_captures=False):
        target.mkdir(parents=True,exist_ok=True);inventory={}
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(source);digest=sha(p)
            if external_captures and p.name=='current_routing.npz':
                inventory[str(rel)]=dict(external_path=str(p),sha256=digest,bytes=p.stat().st_size,reason='Raw current-routing capture retained under data2; no input-token manifest in Git')
                continue
            dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if p.stat().st_size>128*1024 and p.suffix in ('.npy','.csv','.json','.jsonl'):
                dest=dest.with_name(dest.name+'.gz')
                with p.open('rb') as src,dest.open('wb') as dst:
                    with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
                with gzip.open(dest,'rb') as stream:
                    h=hashlib.sha256()
                    for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
                assert h.hexdigest()==digest
            else:shutil.copyfile(p,dest)
            inventory[str(rel)]=dict(source_path=str(p),source_sha256=digest,stored_path=str(dest.relative_to(target)),stored_sha256=sha(dest),bytes=p.stat().st_size)
        write(target/'SOURCE_INVENTORY.json',dict(status='PASS',files=inventory))
    def checkpoint(path,message):
        if commit:
            subprocess.run(['git','add','--',str(path.relative_to(REPO))],cwd=REPO,check=True)
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m',message],cwd=REPO,check=True)
    for key in report['arms']:
        candidate,arm=key.split('/');copy_tree(root/candidate/arm,destination/candidate/arm,True)
        checkpoint(destination/candidate/arm,f'Validate and archive best64 {stage} experiment {candidate} {arm}; preserve distinct selection and final evidence')
    common=destination/'cohort';common.mkdir()
    for p in root.iterdir():
        if p.is_file():shutil.copyfile(p,common/p.name)
    copy_tree(root/'_warmup',common/'warmups')
    checkpoint(common,f'Archive best64 {stage} cohort, full64 warmup numerical gates and all-rank validation')
    print(json.dumps(dict(status='PASS',stage=stage,destination=str(destination),committed=commit)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',action='store_true');p.add_argument('--commit',action='store_true');p.add_argument('--archive-label')
    a=p.parse_args();r=validate_stage(a.root)
    if a.archive:archive(a.root,r,a.commit,a.archive_label)
