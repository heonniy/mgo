"""Validate and archive a completed native baseline, preserving every repeat."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from pcie_host import write

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
DEST=PKG/'experiments/pcie_topology_ablation_20261009/baselines'
BUDGET=17392730112


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def close(a,b):
    assert abs(a-b)<=1e-6*max(1,abs(b)),(a,b)


def run(root,system):
    assert json.loads((root/'status.json').read_text())['status']=='PASS'
    result=json.loads((root/'result.json').read_text());assert result['status']=='PASS' and not result['smoke']
    expected_system={'deepspeed':'DeepSpeed-ZeRO-Inference','infinity':'MoE-Infinity-repaired','llama':'llama.cpp-sync'}[system]
    assert result['system']==expected_system
    launch=json.loads((root/'launch.json').read_text())
    assert launch['physical_gpus']==[0,1,4,5]
    assert launch['environment']['CUDA_VISIBLE_DEVICES']=='0,1,4,5'
    assert launch['environment']['MGO_PCIE_HOST']=='1' and launch['primary_repeats']==5
    assert launch['runtime']=='native baseline, no forced OURS serial phase order'
    spec=launch['workload'];assert spec['cell']=='R4_C30_B16_L512_O64' and spec['expert_budget_bytes']==BUDGET
    manifests={}
    for phase,expected in [('target','dea64853847b1ba1d4f3e0c8cc112eb8ba84dcde5f1d04fb90adb6fca91d67d6'),('warmup','cf2aeb54933d729f55b8169ca9f0a5a5f951ce773d9d9d63490cfe4105e752bd')]:
        assert sha(Path(spec[phase]['path']))==spec[phase]['sha256']==expected
        rows=json.loads(Path(spec[phase]['path']).read_text())['requests']
        assert len(rows)==64 and len({r['request_id'] for r in rows})==64
        assert all(len(r['input_ids'])==512 for r in rows)
        manifests[phase]=[r['request_id'] for r in rows]
    assert not set(manifests['warmup'])&set(manifests['target'])
    samples=[];tokens=[];pinned=[]
    for repeat in range(6):
        phase='warmup' if repeat==0 else 'target'
        row=json.loads((root/f'repeat{repeat}.json').read_text())
        assert row['status']=='PASS' and not row['smoke'] and row['global_requests']==64 and row['output_tokens']==64
        assert row['system']==expected_system
        assert row['repeat']==repeat
        if system=='deepspeed':
            ranks=[json.loads((root/f'repeat{repeat}_rank{rank}.json').read_text()) for rank in range(4)]
            assert len({r['release_ns'] for r in ranks})==1
            generated=[];request_ids=[]
            for rank,r in enumerate(ranks):
                assert r['finite_logits'] and r['kv_gpu_resident'] and r['cache_start']=='all parameters NOT_AVAILABLE'
                assert r['request_ids']==manifests[phase][rank*16:(rank+1)*16]
                assert r['parameter_budget_bytes']==BUDGET//4 and 0<=r['all_parameter_peak_bytes']<=BUDGET//4
                assert r['host_rss_bytes']>0 and r['pinned_host_bytes']>0
                assert len(r['token_ready_ns'])==64
                assert all(a<b for a,b in zip(r['token_ready_ns'],r['token_ready_ns'][1:]))
                assert r['first_ns']==r['token_ready_ns'][0] and r['end_ns']==r['token_ready_ns'][-1]
                generated.extend(r['tokens']);request_ids.extend(r['request_ids'])
            start=ranks[0]['release_ns'];first=max(r['first_ns'] for r in ranks);end=max(r['end_ns'] for r in ranks)
            pinned.append(sum(r['pinned_host_bytes'] for r in ranks))
        else:
            generated=row['tokens'];request_ids=row['request_ids'];start=row['release_ns']
            stamps=row['token_ready_ns'];assert len(stamps)==64 and all(a<b for a,b in zip(stamps,stamps[1:]))
            first,end=stamps[0],stamps[-1]
            if system=='infinity':
                assert row['expert_budget_per_gpu']==[461*9437184,461*9437184,461*9437184,460*9437184]
                assert row['kv_released'] and row['eam_calls']==48*64
                cache=row['cache_after'];assert cache['capacity_bytes']==BUDGET and cache['peak_accounted_bytes']<=BUDGET
                for rank,budget in enumerate(row['expert_budget_per_gpu']):assert cache[f'gpu_{rank}_peak_charged_bytes']<=budget
                assert cache['resident_bytes']+cache['transition_reserved_bytes']+cache['workspace_bytes']<=BUDGET
                assert row['host_rss_bytes']>0 and len(row['peak_allocated_bytes'])==len(row['peak_reserved_bytes'])==4
                assert 0<=row['pinned_host_bytes']<=row['pinned_host_peak_bytes']
                pinned.append(row['pinned_host_bytes'])
            else:
                assert system=='llama' and row['synchronous_batch'] and row['finite_logits']
                assert row['input_tokens']==512 and row['cpu_threads']==row['cpu_batch_threads']==32
                assert not row['op_offload'] and row['offload_kqv'] and row['expert_placement']=='balanced3'
                assert row['expert_resident_bytes']==12*128*9437184<=BUDGET
                assert row['decode_calls']==63 and row['decode_tokens_per_call']==64
                close(sum(row['decode_step_seconds'])/63,row['TPOT'])
                pinned.append(None)  # Never infer CUDA pinning from RSS/VmLck or static weight placement.
        assert request_ids==manifests[phase] and len(generated)==64
        assert all(len(t)==64 and all(isinstance(v,int) and v>=0 for v in t) for t in generated)
        close(row['TTFT'],(first-start)/1e9);close(row['TPOT'],(end-first)/1e9/63);close(row['E2E'],(end-start)/1e9)
        if 'throughput' in row:close(row['throughput'],4096/row['E2E'])
        if repeat:samples.append(row);tokens.append(generated)
    if system=='deepspeed':
        for rank in range(4):
            leaf=json.loads((root/f'zero_leaf_rank{rank}.json').read_text())
            assert leaf['status']=='PASS' and leaf['blocks']==48 and leaf['leaf_class']=='Qwen3MoeSparseMoeBlock'
            calibration=json.loads((root/f'calibration_rank{rank}.json').read_text())
            assert calibration['status']=='PASS' and calibration['budget_bytes']==BUDGET//4
    if system=='llama':
        cfg=json.loads((root/'config.json').read_text());audit=json.loads((root/'placement_audit.json').read_text())
        assert cfg['cuda_graphs_runtime']=='off' and not cfg['llama_graph_reuse']
        assert cfg['cpu_threads']==32 and cfg['expert_placement']=='balanced3'
        assert cfg['unused_expert_budget_bytes']==BUDGET-12*128*9437184
        assert audit['status']=='PASS' and sorted(audit['gpu_expert_layers_by_device'].values())==[3,3,3,3]
        assert json.loads((root/'kv_placement.json').read_text())['status']=='PASS'
    resource=json.loads((root/'resource_peaks.json').read_text())
    assert resource['physical_gpus']==[0,1,4,5] and resource['host_available_min_bytes']>=128*2**30
    assert set(resource['gpu_used_sampled_peak_bytes'])=={'0','1','4','5'}
    stats={}
    recorded=json.loads((root/'statistics.json').read_text())
    for key in ('TTFT','TPOT','E2E','throughput'):
        values=[row[key] if key in row else 4096/row['E2E'] for row in samples]
        entry=dict(values=values,median=statistics.median(values),mean=statistics.mean(values),sd=statistics.stdev(values),min=min(values),max=max(values))
        for name in ('median','mean','sd','min','max'):close(entry[name],recorded[key][name])
        assert recorded[key]['values']==values
        stats[key]=entry
    report=dict(status='PASS',system=system,root=str(root),statistics=stats,resource=resource,
        workload_sha256={p:spec[p]['sha256'] for p in ('warmup','target')},pinned_host_bytes_by_repeat=pinned,
        per_repeat_token_agreement_with_repeat1=[sum(a==b for ra,rb in zip(tokens[0],other) for a,b in zip(ra,rb))/4096 for other in tokens],
        scope='Native serving schedule; no forced OURS G2G/H2D isolation. All five unprofiled repeats retained. Numerical output agreement is recorded, not task accuracy.',
        pinned_scope=('CPU offload parameter tensors successfully reported pinned per rank; excludes cached/unrelated Torch host allocations' if system=='deepspeed' else 'Native expert HostCachingAllocator payload including cached blocks; excludes other allocators' if system=='infinity' else 'Unknown CUDA staging-pinned bytes; never reported as zero from static placement/RSS'))
    write(root/'baseline_validation.json',report)
    with (root/'baseline_table.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['system','repeats','TTFT_s','TPOT_s','E2E_s','TPOT_sd_s'])
        writer.writerow([system,5,stats['TTFT']['median'],stats['TPOT']['median'],stats['E2E']['median'],stats['TPOT']['sd']])
    lines=[f'# Native baseline: {system}', '', report['scope'], '',
        '| Metric | Median | Mean | SD | Min | Max |', '|---|---:|---:|---:|---:|---:|']
    for key,entry in stats.items():lines.append('| '+key+' | '+' | '.join(f'{entry[n]:.6f}' for n in ('median','mean','sd','min','max'))+' |')
    lines += ['', 'All native inputs are the original frozen headline64; the selected best64 search input is separate. Every request generates exactly64 output tokens with no EOS shortening.', '',
        report['pinned_scope']+'.', '', resource['scope'], '',
        'RSS sums can double-count shared pages. Sampled NVML device-used peaks include context/driver allocations and may miss transients; framework allocator peaks are retained separately where available. No unmeasured pinned value is substituted by zero.', '',
        'DeepSpeed is generic native ZeRO-3 CPU parameter offload with MoE leaves, not FastGen. Infinity is the owner-repaired/adapted Qwen3 implementation. llama.cpp is synchronous balanced3,32 threads, CUDA graphs/reuse OFF, leaving307 C30 expert slots unused.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    return report


def archive(root,system,commit):
    target=DEST/system;target.mkdir(parents=True,exist_ok=False);inventory={}
    if commit:assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(root);dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        compressed=path.stat().st_size>128*1024 and path.suffix in ('.json','.jsonl','.csv')
        if compressed:
            dest=dest.with_name(dest.name+'.gz')
            with path.open('rb') as src,dest.open('wb') as dst:
                with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
            with gzip.open(dest,'rb') as f:
                h=hashlib.sha256()
                for block in iter(lambda:f.read(2**20),b''):h.update(block)
            assert h.hexdigest()==sha(path)
        else:shutil.copyfile(path,dest)
        inventory[str(rel)]=dict(source_path=str(path),source_sha256=sha(path),stored_path=str(dest.relative_to(target)),stored_sha256=sha(dest))
    write(target/'SOURCE_INVENTORY.json',dict(status='PASS',files=inventory))
    if commit:
        subprocess.run(['git','add','--',str(target.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',f'Complete native {system} full64 baseline with every repeat and measured resource accounting'],cwd=REPO,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--system',choices=('deepspeed','infinity','llama'),required=True)
    p.add_argument('--archive',action='store_true');p.add_argument('--commit',action='store_true')
    a=p.parse_args();report=run(a.root,a.system)
    if a.archive:archive(a.root,a.system,a.commit)
    print(json.dumps(dict(status='PASS',system=a.system,statistics=report['statistics'])))
