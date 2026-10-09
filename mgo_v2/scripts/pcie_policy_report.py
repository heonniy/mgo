"""Recompute primary metrics and separate phase diagnostics from live policy traces."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG/'scripts'))
from br_carep_cpu import METRICS
from pcie_host import write

EB=9437184


def expected_quota(m,event,group):
    if not group:return [m//4+int(r<m%4) for r in range(4)]
    result=[]
    for g in range(2):
        total=m//2+int(bool(m%2) and g==event%2)
        result.extend(total//2+int(bool(total%2) and r==(event//2+1)%2) for r in range(2))
    return result


def summarize_arm(root):
    result=json.loads((root/'result.json').read_text());assert result['status']=='PASS' and not result['profiled']
    repeats=result['primary_repeats'];assert repeats in (3,5)
    shape=json.loads((root/'repeat1.json').read_text())
    n=int(shape['output_tokens']);global_batch=int(shape['global_requests']);events=n*48
    assert n in (64,128,256) and global_batch in (64,128,256)
    summaries=[];token_hashes=[];state_hashes=[]
    fields=['repeat','event','layer','phase','M','M_mod_4','q0','q1','q4','q5','group_A','group_B','H2D_bytes','group_A_bytes','group_B_bytes','first_fetches','reload_fetches','evictions','cross_group_expert_routes','within_group_expert_routes','cross_group_token_rank_pairs','within_group_token_rank_pairs','rank0_rows','rank1_rows','rank4_rows','rank5_rows']
    with (root/'quota_event_trace.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(fields)
        for repeat in range(1,repeats+1):
            primary=json.loads((root/f'repeat{repeat}.json').read_text());assert not primary['profiled']
            ranks=[json.loads((root/f'repeat{repeat}_rank{r}.json').read_text()) for r in range(4)]
            assert all(x['finite_logits'] and x['no_compile'] and not x['profiled'] for x in ranks)
            assert all(x['output_tokens']==n and x['cache_before']['scheduler']['background_copies']==0 for x in ranks)
            assert all(x['validation']['physical_slots']==1843 and x['debug_plan_checks']==events for x in ranks)
            assert len({x['release_ns'] for x in ranks})==1
            release=ranks[0]['release_ns'];first=max(x['first_ns'] for x in ranks);end=max(x['end_ns'] for x in ranks)
            for key,value in dict(TTFT=(first-release)/1e9,TPOT=(end-first)/1e9/(n-1),E2E=(end-release)/1e9).items():
                assert abs(primary[key]-value)<1e-10
            assert len({x['validation']['state_hash'] for x in ranks})==1
            assert len({i for x in ranks for i in x['request_ids']})==global_batch
            token_hashes.append([x['argmax_hash'] for x in ranks]);state_hashes.append(ranks[0]['validation']['state_hash'])
            trace=np.load(root/f'policy_trace_repeat{repeat}_rank0.npy');assert trace.shape==(events,61)
            for r in range(1,4):np.testing.assert_array_equal(trace,np.load(root/f'policy_trace_repeat{repeat}_rank{r}.npy'))
            quota=np.load(root/f'quota_repeat{repeat}_rank0.npy');np.testing.assert_array_equal(quota[:,1:],trace[:,48:52])
            for event,row in enumerate(trace):
                m=int(row[19]);q=row[48:52].astype(int);group=root.name.startswith('G-')
                assert row[52]==event
                np.testing.assert_array_equal(q,expected_quota(m,event,group))
                writer.writerow([repeat,event,event%48,'prefill' if event<48 else 'decode',m,m%4,*q,q[:2].sum(),q[2:].sum(),m*EB,q[:2].sum()*EB,q[2:].sum()*EB,*row[[20,21,23,53,54,55,56]].astype(int),*row[57:61].astype(int)])
            decode=trace[48:];counts=decode[:,19].astype(int);values,freq=np.unique(counts,return_counts=True)
            summaries.append(dict(repeat=repeat,events=len(decode),miss_histogram=dict(zip(map(str,values),map(int,freq))),
                miss_mean=float(counts.mean()),M_mod_4_equals_2_fraction=float(np.mean(counts%4==2)),
                first_fetches=int(decode[:,20].sum()),reload_fetches=int(decode[:,21].sum()),evictions=int(decode[:,23].sum()),
                H2D_bytes=int(counts.sum())*EB,group_H2D_bytes=(decode[:,48:52].sum(axis=0).reshape(2,2).sum(axis=1)*EB).astype(np.int64).tolist(),
                forward_wire_bytes=sum(x['forward_wire_bytes'] for x in ranks),return_wire_bytes=sum(x['return_wire_bytes'] for x in ranks),
                cross_group_token_rank_pairs=int(decode[:,55].sum()),within_group_token_rank_pairs=int(decode[:,56].sum()),
                forward_cross_group_activation_bytes=int(decode[:,55].sum())*4096,
                projected_critical_rank_rows_mean=float(decode[:,57:61].max(axis=1).mean()),
                peak_allocated_bytes=[x['peak_allocated_bytes'] for x in ranks],peak_reserved_bytes=[x['peak_reserved_bytes'] for x in ranks]))
    assert all(x==token_hashes[0] for x in token_hashes) and len(set(state_hashes))==1
    phase_ranks=[json.loads((root/f'phases_repeat-1_rank{r}.json').read_text()) for r in range(4)]
    profiles={}
    for stage in ('metadata_plan_complete','forward_a2a','forward_global_complete','h2d_only','h2d_global_complete','expert_compute','compute_global_complete','return_a2a','return_global_complete'):
        durations=np.array([[(row[stage][1]-row[stage][0])/1e6 for row in rows[48:]] for rows in phase_ranks])
        profiles[stage]=dict(rank_mean_ms=durations.mean(axis=1).tolist(),critical_rank_mean_ms=float(durations.max(axis=0).mean()),critical_rank_median_ms=float(np.median(durations.max(axis=0))))
    for field in ('h2d_cuda_service_ms','compute_cuda_ms','controller_ms','expert_groups'):
        values=np.array([[row[field] for row in rows[48:]] for rows in phase_ranks])
        profiles[field]=dict(rank_mean=values.mean(axis=1).tolist(),critical_rank_mean=float(values.max(axis=0).mean()))
    profile_notes=['Separate diagnostic repeat -1; excluded from primary and stability statistics.',
       'Host intervals end after CUDA physical synchronization; CPU rendezvous durations include waiting.',
       'metadata_plan_complete is the completion rendezvous; it excludes preceding metadata/controller/layout work.',
       'H2D CUDA span includes enqueue gaps; it is a stream window, not a sum of pure DMA durations.',
       'Captured G-NEAR diagnostic controller_ms includes state capture; replay controller timings exclude state restore/capture.',
       'Cross-group forward bytes count unique token-rank pairs times 4096; NCCL protocol overhead and self traffic are excluded.']
    report=dict(status='PASS',arm=root.name,primary_repeats=repeats,primary=result['statistics'],live_decode=summaries,
                deterministic_tokens_and_final_cache_across_repeats=True,diagnostic_repeat=-1,phase_summary=profiles,notes=profile_notes)
    write(root/'analysis.json',report)
    return report


def main(root):
    cohort=json.loads((root/'result.json').read_text());assert cohort['status']=='PASS'
    reports={arm:summarize_arm(root/arm) for arm in cohort['arms']}
    write(root/'policy_analysis.json',dict(status='PASS',arms=reports,scope='Live independent generations: quota formulas fixed, future cache/miss trajectories can differ'))
    columns=['arm','primary_repeats','TTFT_median_s','TPOT_median_s','E2E_median_s','TPOT_mean_s','TPOT_sample_sd_s','TPOT_min_s','TPOT_max_s','decode_H2D_GiB','decode_M_mean','M_mod4_2_fraction']
    with (root/'main_table.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        for arm,report in reports.items():
            p=report['primary'];d=report['live_decode'][0]
            writer.writerow([arm,report['primary_repeats'],p['TTFT']['median'],p['TPOT']['median'],p['E2E']['median'],p['TPOT']['mean'],p['TPOT']['sd'],p['TPOT']['min'],p['TPOT']['max'],d['H2D_bytes']/2**30,d['miss_mean'],d['M_mod_4_equals_2_fraction']])
    print(json.dumps(dict(status='PASS',arms=list(reports))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
