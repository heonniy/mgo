"""Reconcile diagnostic phase timelines without adding overlapping DMA service."""
import json,statistics as st,hashlib,shutil,csv
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACKET=Path(__file__).resolve().parents[1]/'experiments/main_table_global_workload_20261006/prefill_diagnosis'
GROUPS={
 'router_gate_compute':'Router/Gate',
 'metadata_counts':'Metadata counts','metadata_selected_ids':'Metadata selected IDs','metadata_routing_weights':'Metadata weights','metadata_probability_history':'Metadata Gate probability tail',
 'device_to_host_counts':'D2H/materialization','device_to_host_controller_materialization':'D2H/materialization',
 'placement_controller_cpu':'Placement CPU','layout_cpu':'Layout CPU','layout_device_materialization':'Layout GPU tensor construction',
 'moe.demand_h2d':'H2D enqueue','required_h2d_exposed_wait':'Exposed H2D wait','global_barrier_wait':'Global barrier',
 'moe.forward_a2a':'Forward pack/A2A/completion','forward_token_a2a_submit':'Forward pack/A2A/completion','moe.forward_complete':'Forward pack/A2A/completion',
 'moe.expert_compute':'Expert execution',
 'return_token_a2a':'Return A2A/combine','moe.return_a2a':'Return A2A/combine',
 'attention_dense_residual':'Attention/dense/non-MoE','clock_boundary_residual':'Boundary residual'}

def read(p):return json.loads(p.read_text())
def main():
 results=[];layer_rows=[]
 for cell,job,previous in [('B64/L512','ours_B64_L512_prefill_diagnosis1','ours_B64_L512_prefill_optimized1'),('B16/L256','ours_B16_L256_prefill_diagnosis1','ours_B16_L256_prefill_optimized1')]:
  raw=ROOT/job;status=read(raw/'status.json');assert status['status']=='PASS'
  ranks=[];archive={};dst=PACKET/job;dst.mkdir(exist_ok=True)
  for f in list(raw.glob('*.json'))+[raw/'run.log']:
   shutil.copy2(f,dst/f.name);h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==hashlib.sha256((dst/f.name).read_bytes()).hexdigest();archive[f.name]=h
  for rank in range(4):
   d=read(raw/f'diagnostic_rank{rank}.json');receipt=read(raw/f'repeat1_rank{rank}.json');reference=read(ROOT/previous/f'repeat1_rank{rank}.json')
   assert d['status']=='PASS' and receipt['no_compile'] and receipt['expert_cache_start']=='empty' and receipt['validation']['status']=='PASS'
   assert receipt['request_ids']==reference['request_ids'] and receipt['tokens']==[[r[0]] for r in reference['tokens']]
   groups=defaultdict(float);cpu=defaultdict(float);wall=defaultdict(float);layers=defaultdict(lambda:defaultdict(float))
   for label,seconds in d['exclusive_stream_partition'].items():groups[GROUPS.get(label,'Other MoE/host gaps')]+=seconds
   for seg in d['segments']:
    label=GROUPS.get(seg['phase'],'Other MoE/host gaps');cpu[label]+=seg['cpu_ns']/1e9;wall[label]+=seg['wall_ns']/1e9;layers[seg['layer']][label]+=seg['stream_seconds']
   assert abs(sum(groups.values())-d['wall_seconds'])<1e-6
   assert set(layers)==set(range(48))
   for layer,values in layers.items():
    for label,seconds in values.items():layer_rows.append(dict(cell=cell,rank=rank,layer=layer,component=label,stream_seconds=seconds))
   ranks.append(dict(rank=rank,TTFT=d['wall_seconds'],partition=dict(groups),own_thread_cpu=dict(cpu),host_wall=dict(wall),h2d_service=d['h2d_service_seconds'],h2d_bytes=sum(c['bytes'] for c in d['h2d_copies']),h2d_copies=len(d['h2d_copies'])))
  critical=max(ranks,key=lambda r:r['TTFT']);labels=sorted(set().union(*(r['partition'] for r in ranks)))
  mean={k:st.mean(r['partition'].get(k,0) for r in ranks) for k in labels};maxTTFT=max(r['TTFT'] for r in ranks)
  globalrow=read(raw/'repeat1.json');assert abs(globalrow['TTFT']-maxTTFT)<1e-9
  results.append(dict(cell=cell,job=job,source_commit=status['source_commit'],global_TTFT=maxTTFT,mean_rank_TTFT=st.mean(r['TTFT'] for r in ranks),max_completion_skew=maxTTFT-min(r['TTFT'] for r in ranks),critical_rank=critical['rank'],rank_mean_partition=mean,critical_rank_partition=critical['partition'],ranks=ranks,archive_sha256=archive,first_token_parity='PASS vs prior optimized primary, all requests',headline_eligible=False))
 (PACKET/'SUMMARY.json').write_text(json.dumps(results,indent=2)+'\n')
 with (PACKET/'LAYER_COMPONENTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(layer_rows[0]));w.writeheader();w.writerows(layer_rows)
 with (PACKET/'LAYER_SUMMARY.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['cell','layer','component','rank_mean_seconds','global_critical_rank_seconds','max_rank_service_seconds_nonadditive']);w.writeheader()
  for result in results:
   for layer in range(48):
    subset=[r for r in layer_rows if r['cell']==result['cell'] and r['layer']==layer]
    for label in sorted({r['component'] for r in subset}):
     values={r['rank']:r['stream_seconds'] for r in subset if r['component']==label}
     w.writerow(dict(cell=result['cell'],layer=layer,component=label,rank_mean_seconds=sum(values.values())/4,global_critical_rank_seconds=values.get(result['critical_rank'],0),max_rank_service_seconds_nonadditive=max(values.values())))
 small=next(r for r in results if r['cell'].startswith('B16'));large=results[0]
 order=['Router/Gate','Metadata counts','Metadata selected IDs','Metadata weights','Metadata Gate probability tail','D2H/materialization','Placement CPU','Layout CPU','Layout GPU tensor construction','H2D enqueue','Exposed H2D wait','Global barrier','Forward pack/A2A/completion','Expert execution','Return A2A/combine','Attention/dense/non-MoE','Other MoE/host gaps','Boundary residual']
 lines=['# Remaining prefill TTFT diagnosis','','Diagnostic only: one cold one-token run per cell after one-token warmup, Near/H0/full-pinned optimized prefill, R4/C30 on0/1/4/5. No default runtime change or primary-result replacement. All tokens match the prior optimized primary first token for the same requests. Cache, finite logits and no-compilation guards pass.','','## Additive current-stream completion partition','','Seconds. Critical rank is the rank with the latest first token in that cell. Each critical-rank column reconciles to global TTFT. Rank-mean columns sum to mean local TTFT; remaining completion skew is separately reported. Do not sum maxima chosen independently from each layer.','','| Component | B16 rank mean | B16 critical | B64 rank mean | B64 critical |','|---|---:|---:|---:|---:|']
 for label in order:
  v=[x[k].get(label,0) for x in (small,large) for k in ('rank_mean_partition','critical_rank_partition')];lines.append('| '+label+' | '+' | '.join(f'{n:.6f}' for n in v)+' |')
 lines+=['| **Total** | '+ ' | '.join(f'{v:.6f}' for v in [small['mean_rank_TTFT'],small['global_TTFT'],large['mean_rank_TTFT'],large['global_TTFT']])+' |','','## Host CPU corroboration','','Seconds, critical rank. CPU time is a separate view and must not be added to the above partition.','','| Component | B16 own-thread CPU | B64 own-thread CPU |','|---|---:|---:|']
 for label in ['Placement CPU','Layout CPU','Layout GPU tensor construction','Expert execution']:
  v=[x['ranks'][x['critical_rank']]['own_thread_cpu'].get(label,0) for x in (small,large)];lines.append('| '+label+' | '+' | '.join(f'{n:.6f}' for n in v)+' |')
 lines+=['','## H2D service: non-additive','','| Cell | Mean DMA service s/rank | Mean exposed wait s/rank | Global copies | Global GiB | Completion skew s |','|---|---:|---:|---:|---:|---:|']
 for r in (small,large):lines.append(f"| {r['cell']} | {st.mean(x['h2d_service'] for x in r['ranks']):.6f} | {r['rank_mean_partition'].get('Exposed H2D wait',0):.6f} | {sum(x['h2d_copies'] for x in r['ranks'])} | {sum(x['h2d_bytes'] for x in r['ranks'])/2**30:.6f} | {r['max_completion_skew']:.6f} |")
 delta={k:large['rank_mean_partition'].get(k,0)-small['rank_mean_partition'].get(k,0) for k in order}
 share=sum(large['critical_rank_partition'][k] for k in ['Layout CPU','Layout GPU tensor construction'])/large['global_TTFT']*100
 lines+=['','## Finding and next decision','',f"CPU layout construction plus device-index materialization account for {share:.2f}% of B64 critical-rank TTFT. Their large own-thread CPU times corroborate actual host work, not merely host descheduling or GPU wait. The largest B16-to-B64 increases in rank-mean seconds are: "+', '.join(f'{k} +{v:.3f}s' for k,v in sorted(delta.items(),key=lambda x:-x[1])[:4])+'.','','Prioritize the CPU token-index/layout representation path before grouped expert execution or H2D overlap. `plan_layout` performs token/expert sorting, per-expert scans and Python-list conversion on every layer. `pack_layouts` converts those lists back to NumPy, concatenates them and constructs device views. It was designed for outside-measurement immutable layouts but is called inside live prefill. Exact-mode `return_order` and per-expert `combine` are still built/packed even though rank-partial combine does not consume them. These are code-identified targets; sub-operation shares and repair gains have not been measured. No production repair is performed by this diagnostic.','','H0 expert service and communication are secondary to this measured layout bottleneck. Expert intervals include host launches and gather/weight operations, not pure GEMM kernel activity. Communication completion includes peer arrival wait; do not interpret it as standalone link bandwidth. The metadata D2H reduction is already enabled. Global barrier/H2D waits are visible but are not the dominant remaining term.','','## Limits and evidence','','One diagnostic per cell, without randomization/repeats. This localizes the remaining optimized OURS prefill cost; it does not profile DeepSpeed or prove an exact cross-system causal delta. Additional event instrumentation can perturb execution. Current-stream CUDA spans include CPU gaps, allocator work and dependent-stream waiting; host wall/CPU are independent explanatory views. H2D copy spans are separate DMA-stream service and must not be summed again. Dense residual includes attention, other dense/model work and token selection. All48 layers/four ranks, source commit, raw intervals and copy records are preserved. Full1Hz resource logs remain in raw directories. No new work on GPUs2/3/6/7.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':main()
