"""Physical S1/S2 strict-phase TTFT headroom runner.

R4 defaults to physical GPUs 0,1,4,5. R8 NEVER defaults to 0..7: it requires
MGO_STRICT_R8_GPUS and MGO_STRICT_R8_AUTHORIZED=1 on an authorized 8-GPU host.
"""
import os,time,subprocess,signal,json
from pathlib import Path
from strict_headroom_common import *
import run_timing_stability as h
from strict_headroom_guard import guard,cooldown
from run_strict_headroom_pipeline import publish

def gpu_list(world):
    if world==4:return R4_GPUS
    if world==8:
        if os.environ.get('MGO_STRICT_R8_AUTHORIZED')!='1':
            raise RuntimeError('R8 physical timing requires explicit authorized allocation')
        raw=os.environ.get('MGO_STRICT_R8_GPUS')
        if not raw:raise RuntimeError('set MGO_STRICT_R8_GPUS to eight authorized physical GPUs')
        g=[int(x) for x in raw.split(',')];assert len(g)==8 and len(set(g))==8;return g
    raise ValueError(world)

def input_path(row):
    return ROOT/'inputs'/f'R{row["world"]}_B{row["batch"]}_L{row["context"]}_s{row["sample_seed"]}_d{row["dp_seed"]}_r{row["br_seed"]}'

def run_pair(label,row,candidate,stage):
    world=int(row['world']);gpus=gpu_list(world);out=ROOT/'physical'/label
    out.mkdir(parents=True,exist_ok=False)
    spec=dict(key=label,cache=30,batch=int(row['batch']),context=int(row['context']),policy=candidate,
        workload_seed=int(row['sample_seed']),dp_seed=int(row['dp_seed']),placement_seed=int(row['br_seed']),
        order=(int(row['br_seed'])+int(row['dp_seed']))&1,inputs=str(input_path(row)))
    write(out/'specs.json',[spec])
    env=h.env_for('env1');env.update(
        CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),MGO_V2_PHYSICAL_GPUS=','.join(map(str,gpus)))
    cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
        str(P/'examples/ttft_worker.py'),'--specs',str(out/'specs.json'),'--output',str(out),'--stage',stage]
    state=dict(status='RUNNING',label=label,world=world,gpus=gpus,spec=spec,started_unix=time.time(),command=cmd)
    state['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip()
    state['source_hashes']={str(p.relative_to(P)):sha(p) for p in list((P/'mgo_v2').glob('*.py'))+[P/'examples/ttft_worker.py',P/'examples/la_physical_worker.py',P/'scripts/env_offload_policy.py',P/'scripts/la_placement.py']}
    state['initial']=cooldown(gpus)
    proc=None;released=set();active=None;last_guard=0
    with (out/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=proc.pid;write(out/'status.json',state)
        try:
            while proc.poll() is None:
                if time.time()-state['started_unix']>4*3600:raise TimeoutError('bounded strict TTFT pair')
                if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
                if active and all((out/f'{active}_measure_rank{r}.json').exists() for r in range(world)):
                    active=None
                if active is None and (out/'boundary.json').exists():
                    boundary=json.loads((out/'boundary.json').read_text());key=boundary['key']
                    if key not in released and all((out/f'{key}_ready_rank{r}.json').exists() for r in range(world)):
                        state.setdefault('boundaries',[]).append(dict(key=key,resources=guard(gpus,proc.pid)))
                        write(out/'status.json',state)
                        (out/f'{key}_GO').touch();released.add(key);active=key
                if active is None and time.monotonic()-last_guard>30:
                    guard(gpus,proc.pid);last_guard=time.monotonic()
                time.sleep(1)
            assert proc.returncode==0,str(out/'run.log')
            result=json.loads((out/'result.json').read_text());assert result['status']=='PASS'
            assert all(sha(P/p)==digest for p,digest in state['source_hashes'].items()), 'runtime changed during measurement'
            state.update(status='PASS',result=result)
        except BaseException as exc:
            state.update(status='FAIL',error=repr(exc))
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
        finally:
            state['finished_unix']=time.time();write(out/'status.json',state)
            receipt=PACKET/(label+'.json');write(receipt,state)
            publish('strict TTFT: '+label+' '+state['status'],[receipt])
    return state

def main():
    s0=json.loads((PACKET/'STRICT_HEADROOM_S0.json').read_text());assert s0['status']=='PASS'
    rows=s0['rows'];summary=[];selected=[]
    cells=sorted({(r['world'],r['batch'],r['context']) for r in rows})
    for world,batch,context in cells:
        cell=[r for r in rows if (r['world'],r['batch'],r['context'])==(world,batch,context)]
        try:gpus=gpu_list(world)
        except RuntimeError as exc:
            summary.append(dict(world=world,batch=batch,context=context,status='PHYSICAL_PENDING',reason=str(exc)));continue
        physical=[]
        for i,row in enumerate(cell):
            label=f'S1_R{world}_B{batch}_L{context}_{i}'
            result=run_pair(label,row,'LA_CA_NEAR','S1')
            assert result['result']['results'], 'cross-policy token mismatch; preserve invalid pair and stop'
            rr=result['result']['results'][0];physical.append(dict(row=row,gain=rr['gain'],result=rr))
        winner=max(physical,key=lambda x:x['gain']);selected.append(winner['row'])
        summary.append(dict(world=world,batch=batch,context=context,status='S1_PASS',
            best_gain=winner['gain'],best=winner['row']))
    write(PACKET/'STRICT_HEADROOM_S1.json',dict(status='PASS',summary=summary,selected=selected))
    finals=[]
    for row in selected:
        world,batch,context=row['world'],row['batch'],row['context']
        c=run_pair(f'S2_R{world}_B{batch}_L{context}_LACA',row,'LA_CA_NEAR','S2')
        la=run_pair(f'S2_R{world}_B{batch}_L{context}_LA',row,'LA','S2')
        finals.append(dict(row=row,LA_CA_NEAR=c['result'],LA=la['result']))
    write(PACKET/'STRICT_HEADROOM_RESULTS.json',dict(status='PASS',purpose='maximum BR-adversarial headroom',
        summary=summary,finals=finals))

if __name__=='__main__':main()
