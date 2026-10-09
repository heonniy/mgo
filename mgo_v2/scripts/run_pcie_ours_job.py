"""Bounded portable job supervisor; owns only its worker process group."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from pcie_host import ROOT, GPUS, available_bytes, topology, write
from pcie_gpu_occupancy import gpu_experiment

PKG=Path(__file__).resolve().parents[1]
REPO=PKG.parent
DATA=Path('/data2/esjung')
ARMS={
    'R-NEAR':('LA_CA_NEAR','rank_order'),
    'G-NEAR':('LA_CA_NEAR','group_balanced'),
    'G-BR':('BR','group_balanced'),
    'G-CA':('CA','group_balanced'),
    'G-NUMA-CA':('CA','group_balanced'),
    'R-BR':('BR','rank_order'),
    'R-CA':('CA','rank_order'),
}


def run(a):
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    source=PKG/'experiments/pcie_topology_ablation_20261009'
    search_stage=getattr(a,'search_stage',None);search_spec=getattr(a,'search_spec',None)
    if search_stage:
        assert search_spec and not a.sequence and not a.smoke and not a.overlap and not a.diagnostic
        frozen=json.loads(search_spec.read_text());assert frozen['status']=='FROZEN'
        write(out/'SEARCH_SPEC.json',frozen)
    identity=dict(arm=a.arm,sequence=a.sequence,search_stage=search_stage,
                  search_spec=str(search_spec) if search_spec else None,
                  search_spec_sha256=hashlib.sha256(search_spec.read_bytes()).hexdigest() if search_spec else None,
                  smoke=a.smoke,overlap=a.overlap,diagnostic=a.diagnostic,
                  breakdown_reference=str(a.breakdown_reference) if a.breakdown_reference else None,
                  grouped_probe=a.grouped_probe,
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  base_commit='7ddc55dff8d8a571a79a7fa415ad16fbc47e78dc',seed=42,
                  physical_gpus=list(GPUS),groups=[[0,1],[4,5]],conda_prefix=sys.prefix,
                  cache_slots=1843,main_slots=[459,459,459,458],reserved_slots_per_rank=2,
                  source='numa_shared_full_pinned',source_unique_bytes=108*2**30,
                  controller='native C++',phase_order='native overlap diagnostic' if a.overlap else 'metadata PLAN -> forward -> CPU rendezvous -> H2D -> CPU rendezvous -> compute -> CPU rendezvous -> return -> CPU rendezvous')
    write(out/'config.json',identity);write(out/'topology_before.json',topology())
    if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP file exists')
    available=available_bytes()
    if available<(108+128)*2**30:raise RuntimeError('108-GiB shared source plus 128-GiB host headroom guard failed')
    if shutil.disk_usage('/dev/shm').free<108*2**30:raise RuntimeError('Shared tmpfs source capacity guard failed')
    write(out/'host_guard.json',dict(status='PASS',available_before=available,source_bytes=108*2**30,reserve_bytes=128*2**30))
    pool=Path('/dev/shm')/f'esjung_pcie_ours_{os.getpid()}'
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    policy,quota=ARMS[a.arm]
    command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4',f'--master_port={port}',
             str(PKG/'examples/headline_ours_worker.py'),'--cell','R4_C30_B16_L512_O64',
             '--output',str(out),'--policy',policy,'--expert-executor','native','--native-prefill',
             '--prefetch-off','--prefill-optimized','--prefill-layout-fast','--decode-layout-fast',
             '--pcie-native-controller','--pcie-quota-mode',quota,'--numa-shared-source-root',str(pool),
             '--repeats',str(a.repeats)]
    if not a.overlap:command+=['--pcie-g2g-first-serial']
    if a.smoke:command+=['--smoke']
    if a.diagnostic:command+=['--pcie-phase-diagnostic']
    if a.capture_decisions:
        assert a.diagnostic and not a.sequence
        command+=['--pcie-capture-decisions']
    if a.arm=='G-NUMA-CA':command+=['--pcie-peer-costs-path',str(source/'microbench_physical_cores/peer_costs.json')]
    if a.sequence:
        assert not a.smoke and not a.overlap and not a.diagnostic and a.arm=='R-NEAR'
        command[5]=str(PKG/'examples/pcie_sequence_worker.py')
        command+=['--sequence',a.sequence]
    if a.breakdown_reference:
        assert a.sequence=='stage1'
        command[5]=str(PKG/'examples/pcie_breakdown_worker.py')
        command+=['--reference-cohort',str(a.breakdown_reference)]
    if a.grouped_probe:
        assert a.breakdown_reference and a.sequence=='stage1'
        command[5]=str(PKG/'examples/pcie_grouped_probe_worker.py')
    if search_stage:
        command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4',f'--master_port={port}',
                 str(PKG/'examples/pcie_maxgain_worker.py'),'--output',str(out),
                 '--search-spec',str(search_spec),'--search-stage',search_stage,
                 '--numa-shared-source-root',str(pool)]
    site=DATA/'envs/mgo-pcie/lib/python3.11/site-packages/nvidia'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',
             MGO_V2_NUMA_STRICT='1',CUDA_HOME=str(DATA/'envs/cuda121'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='1',
             TORCH_EXTENSIONS_DIR=str(DATA/'cache/torch_extensions'),NUMBA_CACHE_DIR=str(DATA/'cache/numba'),
             TRITON_CACHE_DIR=str(DATA/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(DATA/'cache/torchinductor'),
             HF_HOME=str(DATA/'cache/huggingface'),PYTHONPATH=':'.join([str(PKG),str(PKG/'scripts'),str(PKG/'examples')]),
             CPATH=':'.join(str(p) for p in site.glob('*/include')),
             MGO_MODEL_PATH=str(DATA/'models/Qwen3-30B-A3B-Instruct-2507'),MGO_EXPERT_STORE=str(DATA/'models/Qwen3-expert-store'),
             MGO_HEADLINE_WORKLOADS=str(DATA/'datasets/frozen_pcie_topology_20261009/WORKLOADS.json'),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NCCL_DEBUG='INFO',
             NCCL_DEBUG_FILE=str(out/'nccl_rank%r_pid%p.log'))
    source_paths=list((PKG/'mgo_v2').glob('*.py'))+list((PKG/'scripts').glob('*pcie*'))+[PKG/'examples/headline_ours_worker.py',PKG/'examples/pcie_sequence_worker.py',PKG/'examples/pcie_breakdown_worker.py',PKG/'examples/pcie_grouped_probe_worker.py',PKG/'examples/pcie_maxgain_worker.py',PKG/'examples/env_offload_worker.py']
    write(out/'launch.json',dict(argv=command,source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.is_file()},environment={k:env[k] for k in env if k.startswith(('MGO','CUDA','TORCH','NCCL','OMP','MKL','CPATH','NUMBA'))}))
    process=None
    with gpu_experiment(f'{a.arm}: '+('smoke' if a.smoke else 'full frozen generation')):
        try:
            with (out/'worker.log').open('w') as log:
                process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.monotonic();deadline=started+a.timeout
                write(out/'status.json',dict(status='RUNNING',pid=process.pid))
                while process.poll() is None:
                    if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP file observed')
                    if time.monotonic()>deadline:raise TimeoutError(f'Job exceeded {a.timeout}s')
                    if (a.sequence or search_stage) and (out/'phase.json').exists():
                        phase=json.loads((out/'phase.json').read_text())
                        if 'generation_started_unix' in phase and time.time()-phase['generation_started_unix']>3600:
                            raise TimeoutError('Generation batch exceeded 3600s')
                    if available_bytes()<128*2**30:raise RuntimeError('128-GiB host headroom was lost')
                    time.sleep(1)
                if process.returncode:raise RuntimeError(f'Worker exited {process.returncode}; inspect worker.log')
                result=json.loads((out/'result.json').read_text());assert result['status']=='PASS'
                write(out/'status.json',dict(status='PASS',pid=process.pid,wall_seconds=time.monotonic()-started))
        except BaseException as exc:
            write(out/'failure.json',dict(status='FAIL',cause=repr(exc),timestamp=time.time()))
            raise
        finally:
            if process is not None:
                # torchrun and its four ranks belong to this freshly created
                # process group. Never select workers by a broad process name.
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            shutil.rmtree(pool,ignore_errors=True)
    write(out/'topology_after.json',topology())
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.npy')}
    write(out/'SHA256.json',artifacts)
    print(json.dumps(dict(status='PASS',arm=a.arm,out=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--overlap',action='store_true');p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--repeats',type=int,choices=range(1,6),default=1);p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--sequence',choices=('stage1','stage2','optimize'))
    p.add_argument('--capture-decisions',action='store_true')
    p.add_argument('--breakdown-reference',type=Path)
    p.add_argument('--grouped-probe',action='store_true')
    p.add_argument('--search-stage',choices=('nomination','screen','final'))
    p.add_argument('--search-spec',type=Path)
    run(p.parse_args())
