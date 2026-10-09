"""Portable bounded native-baseline jobs on physical GPUs 0,1,4,5 only."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import numpy as np
import psutil
from pcie_host import ROOT, GPUS, available_bytes, affinity, topology, write
from pcie_gpu_occupancy import gpu_experiment

PKG=Path(__file__).resolve().parents[1]
REPO=PKG.parent
DATA=Path('/data2/esjung')
WORKERS=dict(deepspeed='headline_deepspeed_worker.py',infinity='headline_infinity_worker.py',llama='headline_llama_sync_worker.py')


class DeviceMemorySamples:
    """NVML only: no CUDA contexts and no access to non-owner GPU indices."""
    class Memory(C.Structure):
        _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]
    def __init__(self):
        self.lib=C.CDLL('libnvidia-ml.so.1')
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[C.c_uint,C.POINTER(C.c_void_p)]
        self.lib.nvmlDeviceGetMemoryInfo.argtypes=[C.c_void_p,C.POINTER(self.Memory)]
        if self.lib.nvmlInit_v2():raise RuntimeError('NVML init failed')
        self.handles=[]
        for gpu in GPUS:
            handle=C.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(gpu,C.byref(handle)):raise RuntimeError(f'NVML GPU{gpu} handle failed')
            self.handles.append(handle)
    def sample(self):
        result={}
        for gpu,handle in zip(GPUS,self.handles):
            info=self.Memory()
            if self.lib.nvmlDeviceGetMemoryInfo(handle,C.byref(info)):raise RuntimeError(f'NVML GPU{gpu} sample failed')
            result[str(gpu)]=int(info.used)
        return result
    def close(self):self.lib.nvmlShutdown()


def run(a):
    assert tuple(GPUS)==(0,1,4,5)
    a.out.mkdir(parents=True,exist_ok=False)
    if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP exists')
    if available_bytes()<(61+128)*2**30:raise RuntimeError('61-GiB model plus 128-GiB available host headroom guard failed')
    write(a.out/'topology_before.json',topology())
    model=DATA/'models/Qwen3-30B-A3B-Instruct-2507'
    workloads=DATA/'datasets/frozen_pcie_topology_20261009/WORKLOADS.json'
    spec=json.loads(workloads.read_text())['cells'][0]
    assert spec['cell']=='R4_C30_B16_L512_O64' and spec['global_requests']==64
    for phase in ('target','warmup'):
        assert hashlib.sha256(Path(spec[phase]['path']).read_bytes()).hexdigest()==spec[phase]['sha256']
    site=DATA/'envs/mgo-pcie/lib/python3.11/site-packages/nvidia'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',MGO_PCIE_HOST='1',
             MGO_MODEL_PATH=str(model),MGO_HEADLINE_WORKLOADS=str(workloads),
             MGO_INFINITY_ROOT=str(DATA/'tools/MoE-Infinity'),MGO_INFINITY_STORE=str(DATA/'models/infinity-bf16-store'),
             MGO_BASELINE_TOOLS=str(DATA/'tools'),MGO_GGUF_PATH=str(DATA/'models/Qwen3-30B-A3B-Instruct-2507-BF16.gguf'),
             MGO_LLAMA_BUILD_RECEIPT=str(ROOT/'LLAMA_BUILD.json'),MGO_TOPOLOGY_PATH=str(a.out/'cpu_topology.json'),
             CUDA_HOME=str(DATA/'envs/cuda121'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='1',
             TORCH_EXTENSIONS_DIR=str(DATA/'cache/torch_extensions'),HF_HOME=str(DATA/'cache/huggingface'),
             PYTHONPATH=':'.join([str(PKG),str(PKG/'scripts'),str(PKG/'examples'),str(DATA/'tools/MoE-Infinity')]),
             CPATH=':'.join(str(p) for p in site.glob('*/include')),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(a.out/'nccl_rank%r_pid%p.log'))
    write(a.out/'cpu_topology.json',dict(fixed_affinity={str(gpu):affinity(rank) for rank,gpu in enumerate(GPUS)}))
    script=PKG/'examples'/WORKERS[a.system]
    command=[sys.executable,str(script),'--cell',spec['cell'],'--output',str(a.out),'--repeats',str(a.repeats)]
    if a.smoke:command+=['--smoke']
    if a.system=='llama':command+=['--threads','32','--cuda-graphs','off','--graph-reuse','off','--expert-placement','balanced3']
    if a.system=='deepspeed':
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4',f'--master_port={port}']+command[1:]
    write(a.out/'launch.json',dict(argv=command,physical_gpus=list(GPUS),model=str(model),workload=spec,
          git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),conda_prefix=sys.prefix,
          source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
          adaptation_source_sha256={str(PKG/'scripts'/'pcie_deepspeed_leaf.py'):hashlib.sha256((PKG/'scripts'/'pcie_deepspeed_leaf.py').read_bytes()).hexdigest()} if a.system=='deepspeed' else {},
          runtime='native baseline, no forced OURS serial phase order',
          primary_repeats=a.repeats,stability_policy='Five repeats predefined for baseline jobs; keep every repeat',
          environment={k:v for k,v in env.items() if k.startswith(('CUDA','MGO','NCCL','TORCH','OMP','MKL','CPATH'))}))
    process=None;resource=[];device_sampler=None
    with gpu_experiment(f'{a.system} native baseline '+('smoke' if a.smoke else 'frozen global64')):
        try:
            device_sampler=DeviceMemorySamples()
            with (a.out/'worker.log').open('w') as log:
                process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.monotonic();last_sample=0
                write(a.out/'status.json',dict(status='RUNNING',pid=process.pid))
                while process.poll() is None:
                    if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP observed')
                    if time.monotonic()-started>a.timeout:raise TimeoutError(f'Baseline exceeded {a.timeout}s')
                    free=available_bytes()
                    if free<128*2**30:raise RuntimeError('128-GiB available host headroom lost')
                    if time.monotonic()-last_sample>=1:
                        children=psutil.Process(process.pid).children(recursive=True);rss={}
                        for child in children:
                            try:rss[str(child.pid)]=child.memory_info().rss
                            except psutil.NoSuchProcess:pass
                        resource.append(dict(unix=time.time(),available_host_bytes=free,rss_by_pid=rss,
                                             gpu_used_bytes=device_sampler.sample()))
                        last_sample=time.monotonic()
                    time.sleep(.5)
                if process.returncode:raise RuntimeError(f'Worker exited {process.returncode}; inspect worker.log')
                result=json.loads((a.out/'result.json').read_text());assert result['status']=='PASS'
                if not a.smoke:
                    statistics={}
                    for metric in ('TTFT','TPOT','E2E','throughput'):
                        rows=[json.loads((a.out/f'repeat{r}.json').read_text()) for r in range(1,a.repeats+1)]
                        values=np.array([row[metric] if metric in row else row['global_requests']*row['output_tokens']/row['E2E'] for row in rows])
                        statistics[metric]=dict(median=float(np.median(values)),mean=float(values.mean()),sd=float(values.std(ddof=1)),min=float(values.min()),max=float(values.max()),values=values.tolist())
                    write(a.out/'statistics.json',statistics)
                write(a.out/'status.json',dict(status='PASS',wall_seconds=time.monotonic()-started))
        except BaseException as exc:
            failure=dict(status='FAIL',cause=repr(exc),timestamp=time.time(),pid=process.pid if process else None)
            write(a.out/'failure.json',failure);write(a.out/'status.json',failure)
            raise
        finally:
            if process is not None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            write(a.out/'host_resource_samples.json',dict(scope='One-second RSS samples; shared pages may appear in multiple RSS values; no inference of CUDA pinning from VmLck',samples=resource))
            if resource:
                write(a.out/'resource_peaks.json',dict(scope='One-second NVML device-used samples include CUDA context/driver allocations; sampled peaks may miss transients. Framework allocator peaks remain separate.',
                      physical_gpus=list(GPUS),gpu_used_sampled_peak_bytes={str(g):max(r['gpu_used_bytes'][str(g)] for r in resource) for g in GPUS},
                      host_rss_sum_sampled_peak_bytes=max(sum(r['rss_by_pid'].values()) for r in resource),host_available_min_bytes=min(r['available_host_bytes'] for r in resource)))
            if device_sampler is not None:device_sampler.close()
    write(a.out/'topology_after.json',topology())
    print(json.dumps(dict(status='PASS',system=a.system,out=str(a.out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--system',choices=WORKERS,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--repeats',type=int,choices=(3,5),default=5);p.add_argument('--timeout',type=int,default=14400)
    run(p.parse_args())
