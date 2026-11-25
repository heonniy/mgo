#!/bin/bash
# Post-job health check (no root needed). Usage: health_check.sh <job label>
# Catches what the job logs may not record: leftover GPU processes or pinned/locked host memory,
# shared-store damage, GPU ECC errors, and error lines anywhere in the job's run.log.
R=/home/hwlee/mgo-results/qwen235_main_table_20261125; L=$1; J=$R/jobs/$L
m() { awk -v k="$1:" '$1==k{printf "%.0f", $2/1048576}' /proc/meminfo; }
issues=()
procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$procs" -gt 0 ] && issues+=("gpu_procs_left=$procs")
used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '($1==0||$1==1||$1==4||$1==5)&&$2>1024{printf "%s:%sMiB ",$1,$2}')
[ -n "$used" ] && issues+=("gpu_mem_left[$used]")
ecc=$(nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | awk -F', ' '$2!="0"&&$2!="[N/A]"{printf "%s:%s ",$1,$2}')
[ -n "$ecc" ] && issues+=("ecc_uncorrected[$ecc]")
mlock=$(m Mlocked); [ "${mlock:-0}" -gt 1 ] && issues+=("mlocked_GiB=$mlock")
python3 - <<'P' || issues+=("shared_store_check_failed")
import json,hashlib,os,sys
d='/dev/shm/mgo_qwen235_expert_store_v1'
man=json.load(open(os.path.join(d,'manifest.json')))
f=[x for x in os.listdir(d) if x.endswith('.bf16')][0]
row=int(man['row_bytes']);bad=[]
with open(os.path.join(d,f),'rb') as fh:
    for r,h in man['sample_sha256'].items():
        fh.seek(int(r)*row);
        if hashlib.sha256(fh.read(row)).hexdigest()!=h: bad.append(r)
sys.exit(1 if bad else 0)
P
errs=$(grep -h -i -E "CUDA error|Xid|unregister failed|cudaHostRegister.*failed|illegal memory|NCCL error|out of memory|Segmentation|core dumped|Traceback" $J/run.log $R/jobs/$L.driver.log 2>/dev/null | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')
[ -n "$errs" ] && issues+=("log_errors[$errs]")
echo "HEALTH $L avail=$(m MemAvailable)GiB shmem=$(m Shmem)GiB mlocked=${mlock}GiB ${issues[*]:-OK}"
