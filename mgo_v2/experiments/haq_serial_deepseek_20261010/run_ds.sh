#!/bin/bash
# DeepSeek-V2-Lite serial-ablation decode policy comparison; starts after the Qwen matrix finishes.
Q=/home/hwlee/mgo-results/haq_serial_decode_20261010
R=/home/hwlee/mgo-results/haq_serial_deepseek_20261010
until grep -q ALLDONE $Q/matrix2.log; do sleep 20; done
while pgrep -f "run_headline_job.py --workloads" >/dev/null; do sleep 10; done
cd /home/hwlee/mgo-haq/mgo_v2
for C in 50 30; do
 L=ds_haq_serial_c${C}_v1
 [ -f $R/$L/result.json ] && continue
 echo "$(date +%T) start $L"
 MGO_HEADLINE_ROOT=$R MGO_DS_DECODE_POLICY_SCHEDULE=BW,BR,HAQ,LA_CA_NEAR,LA_CA_NEAR,HAQ,BR,BW MGO_DS_H2D_SERIAL_ABLATION=1 MGO_DS_PREFILL_POLICY=BR \
 PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1 NUMBA_CACHE_DIR=/tmp/mgo-haq-deepseek-numba-v1 \
 timeout 5600 /home/hwlee/sub-moe/phase01/.venv/bin/python scripts/run_headline_job.py --workloads /home/hwlee/mgo-results/deepseek_cache_ablation_20261009/WORKLOADS.json --label $L --system main_OURS --worker headline_ours_deepseek_multipolicy_worker.py --cell DeepSeekV2Lite_ShareGPT_R4_C${C}_B16_L512_O64 --timeout 5400 > $R/$L.driver.log 2>&1
 echo "$(date +%T) done $L rc=$?"
done
echo DSDONE
