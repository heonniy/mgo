# Execution from dbe17f2

Owner fixes calibration GPUs to 0,1,4,5. The required sequence is Env1 and
Env2 model-free microbench -> PASS microbench_calibration.json -> calibrated
CPU replay -> GO_DECISION.json. No full-model inference timing follows.
Both CPU source/phase-order guards pass before launch.

Stop owned resident-model load during calibration; never terminate a foreign
process. Check selected-GPU exclusivity, host memory >=256 GiB and existing
GPU memory/temperature guard. Each environment has a 15-minute timeout and
process-group cleanup on failure. No resource polling inside microbench timing.
Checkpoint/push calibration before CPU replay. Restore resident-model workers
after calibration (or after failure). CPU replay keeps the existing memory
reserve monitor and hides CUDA. Publication failure retains a local checkpoint
instead of discarding scientific results.

The four-GPU pair calibration supplies measured transfer costs to an R8 CPU
model. This does not establish physical R8/B128/B256 TPOT or E2E.
