# AGENT TASK — BR / CA / CA-rep CPU headroom

Checkpoint: 0cd0fa0.

Read PLAN.md.

1. Freeze two workload manifests:
   - MATH test: 512, seed42, stratified, fixed 256 decode tokens;
   - ShareGPT V3 cleaned: 512 human->assistant turns, seed44, prompt 32--512
     tokens, reference assistant >=128 tokens, fixed 256 generated tokens.
   For both datasets, decode64 is the exact first-64-step prefix of the same
   decode256 capture; do not run a second 64-token generation.
2. Audit the current similarity artifact provenance.
   - Reuse only if it is the same Qwen checkpoint + FineWeb-Edu 400x128 +
     SERE Frobenius calibration.
   - Otherwise run that exact SERE-style calibration once.
   - Do not use MATH/ShareGPT or co-routed cosine for calibration.
3. In one 8-GPU model-loading session, sequentially capture one 512-request,
   256-decode-step master trace for MATH and one for ShareGPT.
4. From each master, derive decode64 (first 64 steps) and decode256 (full 256),
   commit the lightweight horizon audit required by PLAN.md, then derive every
   R={4,8}, B={8,16,32,64} workload offline.
5. Use global cache={30,40,50,60}%, eviction={LRU,Gate},
   substitution={OFF,ON}.
6. Implement/validate BR, CA, CA-rep exactly as PLAN.md defines.
7. Run 768 main CPU replays plus at most 16 total BR seed-audit replays,
   up to 16 single-thread cells in parallel under the memory guards.
8. Report exact/local/substitute/effective hits, residual miss, turnover,
   reload, H2D, peer, locality, BR->CA headroom and CA->CA-rep headroom.
9. Commit compact checkpoints/results and stop.

No Env timing, accuracy, NCCL, B128, Coverage eviction, workload-specific
calibration, replica ratio or replica protection.
