# AGENT TASK — hot-expert replication threshold

**Complete: CURRENT_BATCH_TOO_COLD under the pooled-median H4 rule.**
H0/H1/H2/H4 passed; H3 skipped because all measured concurrent-H2D thresholds
exceed observed demand. R3 has substantial pass/order dependence, so no robust
hardware threshold is established. See RESULTS.md and validation.json.
No B64/B128 capture or extra timing follows. Keep only our restored GPU
0/1/4/5 model workers; our GPU 2/3/6/7 workers remain stopped.
The queued instructions below are retained as history.

This packet is QUEUED. Do not preempt the active
`future_rank_affinity_placement_20261003` study.

After that packet commits its result and stops:

1. run H0 CPU-only on existing B8/B16/B32 exact traces;
2. verify current-event exact marginal peer savings and hotness distributions;
3. run the bounded H1 pair-communication + 9-MiB H2D calibration on
   GPUs 0,1,4,5 under T0-IPC and R3-SHM;
4. derive measured median/p90 break-even n* without extrapolation;
5. map those thresholds back to existing traces (H2);
6. only if candidates exist, run the threshold-derived H3 CPU replay;
7. evaluate the predeclared H4 gate for whether a later B64 capture is worth
   proposing;
8. commit compact results and stop.

Do not:
- capture B64/B128 automatically;
- call R3 a PCIe-only server;
- tune thresholds from trace outcomes;
- add future prediction to H3;
- run R8;
- enable substitution;
- run model F/K/C timing;
- change cache ratio or NCCL protocol/channel settings.
