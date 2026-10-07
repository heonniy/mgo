# BR C60 prefetch physical results

R4, input256,256 decode steps; identical MAIN3678 +8 reserved slots, overlap, ready-first and T2 in OFF/ON. OFF leaves prefetch slots unused. One clean primary and one separate full diagnostic per arm. No attention included in diagnostic MoE metric.

|Batch|Prefetch|Clean TPOT s|Diagnostic MoE s/token|Decode peer GiB|Decode H2D GiB|
|---|---|---:|---:|---:|---:|
|8|off|0.643995|0.688560|8.181|2025.562|
|8|on|0.707467|0.776055|8.175|2259.281|

MoE metric = mean over steps of max-rank sum of48 MLP event durations. Includes router, host gaps, collectives and peer waiting; not pure GPU kernel time. Diagnostic overhead means it must not be subtracted from clean TPOT. Peer payload counts dispatch+return remote sends once, excluding metadata and NCCL protocol overhead.

|Batch|ON TPOT reduction %|ON peer volume reduction %|OFF/ON token agreement %|
|---|---:|---:|---:|
|8|-9.86|0.07|26.40|

Per-rank phases, CPU execution time, expert/token workload and collective CPU-entry skew are in each batch SUMMARY.json. Collective entry timestamps share the host monotonic clock; they are not NCCL GPU start timestamps. H2D service overlaps compute and cannot be added to primary TPOT. Prefetch diagnostics match their arm primary tokens/cache/bytes. One primary is descriptive, not a stability confirmation. Only BR is measured: another admission policy cannot be declared a physical winner from this packet.
