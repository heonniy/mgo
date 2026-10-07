# Ordinary graph reuse OFF: B32 L512 CPU32

R4/C30, global128,input512,output64, fixed32/32 CPU affinity. CUDA Graphs OFF in both arms. Same binary and exact corresponding generated-token parity PASS. Two clean primaries per arm, warmups excluded; older reference reused, not interleaved.

|Metric|Reuse ON mean|Both OFF mean ± sample SD|OFF−ON|Change|
|---|---|---|---|---|
|TTFT|281.730930|281.391786 ± 0.026199|-0.339144|-0.120%|
|TPOT|0.741417|0.744603 ± 0.000823|0.003186|0.430%|
|E2E|328.440185|328.301765 ± 0.025641|-0.138420|-0.042%|

|Both OFF repeat|TTFT|TPOT|E2E|
|---|---|---|---|
|1|281.373261|0.745185|328.319896|
|2|281.410312|0.744021|328.283634|

Seconds. Ordinary graph reuse accounts for only3.19ms of observed TPOT difference here (+0.430% when disabled). Small negative TTFT/E2E differences do not establish a speedup from disabling reuse. Combined with the previous CUDA Graph check, neither mechanism explains the much larger historical timing gap. Previous main-table thread/affinity settings differed. No more experiments queued.
