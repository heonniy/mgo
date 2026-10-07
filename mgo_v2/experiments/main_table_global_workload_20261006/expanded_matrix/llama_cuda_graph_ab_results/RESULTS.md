# CUDA Graph ON/OFF: R4 C30 B32 L512

CPU32/32 with identical fixed affinity; output64;2 primaries per arm, warmups excluded. Same executable, generated tokens identical. All four primaries retained. ON then OFF; not interleaved.

|Mode / repeat|TTFT (s)|TPOT (s)|E2E (s)|
|---|---|---|---|
|ON / 1|281.759467|0.735551|328.099175|
|ON / 2|281.442107|0.738561|327.971447|
|OFF / 1|281.893356|0.741091|328.582091|
|OFF / 2|281.568504|0.741742|328.298280|

|Metric|ON mean ± sample SD|OFF mean ± sample SD|OFF−ON|OFF slowdown|
|---|---|---|---|---|
|TTFT|281.600787 ± 0.224407|281.730930 ± 0.229705|0.130143|0.046%|
|TPOT|0.737056 ± 0.002128|0.741417 ± 0.000461|0.004361|0.592%|
|E2E|328.035311 ± 0.090318|328.440185 ± 0.200684|0.404875|0.123%|

CUDA Graph OFF changes TPOT by4.36ms (+0.592%) and E2E by0.405s (+0.123%). This bounded comparison does not explain the much larger historical timing difference. Earlier main-table values used different CPU thread/affinity settings; do not attribute their full difference to CUDA Graphs. Two sequential repeats per arm do not establish a universal sub-percent effect.
Placement, KV placement and63-interval timing checks passed in both arms. No extra experiment queued.
