# Static layer-split large cell

PASS: all3 repeats satisfy correctness/configuration/raw-clock and <=5% TTFT/TPOT/E2E spread gates. No additional repeats.

- TTFT: median 334.628984114s; range [331.908713410, 343.538012430]s; spread 3.454%.
- TPOT: median 0.404266444s; range [0.403793921, 0.415452857]s; spread 2.859%.
- E2E: median 360.068001114s; range [358.082243410, 369.006798430]s; spread 3.015%.

BF16 weights, native FP16 KV. Static14 GPU expert layers;34 CPU expert layers with host-op offload disabled. GPU KV allocation verified at smoke shape. 1Hz NVML peak and end-of-repeat RSS are in audit.json; these are not a continuous allocation trace.
