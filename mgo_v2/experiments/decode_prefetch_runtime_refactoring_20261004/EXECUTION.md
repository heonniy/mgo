# Autonomous execution amendment

The owner explicitly requests completing implementation and running overnight to
select the strongest measured LA gain, fixing bugs instead of stopping at plan
review checkpoints. Validation and per-milestone commits remain mandatory.
Invalid measurements are discarded, never treated as gains; preserve failure
receipts and source identities before repair/retry. GPU 0–7 remains authorized.
Shared-host memory reserves and owned-process-only cleanup remain in force.

Use the existing MATH LOAD request/routing/teacher inputs for B128 and B256.
B256 evaluates all MATH requests, so predictor calibration uses disjoint
ShareGPT requests, with dataset-qualified identities and token hashes checked
for leakage. Report this cross-dataset split explicitly.

Three-arm selection uses stable repeats, shared BR-only tuned P/trigger, and
min(B128 LA gain, B256 LA gain), plus the absolute-LA-time dominance guard.
No runtime is declared the winner before all three eligible arms are compared.
