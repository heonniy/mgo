# Timing stability + NUMA diagnosis

The physical BR/CA/CA-rep E2E matrix is **paused**. Do not launch any new policy
comparison cell until this packet establishes a stable measurement harness.

Observed problem: identical P/CA-rep/Env1 repeats differed strongly in decode
time (about 605 s vs 466 s) while token/final-cache hashes and no-compile checks
matched. The current external PSS monitor takes roughly 17--25 s per scan and
runs concurrently with decode, so monitor interference is a primary confound
candidate. NUMA placement is a second candidate, especially for Env 2 SHM and
CPU->GPU H2D.

This packet isolates those causes with the smallest useful set of runs.
