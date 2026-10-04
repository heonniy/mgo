# Authorized GPU execution

Owner requested GPU execution after advance CPU preparation.
Wait for all three-policy plan proofs, then stop owned resident-model workers,
cool down and run Env2 SHM preflight. Run R4 on 0,1,4,6, then R8 on 0..7.

Use A3 only. Initial order is BR, CA, OldCA for R4; OldCA, CA, BR for R8.
One eight-step readiness prefix per R, followed by separate full64 counter and
correctness passes; no full warmup before each timing. Cache/KV/RNG reset for
each MEASURE, with compilation forbidden and no heavy monitoring inside timing.

If TPOT or decode wall improves at least 1%, add exactly one BR+policy pair.
Each qualifying policy gets its own BR confirmation measurement. Initial E2E
alone never triggers confirmation. Verify no reusable previous A3 confirmation:
R4 has a different BR seed; the same R8 candidate did not trigger confirmation.
Stop if this assumption fails rather than duplicate an existing confirmation.

Validate physical H2D, peer traffic, fan-out and per-event critical bytes against
CPU schedules before reporting results. Commit/push at each R boundary and on
failure. Restore all eight resident-model workers after completion or failure.
No subsequent experiment is automatically enabled.
