# AGENT TASK

Checkpoint: 34c870c.

1. Verify the existing five-point B8 frontier.
2. Run/commit C0 exact resource-price sweep.
3. Freeze exact 384-event schedules for all five rho values.
4. On GPU 0/1/4/5 only, run T0 actual-vs-self communication replay in both
   policy orders; restore our workers on every exit.
5. Apply the predeclared timing gate without extra repeats.
6. Reuse existing H2D calibration to build the per-event max-rank fetch model.
7. If valid, sweep gamma and solve the lower-envelope crossovers.
8. Treat R3 as historical unstable context only.
9. Commit compact results and stop.

No model, new H2D measurement, R3 rerun, R8, batch/cache sweep, substitution,
new policy, or physical F/K/C timing.
