# Synthetic communication-price crossover

**Complete: TIMING_UNSTABLE.** C0 independently establishes
RESOURCE_PRICE_SHIFT with first lambda crossover 420.223139. All five frozen
rho schedules and twenty T0 actual/self timing cells passed correctness,
but C2's timing gate failed. No time-price gamma crossover is published.
Read [RESULTS.md](RESULTS.md) and [validation.json](validation.json).
Stop without extra repetitions or transport tuning.

Goal: directly test whether making remote communication progressively more
expensive moves the preferred replica budget toward larger rho.

Primary workload is the existing validated R4/B8/cache30 exact-routing trace.
This is a mechanism/sensitivity study, not a claim about a physical PCIe-only
server.

See PLAN.md and AGENT_TASK.md.
