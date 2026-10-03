# Cache size × eviction × substitution

**Complete: 264 CPU cells and two diagnostic feature captures, all validated.**
Read [RESULTS.md](RESULTS.md), [validation.json](validation.json), and
[EXECUTION.md](EXECUTION.md) for outcomes, evidence, and frozen accounting rules.
Four predeclared labels have witnesses; no quality or timing claim is made.
The plan below is historical. No automatic follow-up is authorized.

This packet tests whether the earlier negative replication results were caused
by the specific historical regime:

```
cache30 + LRU + exact-only
```

rather than by replication itself.

Primary dimensions:
- cache ratio: 30/40/50/60%;
- eviction: LRU / gate / coverage;
- substitution: OFF / ON;
- replica budget rho: 0/.125/.25/.5/.75.

Primary workload is B8; B32 is a fixed secondary scaling check.

See PLAN.md and AGENT_TASK.md.
