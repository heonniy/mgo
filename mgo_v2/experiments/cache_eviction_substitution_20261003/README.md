# Cache size × eviction × substitution

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
