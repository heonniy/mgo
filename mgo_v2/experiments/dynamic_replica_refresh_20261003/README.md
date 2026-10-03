# Dynamic replica refresh headroom

The previous cache/eviction/substitution sweep showed a new failure mode:
with more cache, replicas survive much longer and H2D/reloads collapse, but
peer traffic can rise because the historical duplicate budget stops admitting
new replicas once the cap is full.

This packet tests whether replacing **stale duplicate copies** can preserve the
cache-relief benefit while recovering rank locality.

No new model capture is needed. Use the validated B8/B32 policy-feature traces
from `cache_eviction_substitution_20261003`.

See PLAN.md and AGENT_TASK.md.
