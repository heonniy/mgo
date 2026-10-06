# Execution and timing eligibility

The requested experiment executes all eight matrix rows and preserves their outcomes. Timing stability is a separate result: an unstable completed triplet must not be relabelled PASS, filtered, or followed by an unbounded repeat loop.

`MAIN_TABLE_OUTCOMES.json` fixes the final reporting attempt for every row: the already scheduled bounded confirmation when the initial valid triplet was unstable, otherwise the completed stable primary. It does not choose the fastest sample or attempt. Invalid earlier preparations remain archived but cannot satisfy execution completion.

`audit_headline_execution.py` reuses every raw measurement check in `summarize_headline_r4.py`, additionally requires successful supervisor completion and exact cell/system coverage, and verifies each declared prior unstable attempt. A pending job cannot close an obligation. A final unstable row requires the completed bounded confirmation and remains ineligible for stable headline claims.

The existing `HEADLINE_SELECTION.json` and stable-panel audit are unchanged: only PASS triplets can enter that selection. `execution_complete` does not imply `stable_headline_panel_complete`. Final completion also requires archived provenance, resource caveats, clean GPU handoff, committed reports and verified push; those are not certified by this script alone.

Validation against current real receipts: six execution outcomes complete, three timing-stable, two jobs pending. Removing the prior-attempt declaration from a final unstable row is rejected; a duplicate matrix row is rejected. These were CPU-only checks without a new model run.

Regenerate from the repository root:

```sh
python mgo_v2/scripts/audit_headline_execution.py --outcomes mgo_v2/experiments/main_table_global_workload_20261006/MAIN_TABLE_OUTCOMES.json --output mgo_v2/experiments/main_table_global_workload_20261006/EXECUTION_AUDIT.json
```
