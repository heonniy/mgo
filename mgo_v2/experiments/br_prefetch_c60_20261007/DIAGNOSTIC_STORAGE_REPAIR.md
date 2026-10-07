# B8 diagnostic storage repair

The initial worker saved detailed diagnostic_rankN.json and then overwrote it
with the generic phase timing receipt using the same name. Primary timings
and their rank receipts were unaffected. Both B8 OFF/ON primaries completed.
The assistant then used the supervisor STOP mechanism to stop the defective
ON diagnostic pass; the generic supervisor labels this owner STOP, but it was
an assistant repair, not a new owner cancellation.

Detailed diagnostics now use diagnostic_rankN.json and timing receipts use
diagnostic_summary_rankN.json. A temporary-directory write/read regression
check confirms primary, diagnostic details and timing receipts coexist.
B8 v2 reuses the saved v1 primary receipts and runs only warmup+diagnostic for
each arm. Source parity is checked against v1 tokens/cache/bytes. B16/B64 use
the corrected writer from their first run. RUN_LABELS.json selects v2 for B8.
No primary samples are discarded or rerun. Final source archives identify the
primary-reference path. Existing v1 artifacts remain available for audit.
