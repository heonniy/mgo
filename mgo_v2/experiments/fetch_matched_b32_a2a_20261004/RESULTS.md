# NO_FETCH_MATCH — bounded decode64 screen complete

Reused only the first64 decode steps and original prefill of the existing
2048-request ShareGPT exact256 pool. No new model trace was captured.

The two routing-only objectives prescreened 65,536 sample/DP pairs. Their
independent top64 lists retained121 R4 candidates and128 R8 candidates after
deduplication. Exact Gate replay evaluated CA once and eight BR seeds per
candidate: 2,241 policy replays and1,992 BR/CA pairs total.

**Zero pairs satisfy exact total-fetch and H2D equality.** In every retained
pair, CA fetches more experts than BR. The bounded search cannot establish
that no matching workload exists outside the retained set.

| R | Exact BR/CA pairs | Eligible | BR−CA fetch difference range | Closest absolute difference |
|---|---:|---:|---:|---:|
| 4 | 968 | 0 | −672 to −129 | 129 fetches =1161 MiB |
| 8 | 1024 | 0 | −667 to −236 | 236 fetches =2124 MiB |

Closest R4: sample251 / DP42 / BR181, BR220706 versus CA220835 fetches.
Closest R8: sample484 / DP13 / BR73, BR244965 versus CA245201 fetches.
These are ineligible diagnostics, not substituted winners. Exact integer
matching was not relaxed, rounded, padded or retuned. First-fetch/reload
counts and all exact resource metrics remain in the candidate receipts.

The instrumented replay reproduced every array from the unchanged reference
policy on the full selected64-prefix compatibility input for BR and CA.
CPU_validation.json records this gate. CPU_A_resources.json and
CPU_B_resources.json record adaptive concurrency and memory guards; summed
RSS double-counts shared memory maps and is not physical-memory consumption.

No physical PLAN, A2 correctness pass or A3/A2 timing was launched because
there is no eligible common workload. No expanded search or automatic
followup is authorized. The old R4 experiment and queued R4_0123 were cancelled
by the owner; the interrupted sample is excluded. Completed old R8 results
remain intact. All eight resident model workers were verified after the run.
