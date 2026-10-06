# B5 Old CA addition

Same C30/local B128/R4 GPUs 0,1,4,5/frozen decode64/BF16/H1b/V3/P2/T2/post-expert barrier condition. No new trace.

Canonical OLD_CA tokens, physical copies/bytes, controller and wire traffic passed; 3072 barriers per rank. No scientific retry.

| Policy | TPOT (s) | E2E (s) | Repeats | TPOT range (s) |
|---|---:|---:|---:|---|
| BR | 1.481771 | 105.277789 | 2 | [1.47828271484375, 1.485260009765625] |
| FCA | 1.542861 | 110.786828 | 3 | [1.5368450927734374, 1.581003173828125] |
| OLD_CA | 1.516545 | 107.945498 | 2 | [1.5129246826171876, 1.52016455078125] |

Old CA timing unstable: False. All valid repeats retained; two-repeat estimate is mean/median, three-repeat estimate is median.

Old CA was added after the BR/FCA packet, rather than interleaved with it. Differences are descriptive comparisons and can include session drift. BR/FCA were not rerun. Old CA extension adds clean physical timing only; BR/FCA have the separate first8 barrier diagnostics.
