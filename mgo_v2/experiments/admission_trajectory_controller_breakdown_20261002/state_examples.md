# Matched-demand cache-state examples

Selection: for each batch/source stream, the five largest positive and five largest negative instantaneous admission-count deltas after prefill; earliest event wins ties. Snapshots are captured on replay repetition 1 at the same event indexes for both policies. They retain every resident key/rank, execution owners, gate score and independently computed Coverage damage. Selection is descriptive, not policy tuning.

“Positive/negative” below refers to fewer/more Current fetches at that event, not an E2E performance win/loss. These state examples do not establish the total future value of an individual placement. Next-demand distance is in global layer events; absence before eviction is not proof of no later demand.

| B | Raw source | Event (step/layer) | Current − Random fetches | Focus (layer, expert) | Random state / future uses | Current state / future uses |
|---:|---|---|---:|---|---|---|
| 4 | random | 96 (2/0) | -3 | (0, 9) | absent | rank 0, gate 0.00750, damage 1; next 384, exact 6, anchor 0, evict None |
| 4 | random | 389 (8/5) | +3 | (1, 22) | rank 3, gate 0.01031, damage 127; next 44, exact 1, anchor 0, evict 46 | absent |
| 4 | random | 395 (8/11) | +3 | (1, 22) | rank 3, gate 0.01031, damage 127; next 38, exact 1, anchor 0, evict 40 | absent |
| 4 | random | 437 (9/5) | +3 | (1, 27) | rank 3, gate 0.01223, damage 127; next 44, exact 1, anchor 44, evict 45 | absent |
| 4 | random | 510 (10/30) | +3 | (0, 32) | rank 3, gate 0.00736, damage 1; next 18, exact 1, anchor 0, evict 66 | absent |
| 4 | random | 559 (11/31) | +3 | (0, 78) | rank 0, gate 0.00747, damage 1; next 65, exact 1, anchor 0, evict 209 | absent |
| 4 | random | 587 (12/11) | -3 | (0, 83) | absent | rank 3, gate 0.00718, damage 1; next 37, exact 5, anchor 0, evict 950 |
| 4 | random | 821 (17/5) | -3 | (3, 50) | absent | rank 3, gate 0.01189, damage 127; next 46, exact 1, anchor 0, evict 48 |
| 4 | random | 1589 (33/5) | -3 | (0, 100) | absent | rank 3, gate 0.00727, damage 1; next 43, exact 1, anchor 0, evict 53 |
| 4 | random | 2509 (52/13) | -3 | (3, 84) | absent | rank 3, gate 0.02281, damage 127; next 38, exact 2, anchor 0, evict 87 |
| 4 | hungarian_current | 96 (2/0) | -3 | (0, 9) | absent | rank 0, gate 0.00748, damage 1; next 48, exact 8, anchor 0, evict None |
| 4 | hungarian_current | 292 (6/4) | -3 | (0, 107) | absent | rank 3, gate 0.00645, damage 1; next 44, exact 18, anchor 0, evict 1183 |
| 4 | hungarian_current | 344 (7/8) | -3 | (0, 113) | absent | rank 3, gate 0.00682, damage 1; next 232, exact 1, anchor 0, evict 280 |
| 4 | hungarian_current | 349 (7/13) | +4 | (0, 22) | rank 2, gate 0.00802, damage 1; next 35, exact 18, anchor 0, evict None | absent |
| 4 | hungarian_current | 432 (9/0) | -3 | (0, 6) | absent | rank 3, gate 0.00697, damage 1; next 48, exact 7, anchor 0, evict 792 |
| 4 | hungarian_current | 470 (9/38) | -3 | (0, 6) | absent | rank 3, gate 0.00697, damage 1; next 10, exact 7, anchor 0, evict 754 |
| 4 | hungarian_current | 490 (10/10) | +4 | (1, 127) | rank 0, gate 0.02826, damage 127; next 39, exact 1, anchor 27, evict 40 | absent |
| 4 | hungarian_current | 536 (11/8) | +4 | (0, 78) | rank 1, gate 0.00765, damage 1; next 40, exact 2, anchor 0, evict 189 | absent |
| 4 | hungarian_current | 541 (11/13) | +4 | (0, 78) | rank 1, gate 0.00765, damage 1; next 35, exact 2, anchor 0, evict 184 | absent |
| 4 | hungarian_current | 544 (11/16) | +4 | (0, 78) | rank 1, gate 0.00765, damage 1; next 32, exact 2, anchor 0, evict 181 | absent |
| 8 | random | 100 (2/4) | -3 | (1, 87) | absent | rank 2, gate 0.02058, damage 127; next 45, exact 2, anchor 78, evict 94 |
| 8 | random | 108 (2/12) | -3 | (0, 6) | absent | rank 1, gate 0.00621, damage 1; next 180, exact 6, anchor 0, evict 670 |
| 8 | random | 122 (2/26) | -3 | (0, 6) | absent | rank 1, gate 0.00621, damage 1; next 166, exact 6, anchor 0, evict 656 |
| 8 | random | 220 (4/28) | +4 | (1, 119) | rank 1, gate 0.03284, damage 127; next 21, exact 6, anchor 0, evict 263 | absent |
| 8 | random | 318 (6/30) | -3 | (0, 63) | absent | rank 3, gate 0.00606, damage 1; next 18, exact 2, anchor 0, evict 138 |
| 8 | random | 720 (15/0) | +4 | (2, 45) | rank 0, gate 0.02785, damage 127; next 2, exact 2, anchor 0, evict 52 | absent |
| 8 | random | 816 (17/0) | +5 | (0, 20) | rank 2, gate 0.00768, damage 1; next 48, exact 1, anchor 0, evict 144 | absent |
| 8 | random | 871 (18/7) | +4 | (0, 40) | rank 0, gate 0.00781, damage 1; next 41, exact 4, anchor 0, evict 377 | absent |
| 8 | random | 897 (18/33) | -4 | (1, 119) | absent | rank 0, gate 0.01895, damage 127; next 16, exact 2, anchor 0, evict 66 |
| 8 | random | 1062 (22/6) | +4 | (2, 65) | rank 2, gate 0.01533, damage 127; next 44, exact 1, anchor 0, evict 46 | absent |
| 8 | hungarian_current | 92 (1/44) | -3 | (0, 40) | absent | rank 1, gate 0.00558, damage 1; next 4, exact 20, anchor 0, evict 1221 |
| 8 | hungarian_current | 122 (2/26) | -4 | (0, 6) | absent | rank 1, gate 0.00622, damage 1; next 118, exact 13, anchor 0, evict 1149 |
| 8 | hungarian_current | 129 (2/33) | +4 | (0, 28) | rank 3, gate 0.00678, damage 1; next 15, exact 1, anchor 0, evict 16 | absent |
| 8 | hungarian_current | 202 (4/10) | -4 | (0, 6) | absent | rank 1, gate 0.00607, damage 1; next 38, exact 13, anchor 0, evict 1069 |
| 8 | hungarian_current | 293 (6/5) | +3 | (0, 20) | rank 1, gate 0.00823, damage 1; next 43, exact 1, anchor 0, evict 139 | absent |
| 8 | hungarian_current | 384 (8/0) | -5 | (2, 95) | absent | rank 3, gate 0.03210, damage 127; next 2, exact 3, anchor 0, evict 99 |
| 8 | hungarian_current | 549 (11/21) | +3 | (0, 118) | rank 3, gate 0.00736, damage 1; next 27, exact 15, anchor 0, evict 1131 | absent |
| 8 | hungarian_current | 586 (12/10) | +4 | (0, 88) | rank 1, gate 0.00752, damage 1; next 38, exact 5, anchor 0, evict 240 | absent |
| 8 | hungarian_current | 2415 (50/15) | -4 | (5, 108) | absent | rank 2, gate 0.00768, damage 1; next 38, exact 1, anchor 0, evict 41 |
| 8 | hungarian_current | 2793 (58/9) | +4 | (2, 115) | rank 0, gate 0.01209, damage 127; next 41, exact 1, anchor 0, evict 42 | absent |
| 16 | random | 68 (1/20) | -4 | (0, 4) | absent | rank 2, gate 0.00632, damage 1; next 28, exact 42, anchor 0, evict 2199 |
| 16 | random | 83 (1/35) | -3 | (0, 4) | absent | rank 2, gate 0.00632, damage 1; next 13, exact 42, anchor 0, evict 2184 |
| 16 | random | 96 (2/0) | +3 | (1, 119) | rank 2, gate 0.02726, damage 127; next 1, exact 2, anchor 0, evict 50 | absent |
| 16 | random | 198 (4/6) | +4 | (1, 44) | rank 1, gate 0.02210, damage 127; next 43, exact 1, anchor 6, evict 46 | absent |
| 16 | random | 217 (4/25) | +4 | (1, 44) | rank 1, gate 0.02210, damage 127; next 24, exact 1, anchor 6, evict 27 | absent |
| 16 | random | 384 (8/0) | -4 | (2, 47) | absent | rank 2, gate 0.02440, damage 127; next 2, exact 1, anchor 0, evict 4 |
| 16 | random | 492 (10/12) | +4 | (4, 37) | rank 3, gate 0.00750, damage 1; next 136, exact 1, anchor 0, evict 348 | absent |
| 16 | random | 680 (14/8) | -4 | (0, 1) | absent | rank 2, gate 0.00756, damage 1; next 40, exact 1, anchor 0, evict 69 |
| 16 | random | 1360 (28/16) | +4 | (2, 45) | rank 3, gate 0.01952, damage 127; next 34, exact 3, anchor 0, evict 132 | absent |
| 16 | random | 1597 (33/13) | -4 | (2, 47) | absent | rank 3, gate 0.01733, damage 127; next 37, exact 1, anchor 0, evict 39 |
| 16 | hungarian_current | 68 (1/20) | -4 | (0, 4) | absent | rank 2, gate 0.00632, damage 1; next 28, exact 28, anchor 0, evict 1516 |
| 16 | hungarian_current | 83 (1/35) | -3 | (0, 4) | absent | rank 2, gate 0.00632, damage 1; next 13, exact 28, anchor 0, evict 1501 |
| 16 | hungarian_current | 94 (1/46) | -3 | (0, 4) | absent | rank 2, gate 0.00632, damage 1; next 2, exact 28, anchor 0, evict 1490 |
| 16 | hungarian_current | 96 (2/0) | +5 | (0, 48) | rank 3, gate 0.00980, damage 1; next 96, exact 50, anchor 0, evict None | absent |
| 16 | hungarian_current | 127 (2/31) | +4 | (0, 48) | rank 3, gate 0.00980, damage 1; next 65, exact 50, anchor 0, evict None | absent |
| 16 | hungarian_current | 297 (6/9) | +5 | (0, 69) | rank 3, gate 0.00722, damage 1; next 87, exact 1, anchor 0, evict 111 | absent |
| 16 | hungarian_current | 501 (10/21) | +4 | (1, 86) | rank 2, gate 0.02085, damage 127; next 28, exact 2, anchor 0, evict 77 | absent |
| 16 | hungarian_current | 772 (16/4) | -4 | (0, 45) | absent | rank 2, gate 0.00753, damage 1; next 44, exact 5, anchor 0, evict 242 |
| 16 | hungarian_current | 1473 (30/33) | +5 | (0, 40) | rank 1, gate 0.00772, damage 1; next 15, exact 2, anchor 0, evict 75 | absent |
| 16 | hungarian_current | 2982 (62/6) | -4 | (0, 125) | absent | rank 1, gate 0.00758, damage 1; next 42, exact 2, anchor 0, evict None |

Retained 30 fewer-fetch and 30 more-fetch examples. Full state and raw snapshot paths: [state_examples.json](state_examples.json).
