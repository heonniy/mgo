# R8 LOAD BR vs LA physical validation

Frozen exact routes and teacher tokens; live placement; pinned H2D; fetch barrier; Env1.
E2E includes prefill + decode256. TPOT covers decode. Values are seconds.

- B128: E2E gain 6.46%; TPOT gain 6.48%; stable comparison=True.
- B256: E2E gain 1.20%; TPOT gain 1.04%; stable comparison=False.
