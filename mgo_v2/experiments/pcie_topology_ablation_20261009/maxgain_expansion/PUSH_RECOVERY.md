# B16/O128 publication failure and recovery

The B16/O128 physical screen and final experiments passed, including five unfiltered repeats and eight separate diagnostics. The measured winner was the prior-winner control: R-NEAR median TPOT 1.0794297987952757 s, G-NEAR 1.0622144029133858 s, reduction 1.5948601660898731%. The individual experiment archives and matrix were committed before publication failed.

At 2026-10-10T13:54:45.247077+09:00, the final push exited 128 because the VS Code Git askpass socket refused connections; GitHub rejected anonymous writes. No GPU inference failed. The expansion supervisor stopped at publication, and the baseline waiter exited without starting inference. Original failure receipts, push output and supervisor logs remain under the external attempt directory. Idle burn on physical GPUs 0,1,4,5 was running during recovery.

After the same Git authentication socket became available, the recovery push succeeded and remote HEAD was verified as `0e25f6cb04091b6d576830ddf55747ad4287d2d0`. The branch was not force pushed.

`run_pcie_maxgain_expansion.py --resume` now accepts this specific failed-push state after checking completed GPU PASS receipts, previous publication receipts, completed-step consistency and byte-identical frozen registrations/selections. It snapshots the original failure state, preserves the failed log, retries only publication, and reuses all completed measurements. The next GPU cell is B16/O256; the remaining order is B32/O128, B32/O256, B64/O128, B64/O256. Failed measurements and owner interruptions do not qualify for this recovery path.

The CPU-only mocked continuation used the actual saved registrations and receipts. It issued exactly one push retry and reached B16/O256 without restarting any completed GPU job. This preflight is not a physical B16/O256 result. Syntax and whitespace checks also passed.

The baseline waiter must be relaunched against the resumed expansion PID/start identity. Its full eight-cell validation and final-push gate remain required.
