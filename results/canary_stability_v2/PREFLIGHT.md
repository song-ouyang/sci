# Frozen bounded stability session

Prepared 30 September 2026 before any new inference. This document is a preflight record, not a result.

- Frozen protocol SHA-256: `96401f1fe50b62e43f46f0e99834bec298e7b60664bcf1c56dc2922d1a7411fe`.
- 18 jobs; C/R × 3 chronological blocks × 3 byte-identical attempts; 2 workers; no retries.
- Same rig, 18 new focal cycles from 9 new blocks and 9 new canaries from 9 other new blocks; all 27 cycles and 18 blocks excluded from prior hydraulic selections.
- 28 features; all 27 selected serialized feature vectors distinct.
- Existing subscription transport only; model gpt-5.6-sol, medium; CLI 0.159.0 versus 0.155.0-alpha.9.2 originally.
- 600000 observed input+output-token dispatch threshold; up to two in-flight requests can overshoot. Estimated 402481 tokens from original mean. No promise of remaining account capacity.
- 3600-second session limit, 900-second per-call timeout, no dispatch when fewer than 900 seconds remain.
- Eight new offline gate/selection tests and ten inherited runner tests passed. Runner tests mock the CLI subprocess and test label separation, hashes, input tampering, event isolation, no-retry and bounded stopping.
- Independent validate reports 18 groups, 3 blocks, label_access=false; no real inference executed during preflight.

Launch, then audit and freeze before scoring:

```sh
.venv/bin/python src/experiments/run_hydraulic_v1.py run --root results/canary_stability_v2 --workers 2
.venv/bin/python src/experiments/run_hydraulic_v1.py audit --root results/canary_stability_v2
.venv/bin/python src/experiments/run_hydraulic_v1.py freeze --root results/canary_stability_v2
.venv/bin/python src/analysis/canary_controls_v2.py --root results/canary_stability_v2 --out results/canary_stability_v2/analysis
```

Primary Known_answer and Collapse_90, all sensitivity thresholds and tie rules are fixed in the hashed scorer. All planned windows retain denominators even if execution stops. New source, canary panel, time period and CLI jointly differ: results cannot identify their separate causal effects or cross-device transfer.
