# Frozen-output context and repeat audit

This is a post hoc descriptive analysis prepared during the 2026-09-28 author revision. It adds no model calls or independent source units and does not change the original protocol, frozen outputs or scores.

Run locally from the project root:

```sh
.venv/bin/python results/focused_review/rebuild_context_repeat_audit.py
```

The script verifies original frozen hashes for 108 request/result/metadata files, confirms byte-identical requests for 36 same-condition repeat pairs, recomputes 36 focal correct counts against the original scores, and compares predictions by the 21 fixed focal IDs. Each condition is reported separately; all repeat pairs are retained.

- `context_repeat_audit.json`: summaries, verification counts and input hashes.
- `identical_input_focal_flips.csv`: same-condition cross-repeat flips.
- `fixed_size_context_focal_flips.csv`: A/B and front/back contrasts, both requests containing 42 records.
- `focal_correct_counts.csv`: each condition/repeat's focal correct count and full request size.
- `focal_disagreement_cases.csv`: case-level paired decisions for every reported comparison.

These repeated-call comparisons reuse cases and calls. They are not independent observations; no confidence intervals, p-values or noise-subtracted context effects are estimated. Conditional error rates under a fixed output contract and software normalization properties should not be confused with mitigation effectiveness.
