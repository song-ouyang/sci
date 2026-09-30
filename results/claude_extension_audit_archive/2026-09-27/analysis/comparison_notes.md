# Comparison eligibility: withheld pending a valid experiment

The first Claude extension cannot enter the controlled GPT comparison on the
present record. See [the integrity audit](../audit/report.md).

The originally submitted table used available-repeat means, while the GPT
manuscript's primary comparison uses repeat 1. Even with valid provenance,
these estimands would have to be reported separately. The old Claude scorer
also used percentile bootstrap accuracy intervals despite a protocol specifying
Wilson intervals; it did not implement the requested paired contrasts or
operational miss/false-alarm summaries.

The confirmed `CONTAMINATED` flag was ignored by that scorer. Removing that row
alone would not resolve missing native logs, retrospective request construction,
tool access, model-identity inconsistencies and the additional hash/response
problems affecting other records.

Current score summaries therefore contain N/A classification metrics for the
controlled comparison. Original numerical calculations are preserved in
`results/claude_extension_audit_archive/2026-09-27/`, not relabeled as verified
results. No conclusion about which Claude model is better is drawn here.

Identical labels or perfect accuracy cannot by themselves identify internal
reasoning, correct use of sensor descriptions, benchmark memorization, or copying.
Such claims require appropriate controls and independently verifiable inference.
