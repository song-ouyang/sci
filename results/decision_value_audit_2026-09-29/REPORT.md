# Descriptive decision-value audit (2026-09-29 schema; completed 2026-09-30)

This result-informed offline analysis adds no model calls and changes no policy or frozen outcome. Same retains F1 when F1=F2; View retains F1 when F1=R2. Both use two requests per panel-round, although observed token use and duration differ. Rejecting a base-correct answer is a lost correct decision; rejecting a base-error only withholds it and does not establish that anyone repaired it.

Assign loss 0 to an accepted correct diagnosis, loss 1 to an accepted error, and loss c to every rejection/referral. Here c is the TOTAL expected downstream loss of withholding/referring a diagnosis, which may include handling, delay and residual errors after referral; it does not assume that a human perfectly repairs rejected errors. A common c for all rejections and a common unit loss for all accepted errors are simplifying sensitivity assumptions, not measured factory costs. Heterogeneous severity, class-dependent referral outcomes and referral capacity are not modeled. The inspected range 0<=c<=1 is illustrative, not a claim that real referral losses cannot exceed 1.

For policy p, L_p(c,K_p)=(E_p+cR_p+K_p)/N, where E_p is accepted errors, R_p all rejections, N the planned focal decisions, and K_p the TOTAL loss attributable to that policy's prescribed diagnostic requests in the same normalized units. A hypothetical per-request loss kappa would give K_p=kappa*Q_p; alternatively one may supply an independently justified K_p using tokens, elapsed time or other resources without counting downstream referral loss twice. The function empirical_loss(stats,c,total_request_loss) supports this scenario input. No monetary calibration is inferred here. All reported curves, crossings and sensitivity summaries set K_p=0; consequently they are decision-loss comparisons that omit request costs, NOT cost-effectiveness estimates. They do not include downstream referral outcomes beyond the assumed c.

Accept-all uses the same initial F1 predictions (one request per panel-round); reject-all needs no diagnostic requests but incurs c for every decision. Only Same/View have equal request counts; their measured tokens and summed client service durations differ, so their actual K_p need not be equal. For unequal K_p the View-minus-Same numerator gains K_View-K_Same and the displayed crossings move. All policy resource totals, including accept-all and reject-all, are retained in audit.json.

|Dataset|Policy|N|Accepted correct|Accepted errors|Rejected base-correct|Rejected base-error|Coverage|Correct yield|Empirical loss|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|hydraulic|Accept_all_F1|54|44|10|0|0|1.000|0.815|(10+0c)/54|
|hydraulic|Same|54|33|4|11|6|0.685|0.611|(4+17c)/54|
|hydraulic|View|54|39|5|5|5|0.815|0.722|(5+10c)/54|
|hydraulic|Reject_all|54|0|0|44|10|0.000|0.000|(0+54c)/54|
|paderborn|Accept_all_F1|72|11|61|0|0|1.000|0.153|(61+0c)/72|
|paderborn|Same|72|8|58|3|3|0.917|0.111|(58+6c)/72|
|paderborn|View|72|9|21|2|40|0.417|0.125|(21+42c)/72|
|paderborn|Reject_all|72|0|0|11|61|0.000|0.000|(0+72c)/72|

## English text for manuscript integration

A result-informed loss sensitivity analysis compared the fixed two-request checks without selecting a new deployment rule. When total request loss is omitted (K_p=0), hydraulic Same and View have empirical losses (4+17c)/54 and (5+10c)/54. Their difference, View minus Same, is (1-7c)/54 and favors View only when c>1/7: its six additional accepted correct diagnoses accompany one additional accepted error. Compared with accepting every F1 output, the empirical break-even referral losses are 6/17 for Same and 1/2 for View. Neither check improves every endpoint.

In the bearing study, the corresponding losses are (58+6c)/72 and (21+42c)/72. View has lower observed loss than Same throughout 0<=c<=1, but this comparison alone does not establish useful deployment: rejecting every diagnosis has lower observed loss than View when c<7/10, and accepting every F1 diagnosis has lower observed loss when c>20/21. Only between these empirical intersections does View beat both reference actions under this loss specification with K_p=0. The retained error fraction remains 21/30 (70.0%). These are descriptive sample-dependent tradeoffs, not optimized, validated or calibrated industrial decisions.

The decision counts contain two rounds on 27 hydraulic cycles and 36 bearing recordings, not 54 or 72 independent devices. Source and round breakdowns retain the dependence structure. The hydraulic data concern one rig; its 27 condition blocks are not 27 physical rigs, and no rig-level confidence interval is estimated. For Paderborn, class-stratified paired resampling keeps all four regimes and both rounds together for each of nine physical bearings. Its 2.5th--97.5th percentiles describe sensitivity to empirical specimen weights conditional on the observed requests and fixed backgrounds. Shared batch responses leave dependence across bearings, so these bands are not population confidence intervals or evidence of cross-rig generalization.

## Round and source sensitivity (request costs omitted)

Round and deletion sensitivity limits any stability claim. Hydraulic Same and View have equal aggregate counts in round 1; the entire six-correct/one-error increment occurs in round 2. Deleting one hydraulic panel moves the View--Same crossing among 0, 1, 1/7; therefore the pooled crossing is not a stable engineering constant.

For bearings, View reduces accepted errors in both rounds (by 20 and 17) but retains -1 and +2 correct diagnoses relative to Same. Its round-1 loss becomes higher than Same for c>20/21; round 2 favors View over the entire displayed range. Deleting each physical bearing still reduces accepted errors (by 29--40), but deleting KI04 reverses the high-cost comparison for c>35/37. Thus error-release reduction survives these particular deletions, whereas a uniformly lower loss does not.

At c=1 the conditional bearing reweighting percentiles for View-minus-Same loss are [-0.0972, 0.0556], spanning both signs. This is a sensitivity to the weights assigned to the nine observed bearings, not a population confidence interval or a validation on additional specimens.

## Reproduction and boundaries

Run `.venv/bin/python src/analysis/audit_decision_value.py`. Frozen response member hashes, protocol-linked source/truth/dataset hashes, exported predictions against frozen parsed responses, and independent F1/F2/R2 gate reconstruction are checked before writing new outputs. Existing policy counts must match exactly. The analysis reads but never executes old inference or scoring code. No files in the source studies are changed.

`decision_cases.csv` has one row per initial decision and action; `summary.csv` reports the partition and normalized loss coefficients; `source_round_breakdown.csv` covers all sources, rounds, panels, true labels and operating groups. `loss_curves.csv` evaluates c from 0 to 1 in steps of 0.01. `intersections.csv` includes all exact pairwise intersections, including those outside the stipulated cost range; these are not fitted thresholds. `leave_one_group_out.csv` is deterministic deletion sensitivity. `bearing_resampling_sensitivity.csv` contains conditional percentiles with the limitations above. `audit.json` binds input/output hashes and resampling settings.

This audit was designed after aggregate outcomes were known. No subgroup, record, round, cost value or policy is dropped or selected for favorable performance. No hypothesis tests or causal effect claims are added. Two rounds cannot establish service stability. Equal request counts do not imply equal monetary cost or latency, and no token-to-currency conversion is assumed.
