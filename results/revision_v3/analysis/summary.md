# Revision v3: frozen exploratory comparison

Verified all 27 planned terminal outcomes: 26 valid sessions and 1 semantic failures, with 1740/1800 valid predictions before opening answer labels. Repeat 1 was designated primary before inference. Rates below are percentages.

Descriptive 95% intervals conditional on the fixed selected devices/runs and observed grouped-session predictions. Resampling devices/runs does not rerun the model or capture dependence induced by a shared batch, model/session variability, selection bias, or benchmark contamination. They are not full independent-experiment intervals.

## Primary clean comparison

| Dataset | Method | Accuracy, Wilson 95% | Macro F1, conditional group bootstrap 95% | Misses |
|---|---|---:|---:|---:|
| cmapss | gpt-5.5 | N/A | N/A | N/A |
| tep | gpt-5.5 | 72.9 [61.5, 81.9] | 72.1 [69.4, 75.2] | 1/60 |
| cmapss | gpt-5.6-sol | 71.7 [59.2, 81.5] | 71.6 [59.8, 81.9] | 2/20 |
| tep | gpt-5.6-sol | 100.0 [94.8, 100.0] | 100.0 [100.0, 100.0] | 0/60 |
| cmapss | gpt-6-astra | 71.7 [59.2, 81.5] | 72.5 [60.4, 82.8] | 5/20 |
| tep | gpt-6-astra | 100.0 [94.8, 100.0] | 100.0 [100.0, 100.0] | 0/60 |
| cmapss | RF | 81.7 [70.1, 89.4] | 81.4 [70.4, 90.3] | 6/20 |
| cmapss | SVM | 75.0 [62.8, 84.2] | 74.1 [62.1, 84.2] | 10/20 |
| cmapss | Majority | 33.3 [22.7, 45.9] | 16.7 [11.9, 20.7] | 20/20 |
| tep | RF | 100.0 [94.8, 100.0] | 100.0 [100.0, 100.0] | 0/60 |
| tep | SVM | 100.0 [94.8, 100.0] | 100.0 [100.0, 100.0] | 0/60 |
| tep | Majority | 14.3 [7.9, 24.3] | 3.6 [1.9, 5.3] | 60/60 |

C-MAPSS misses are Critical classified as Healthy/Degrading. TEP misses are a fault classified as Normal; confusion between fault types is an accuracy/F1 error but not a fault-as-normal miss.

## Primary descriptor mapping stress test

The same TEP observations and numeric strings receive a cyclic +1 rotation of all sensor descriptors, including units. A positive difference means the clean condition scores higher. This is a synthetic gross mapping defect.

| Model | Accuracy difference, pp [95%] | Macro F1 difference, pp [95%] | Prediction flip rate [95%] |
|---|---:|---:|---:|
| gpt-5.5 | 28.6 [11.4, 45.7] | 34.2 [27.7, 40.3] | 71.4 [60.0, 81.4] |
| gpt-5.6-sol | 77.1 [67.1, 87.1] | 77.1 [72.6, 82.4] | 77.1 [67.1, 87.1] |
| gpt-6-astra | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## Repeated-session sensitivity

Three planned sessions reuse the same observations and fixed order. Ranges use the available valid sessions and are not an independent-observation confidence interval. No repeated predictions are pooled into a larger test set.

| Dataset / condition | Model | Accuracy r1 / r2 / r3 | Macro F1 r1 / r2 / r3 |
|---|---|---:|---:|
| cmapss / clean | gpt-5.5 | N/A / 68.3 / 66.7 | N/A / 68.9 / 67.5 |
| tep / clean | gpt-5.5 | 72.9 / 80.0 / 57.1 | 72.1 / 77.4 / 52.4 |
| tep / misnamed | gpt-5.5 | 44.3 / 40.0 / 30.0 | 37.9 / 37.7 / 26.3 |
| cmapss / clean | gpt-5.6-sol | 71.7 / 68.3 / 70.0 | 71.6 / 68.9 / 70.6 |
| tep / clean | gpt-5.6-sol | 100.0 / 70.0 / 100.0 | 100.0 / 70.0 / 100.0 |
| tep / misnamed | gpt-5.6-sol | 22.9 / 18.6 / 24.3 | 22.9 / 18.6 / 23.8 |
| cmapss / clean | gpt-6-astra | 71.7 / 73.3 / 75.0 | 72.5 / 74.0 / 75.5 |
| tep / clean | gpt-6-astra | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 |
| tep / misnamed | gpt-6-astra | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 |

## Semantic failures and unavailable comparisons

A pre-score reporting amendment retains failed sessions as N/A; it was not an originally prespecified missing-data rule. No partial response is scored, no missing prediction is manually filled, and no later repeat replaces primary repeat 1.

- gpt-5.5/cmapss_clean_r1: Incomplete prediction count; raw array count 59, expected 60; zero observations scored from this session.

34/39 planned contrasts are available. Every unavailable contrast and its reason is in unavailable_contrasts.csv; valid contrasts are in paired.csv. Intervals are unadjusted and descriptive; no significance-based claim selection was performed. The bootstrap uses 10,000 group resamples with seed 20260926. At n=60/70, worst-case Wilson half-widths are approximately 12.3/11.4 percentage points, limiting small-effect and ranking conclusions.
