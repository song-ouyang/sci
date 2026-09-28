# Exploratory mechanism extension

Post-v3, same-source follow-up; no independent confirmation or deployment claim.

{'planned': 84, 'valid': 84, 'requested_labels': 2604, 'valid_labels': 2604}

| Model | Condition | R1 accuracy | R2 | R3 |
|---|---|---:|---:|---:|
| gpt-5.6-sol | clean | 100.0% | 100.0% | 100.0% |
| gpt-5.6-sol | rotation | 24.3% | 52.9% | 21.4% |
| gpt-5.6-sol | pair_shuffle | 100.0% | 71.4% | 57.1% |
| gpt-5.6-sol | anonymous | 100.0% | 94.3% | 100.0% |
| gpt-5.6-sol | random_mapping | 50.0% | 55.7% | 52.9% |
| gpt-5.6-sol | sample_shuffle | 100.0% | 100.0% | 98.6% |
| gpt-5.6-sol | subset_batch | 28.6% | 28.6% | 28.6% |
| gpt-6-astra | clean | 100.0% | 100.0% | 100.0% |
| gpt-6-astra | rotation | 100.0% | 100.0% | 100.0% |
| gpt-6-astra | pair_shuffle | 100.0% | 100.0% | 100.0% |
| gpt-6-astra | anonymous | 100.0% | 100.0% | 100.0% |
| gpt-6-astra | random_mapping | 100.0% | 100.0% | 100.0% |
| gpt-6-astra | sample_shuffle | 100.0% | 100.0% | 100.0% |
| gpt-6-astra | subset_batch | 85.7% | 85.7% | 85.7% |

Paired contrasts and seven-case context probes: see scores.json. Missing calls remain unavailable.
