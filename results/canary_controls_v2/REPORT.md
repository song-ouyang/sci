# Fixed canary/collapse control

posthoc_existing_27_calls

All tables reconstruct frozen outputs in integer schedule order. No model calls are made by this analysis.

- Single: AC=105, AE=57, withheld=0, calls/window=1.000.
- Known_answer: AC=123, AE=3, withheld=36, calls/window=1.889.
- Collapse_90: AC=123, AE=3, withheld=36, calls/window=1.889.
- Vote_first_tie: AC=102, AE=60, withheld=0, calls/window=3.000.
- Vote_abstain_tie: AC=97, AE=59, withheld=6, calls/window=3.000.
- Collapse_85: AC=123, AE=3, withheld=36, calls/window=1.889.
- Collapse_95: AC=105, AE=57, withheld=0, calls/window=1.000.
- Collapse_100: AC=105, AE=57, withheld=0, calls/window=1.000.

The full acceptance cross-tab and threshold sensitivity are retained. This control is not evidence that canaries can be removed from prompts without changing model behavior.
