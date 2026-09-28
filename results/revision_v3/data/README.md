# Revision v3: clean sensor-only data and matched-feature baselines

This is a new, frozen evaluation. It does not replace or alter v1/v2 artifacts.

## Reproduce

From the project root:

```bash
.venv/bin/python3 -B src/experiments/build_revision_v3.py --data-only
.venv/bin/python3 -B src/experiments/build_revision_v3.py --baselines-only
```

The first invocation builds the data only when `datasets.json` is absent. Once
written, its SHA-256 is frozen in `manifest.json`; later invocations verify and
reuse it, with no overwrite switch. The second command trains fixed baselines
without changing the data. Original raw files are never modified. Source hashes,
software versions, selected groups, feature precision, and selection parameters
are recorded. Dataset rebuilds require a separate output version, not editing
this frozen JSON.

## Contract and inference boundary

`datasets.json` has `datasets.cmapss` and `datasets.tep`. Each dataset defines
`valid_labels`, `feature_names`, `feature_display_names`, `feature_precisions`,
`prompt_header`, `train`, `test`, and `methods`.

Every row contains `id`, `group`, `label`, `features`, and `provenance`. Every test
row additionally contains `sensor_lines` and a canonical `prompt`. The ID is an
opaque eight-character hex string; test rows are shuffled with fixed seeds.
`features` is already rounded to exactly the decimal values shown in the prompt.

**Only `prompt`/`sensor_lines` and the common `valid_labels` may enter inference.**
`label`, `provenance`, `group`, source indices, engine/run IDs, and RUL are strictly
evaluation-only information. A separate experimental wrapper may add randomized
IDs, but the canonical sensor-only prompt contains none. Every TEP variable is
one separate `descriptor: value` line, in canonical `xmeas_1..41,xmv_1..11` order;
`feature_display_names` includes code, name and unit, allowing controlled
name-only permutations without parsing free text. Such an intervention changes
only descriptors, not numeric values or their sequence.

## C-MAPSS FD001

- The official raw `train_FD001.txt` provides training trajectories;
  `test_FD001.txt` and `RUL_FD001.txt` provide held-out trajectories and final RUL.
  Numeric engine IDs are reused in the two raw files, but the files denote
  different engines, as explained in the distributed `readme.txt`.
- Training has at most five equally spaced observation endpoints per engine.
  Each test engine contributes only its last observed endpoint. Each feature
  window is exactly 30 consecutive observations, `[t-29,t]`, including `t` and
  excluding every future observation. The builder recomputes all stored windows
  directly from raw rows to verify this condition.
- Labels are Healthy for RUL >80, Degrading for 30<RUL<=80, and Critical for
  RUL<=30. Training RUL uses final failure cycle minus the endpoint. Test RUL uses
  the supplied official RUL vector, not the last observed cycle as a failure.
- The official 100-engine test population has 55 Healthy, 20 Degrading, and 25
  Critical engines. Seed-stratified selection retains 20 per class, giving 60
  distinct held-out engines. This balanced subset is not the natural deployment
  prevalence distribution.
- Sensors are selected using training-only standard deviation >0.001, yielding
  2,3,4,6,7,8,9,11,12,13,14,15,17,20,21. Mean, population standard deviation,
  least-squares slope and last value give 60 features. Mean/std/last use four
  decimals; slope uses six. Engine number, cycle, operating settings and RUL
  never enter features or prompts. htBleed's unspecified unit is conservatively
  called `dataset units`.
- Source: [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
  The original archive's README is also hashed in the manifest.

## Tennessee Eastman Process

- Source: [Rieth et al. publisher dataset](https://doi.org/10.7910/DVN/6C3JR1).
  The exact publisher-supplied DOI description is saved as
  `tep_datacite_metadata.json`, retrieved through the
  [DataCite registration API](https://api.datacite.org/dois/10.7910/DVN/6C3JR1).
  It establishes 500 samples per Training run, 3-minute sampling, and fault
  injection after one hour (sample boundary 20). The larger Testing partition
  has a different injection time; it is not used here.
- A new run-disjoint split is constructed from the two original **Training**
  RData files. This must not be described as the official Training/Testing split.
  Seventy globally distinct `simulationRun` IDs are assigned to test, ten per
  class (Normal and Fault_1/2/4/6/7/13). Each test run contributes one seeded
  random sample in inclusive range 200..500. Another 100 nonoverlapping run IDs
  contribute samples 200,350,500 for every class, giving 2100 training records.
  Sharing training run IDs across fault scenarios is recorded explicitly;
  training has 100 distinct RNG/run IDs and 700 fault-by-run trajectories.
- Selected faulty observations occur at least nine hours after injection. This
  is an established post-injection regime, not a claim that every fault reaches
  physical steady state (Fault_13 is a slow drift). Normal observations use the
  same time restriction. This task assesses late-state fault classification,
  not early detection or detection delay.
- Exactly the named raw columns `xmeas_1..41,xmv_1..11` are used. Metadata columns
  `faultNumber`, `simulationRun`, and `sample` remain only in provenance and never
  enter numerical features or prompts. The builder requires the original
  55-column schema and complete 500-point selected runs.
- Variable 22 is separator cooling-water outlet temperature. Variables 23..28,
  29..36, and 37..41 are composition measurements for streams 6,9,11 respectively;
  variable 41 is component H in stream 11. The 11 manipulated variables are
  valve command percentages. All values are printed/stored to four decimals.
  Reference: [variable mapping, Table 2](https://petertino.github.io/web/PAPERS/CCE_14.pdf),
  [MathWorks documentation of this same data release](https://www.mathworks.com/help/predmaint/ug/industrial-process-anomaly-detection-using-statistical-process-control.html).

## Baseline comparability

`../baselines/predictions.json` contains `{dataset: {RF: [{id,predicted}],
SVM: [...], Majority: [...]}}`. Prediction order exactly follows the frozen test
rows. Baselines see precisely the same stored rounded numeric features as LLMs.

RF uses 300 trees and the fixed seed. SVM uses a training-only StandardScaler and
RBF kernel, C=1, gamma=scale. No model selection or parameter tuning uses test
labels. Majority predicts the training majority label, resolving ties by
`valid_labels` order. `scores.json` records complete-set accuracy/macro F1,
confusion matrices, training priors, and analytical prior-random/uniform-random
expected accuracy. Expected random macro F1 is not approximated from accuracy.
These are supervised comparisons; LLM prompts contain task definitions but no
labeled examples or learned normal-reference statistics.

## Validation

`validation.json` records class coverage, complete shapes, finite values,
precision equality, prompt/numeric-feature equality, disjoint training/test
groups, exactly one test row per independent group, and raw C-MAPSS window/RUL
recomputation. Raw source hashes and exact selected group lists are in
`manifest.json`. This does not prove real-world generalization or unseen-model
training data: both benchmarks are public simulations, and this evaluation is
limited to FD001 and seven selected TEP operating conditions.
