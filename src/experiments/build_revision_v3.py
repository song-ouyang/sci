"""Build immutable revision-v3 sensor-only datasets and matched-feature baselines.

No old processed CSV, predictions or paper file is read or modified. Inference
must use only the public `prompt`/`sensor_lines`, never labels or provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyreadr
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_v3"
SEED = 20260926
WINDOW = 30
CM_SENSORS = [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
CM_NAMES = {
    2: ("LPC outlet temperature T24", "deg R"),
    3: ("HPC outlet temperature T30", "deg R"),
    4: ("LPT outlet temperature T50", "deg R"),
    6: ("Bypass duct pressure P15", "psia"),
    7: ("HPC outlet pressure P30", "psia"),
    8: ("Physical fan speed Nf", "rpm"),
    9: ("Physical core speed Nc", "rpm"),
    11: ("HPC outlet static pressure Ps30", "psia"),
    12: ("Fuel flow to Ps30 ratio phi", "pps/psi"),
    13: ("Corrected fan speed NRf", "rpm"),
    14: ("Corrected core speed NRc", "rpm"),
    15: ("Bypass ratio BPR", "dimensionless"),
    17: ("Bleed enthalpy htBleed", "dataset units"),
    20: ("HPT coolant bleed W31", "lbm/s"),
    21: ("LPT coolant bleed W32", "lbm/s"),
}
TEP_NAMES = [
    "A feed, stream 1 (kscmh)", "D feed, stream 2 (kg/hr)",
    "E feed, stream 3 (kg/hr)", "A and C feed, stream 4 (kscmh)",
    "Recycle flow, stream 8 (kscmh)", "Reactor feed rate, stream 6 (kscmh)",
    "Reactor pressure (kPa gauge)", "Reactor level (%)", "Reactor temperature (deg C)",
    "Purge rate, stream 9 (kscmh)", "Product separator temperature (deg C)",
    "Product separator level (%)", "Product separator pressure (kPa gauge)",
    "Product separator underflow, stream 10 (m3/hr)", "Stripper level (%)",
    "Stripper pressure (kPa gauge)", "Stripper underflow, stream 11 (m3/hr)",
    "Stripper temperature (deg C)", "Stripper steam flow (kg/hr)", "Compressor work (kW)",
    "Reactor cooling water outlet temperature (deg C)",
    "Separator cooling water outlet temperature (deg C)",
] + [f"Component {letter}, stream 6 (mol%)" for letter in "ABCDEF"] + [
    f"Component {letter}, stream 9 (mol%)" for letter in "ABCDEFGH"
] + [f"Component {letter}, stream 11 (mol%)" for letter in "DEFGH"] + [
    "D feed valve, stream 2 (%)", "E feed valve, stream 3 (%)", "A feed valve, stream 1 (%)",
    "A and C feed valve, stream 4 (%)", "Compressor recycle valve (%)", "Purge valve, stream 9 (%)",
    "Separator pot liquid valve, stream 10 (%)", "Stripper liquid product valve, stream 11 (%)",
    "Stripper steam valve (%)", "Reactor cooling water valve (%)", "Condenser cooling water valve (%)",
]
TEP_COLS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
TEP_FAULTS = [0, 1, 2, 4, 6, 7, 13]
TEP_SAMPLES = [200, 350, 500]
TEP_METADATA_URL = "https://api.datacite.org/dois/10.7910/DVN/6C3JR1"
CM_SOURCE_URL = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
TEP_SOURCE_URL = "https://doi.org/10.7910/DVN/6C3JR1"
TEP_SCHEMA_URL = "https://www.mathworks.com/help/predmaint/ug/industrial-process-anomaly-detection-using-statistical-process-control.html"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(path)


def cm_label(rul):
    return "Healthy" if rul > 80 else "Degrading" if rul > 30 else "Critical"


def opaque_ids_and_shuffle(records, seed):
    # The seed and opaque IDs do not depend on labels or source record numbers.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    records = [records[int(i)] for i in order]
    ids = [f"{int(x):08x}" for x in rng.integers(0, 2**32, size=len(records), dtype=np.uint64)]
    require(len(ids) == len(set(ids)), "Opaque ID collision")
    for rec, sid in zip(records, ids):
        rec["id"] = sid
    return records


def rounded(values, precisions):
    # Convert through exactly the printed decimal representation: ML and LLM
    # consume numerically identical values, including six-decimal slopes.
    return [float(f"{float(x):.{p}f}") for x, p in zip(values, precisions)]


def add_prompts(ds):
    for rec in ds["test"]:
        lines = [f"{name}: {val:.{p}f}" for name, val, p in zip(
            ds["feature_display_names"], rec["features"], ds["feature_precisions"])]
        rec["sensor_lines"] = lines
        rec["prompt"] = ds["prompt_header"] + "\n\n" + "\n".join(lines)


def cm_features(window):
    require(len(window) == WINDOW, "C-MAPSS window must contain exactly 30 observations")
    values = window[:, [4 + i for i in CM_SENSORS]]
    x = np.arange(WINDOW, dtype=float)
    xc = x - x.mean()
    slopes = (xc[:, None] * values).sum(axis=0) / (xc @ xc)
    stats = np.column_stack([values.mean(axis=0), values.std(axis=0, ddof=0), slopes, values[-1]])
    return rounded(stats.ravel(), [4, 4, 6, 4] * len(CM_SENSORS))


def build_cmapss():
    paths = {split: ROOT / f"data/cmapss/raw/{split}_FD001.txt" for split in ["train", "test"]}
    raw = {split: np.loadtxt(path) for split, path in paths.items()}
    rul = np.loadtxt(ROOT / "data/cmapss/raw/RUL_FD001.txt", dtype=int)
    require(all(a.shape[1] == 26 for a in raw.values()), "FD001 raw schema must have 26 columns")
    require(len(rul) == len(np.unique(raw["test"][:, 0])) == 100, "FD001 test engine/RUL counts differ")
    selected_by_train_std = [i for i in range(1, 22) if raw["train"][:, 4 + i].std(ddof=1) > 0.001]
    require(selected_by_train_std == CM_SENSORS, "Unexpected train-only sensor selection")
    records = {"train": [], "test": []}
    for split in ["train", "test"]:
        for unit in np.unique(raw[split][:, 0]).astype(int):
            trajectory = raw[split][raw[split][:, 0] == unit]
            trajectory = trajectory[np.argsort(trajectory[:, 1])]
            require(np.all(np.diff(trajectory[:, 1]) == 1), f"Noncontiguous engine {split}/{unit}")
            require(len(trajectory) >= WINDOW, f"Insufficient history {split}/{unit}")
            ends = np.unique(np.linspace(WINDOW - 1, len(trajectory) - 1, 5, dtype=int)) if split == "train" else [len(trajectory) - 1]
            for end in ends:
                end = int(end)
                window = trajectory[end - WINDOW + 1:end + 1]
                remaining = int(trajectory[-1, 1] - trajectory[end, 1]) if split == "train" else int(rul[unit - 1])
                records[split].append({
                    "group": f"FD001/{split}/engine-{unit:03d}",
                    "label": cm_label(remaining), "features": cm_features(window),
                    "provenance": {
                        "raw_file": str(paths[split].relative_to(ROOT)), "raw_partition": split,
                        "engine_id": int(unit), "end_cycle": int(trajectory[end, 1]),
                        "window_start_cycle": int(window[0, 1]), "window_end_cycle": int(window[-1, 1]),
                        "window_rows": WINDOW, "raw_row_indices_zero_based": np.flatnonzero(
                            (raw[split][:, 0] == unit) & (raw[split][:, 1] >= window[0, 1]) &
                            (raw[split][:, 1] <= window[-1, 1])).astype(int).tolist(),
                        "rul_label_only": remaining,
                        "rul_source": "train final cycle minus observation cycle" if split == "train" else "data/cmapss/raw/RUL_FD001.txt",
                    },
                })
    labels = ["Healthy", "Degrading", "Critical"]
    population = dict(Counter(r["label"] for r in records["test"]))
    rng = np.random.default_rng(SEED)
    chosen = []
    for label in labels:
        candidates = [r for r in records["test"] if r["label"] == label]
        chosen += [candidates[int(i)] for i in rng.choice(len(candidates), min(20, len(candidates)), replace=False)]
    names, displays = [], []
    for sensor in CM_SENSORS:
        name, unit = CM_NAMES[sensor]
        for stat in ["mean", "std", "slope", "last"]:
            names.append(f"sensor_{sensor}_{stat}")
            displays.append(f"{name}, {stat} ({unit + '/cycle' if stat == 'slope' else unit})")
    ds = {
        "valid_labels": labels, "feature_names": names, "feature_display_names": displays,
        "feature_precisions": [4, 4, 6, 4] * 15,
        "prompt_header": "Classify turbofan engine health using only the following sensor summaries. The summaries use exactly the last 30 observed cycles, including the current observation; std is the population standard deviation and slope is a least-squares linear slope per cycle. Choose exactly one label: Healthy (remaining useful life >80 cycles), Degrading (30<remaining useful life<=80 cycles), or Critical (remaining useful life<=30 cycles). Remaining useful life is not supplied; infer the stage from the sensor values.",
        "train": opaque_ids_and_shuffle(records["train"], SEED + 101),
        "test": opaque_ids_and_shuffle(chosen, SEED + 102),
        "methods": {
            "source_url": CM_SOURCE_URL, "split": "Official FD001 train/test trajectories; numeric engine IDs are namespaced by raw partition and do not denote the same engines.",
            "window": "Exactly 30 consecutive rows [t-29,t]; no observation after t is used for features.",
            "training_windows": "Up to five equally spaced observation endpoints per training engine from the first complete 30-row window to its final cycle.",
            "test_selection": "One last observed window per official test engine; seed-stratified selection of at most 20 engines per RUL class; no model-output-based selection.",
            "test_population_class_counts": population, "selected_sensors": CM_SENSORS,
            "sensor_selection": "Training-only sample standard deviation >0.001; asserted to yield the fixed 15 sensors.",
            "label_definition": "Healthy RUL>80; Degrading 30<RUL<=80; Critical RUL<=30. Test uses official RUL_FD001 at last observation; train uses failure cycle minus endpoint.",
            "seed": SEED, "metadata_excluded_from_features_and_prompt": ["unit_id", "cycle", "RUL", "operating settings", "source", "id", "label"],
            "n_train_independent_engines": 100, "n_test_independent_engines": len(chosen),
        },
    }
    add_prompts(ds)
    return ds, list(paths.values()) + [ROOT / "data/cmapss/raw/RUL_FD001.txt", ROOT / "data/cmapss/raw/readme.txt"]


def source_metadata():
    path = OUT / "data/tep_datacite_metadata.json"
    if not path.exists():
        with urlopen(TEP_METADATA_URL, timeout=40) as response:
            metadata = json.load(response)
        write_json(path, metadata)
    meta = json.loads(path.read_text())
    descriptions = meta["data"]["attributes"]["descriptions"]
    text = "\n".join(d["description"] for d in descriptions)
    require("faults were introduced 1 and 8 hours" in text, "Original publisher injection-time metadata not verified")
    require("sampled every 3 minutes" in text, "Sampling period metadata not verified")
    return path


def build_tep():
    metadata = source_metadata()
    require(len(TEP_NAMES) == len(TEP_COLS) == 52, "TEP display mapping must have 52 variables")
    rng = np.random.default_rng(SEED + 201)
    all_runs = rng.permutation(np.arange(1, 501)).astype(int)
    test_runs = {fault: all_runs[index * 10:(index + 1) * 10].tolist() for index, fault in enumerate(TEP_FAULTS)}
    train_runs = sorted(all_runs[70:170].tolist())
    test_samples = {int(run): int(rng.integers(200, 501)) for runs in test_runs.values() for run in runs}
    train_records, test_records = [], []
    sources = []
    schema = {}
    for is_faulty in [False, True]:
        path = ROOT / ("data/tep/raw/TEP_Faulty_Training.RData" if is_faulty else "data/tep/raw/TEP_FaultFree_Training.RData")
        print(f"Loading {path.name}", flush=True)
        loaded = pyreadr.read_r(str(path))
        require(len(loaded) == 1, "Expected one dataframe per TEP RData")
        object_name, frame = next(iter(loaded.items()))
        expected = ["faultNumber", "simulationRun", "sample"] + TEP_COLS
        require(frame.columns.tolist() == expected, f"Unexpected TEP schema in {path}")
        require(frame['simulationRun'].nunique() == 500, "Expected 500 TEP simulation runs")
        require(frame['sample'].min() == 1 and frame['sample'].max() == 500, "Unexpected TEP Training time index")
        schema[path.name] = {"object_name": object_name, "raw_rows": len(frame), "columns": expected}
        # Discard most of the large R dataframe before iterating. All subsequent
        # selection uses named columns; metadata is retained only for provenance.
        faults_here = TEP_FAULTS[1:] if is_faulty else [0]
        mask = frame['faultNumber'].isin(faults_here) & frame['simulationRun'].isin(train_runs + [r for f in faults_here for r in test_runs[f]])
        frame = frame.loc[mask].copy()
        del loaded
        gc.collect()
        for fault in faults_here:
            label = "Normal" if fault == 0 else f"Fault_{fault}"
            for split, runs in [("train", train_runs), ("test", test_runs[fault])]:
                samples = TEP_SAMPLES if split == "train" else None
                selected = frame[(frame['faultNumber'] == fault) & frame['simulationRun'].isin(runs)]
                for run in runs:
                    run_frame = selected[selected['simulationRun'] == run]
                    require(len(run_frame) == 500 and set(run_frame['sample']) == set(range(1, 501)), f"Incomplete TEP run {fault}/{run}")
                    for sample in samples if samples is not None else [test_samples[run]]:
                        match = run_frame[run_frame['sample'] == sample]
                        require(len(match) == 1, "TEP provenance does not uniquely identify a row")
                        row = match.iloc[0]
                        values = rounded(row[TEP_COLS].to_numpy(dtype=float), [4] * 52)
                        require(np.isfinite(values).all(), "Nonfinite selected TEP sensor value")
                        rec = {
                            "group": f"TEP/Training/run-{run:03d}", "label": label, "features": values,
                            "provenance": {"raw_file": str(path.relative_to(ROOT)), "rdata_object": object_name,
                                "raw_partition": "Training", "simulation_run": int(run), "sample": int(sample),
                                "fault_number_label_only": fault, "raw_dataframe_rowname": str(match.index[0]),
                                "minutes_from_run_start": 3 * int(sample),
                                "fault_injection_after_sample": 20 if fault else None,
                                "selected_time_policy": "sample>=200; faulty runs at least 9 hours after injection; no claim of strict physical steady state"},
                        }
                        (train_records if split == "train" else test_records).append(rec)
        sources.append(path)
        del frame
        gc.collect()
    labels = ["Normal"] + [f"Fault_{i}" for i in TEP_FAULTS[1:]]
    ds = {
        "valid_labels": labels, "feature_names": TEP_COLS,
        "feature_display_names": [f"{column} {name}" for column, name in zip(TEP_COLS, TEP_NAMES)],
        "feature_precisions": [4] * 52,
        "prompt_header": "Classify this Tennessee Eastman chemical process snapshot using only the measured and manipulated process variables below. Choose exactly one label: Normal (no fault); Fault_1 (step change in A/C feed ratio, B composition constant, stream 4); Fault_2 (step change in B composition, A/C ratio constant, stream 4); Fault_4 (step change in reactor cooling water inlet temperature); Fault_6 (A feed loss, stream 1); Fault_7 (C header pressure loss, reduced availability in stream 4); Fault_13 (slow drift in reaction kinetics). Each following line names one process variable and its unit.",
        "train": opaque_ids_and_shuffle(train_records, SEED + 202),
        "test": opaque_ids_and_shuffle(test_records, SEED + 203),
        "methods": {"source_url": TEP_SOURCE_URL, "publisher_metadata_url": TEP_METADATA_URL,
            "schema_reference_url": TEP_SCHEMA_URL,
            "variable_mapping_reference_url": "https://petertino.github.io/web/PAPERS/CCE_14.pdf",
            "split": "New run-disjoint split using only official Training RData; this is not the official Training-versus-Testing split.",
            "seed": SEED + 201, "training_simulation_runs": train_runs,
            "test_simulation_runs_by_fault": {str(f): runs for f, runs in test_runs.items()},
            "train_selection": "100 run IDs shared across the seven class scenarios, excluding all 70 test IDs; sample 200,350,500 per run/class (2100 records).",
            "test_selection": "10 distinct runs per class, globally 70 different run IDs; one seed-selected sample in inclusive range [200,500] per test run.",
            "fault_timing": "Publisher DOI metadata: Training faults injected after 1 hour; 3-minute sampling gives boundary 20. Select samples>=200 for an established post-injection regime, not a proven steady state.",
            "metadata_excluded_from_features_and_prompt": ["faultNumber", "simulationRun", "sample", "source", "id", "label"],
            "n_train_run_ids": len(train_runs), "n_train_fault_run_trajectories": len(train_runs) * len(TEP_FAULTS),
            "n_test_independent_run_ids": 70, "raw_schema": schema,
        },
    }
    add_prompts(ds)
    return ds, sources + [metadata]


def validate_datasets(data, verify_raw=False):
    report = {}
    for name, ds in data['datasets'].items():
        labels = ds['valid_labels']; width = len(ds['feature_names'])
        require(len(set(ds['feature_names'])) == width, f"{name}: duplicate features")
        require(len(ds['feature_display_names']) == len(ds['feature_precisions']) == width, "Feature mapping width")
        train, test = ds['train'], ds['test']
        require({r['group'] for r in train}.isdisjoint({r['group'] for r in test}), f"{name}: group leakage")
        for split, rows in [('train', train), ('test', test)]:
            require(set(r['label'] for r in rows) == set(labels), f"{name}/{split}: class coverage")
            require(len({r['id'] for r in rows}) == len(rows), f"{name}/{split}: duplicate IDs")
            for r in rows:
                require(re.fullmatch(r'[0-9a-f]{8}', r['id']) is not None, 'Invalid opaque ID')
                require(len(r['features']) == width and np.isfinite(r['features']).all(), 'Feature schema')
                require(r['features'] == rounded(r['features'], ds['feature_precisions']), 'Feature precision')
                if name == 'cmapss':
                    p = r['provenance']
                    require(p['window_rows'] == 30 and p['window_end_cycle'] - p['window_start_cycle'] == 29, 'Not exactly 30 cycles')
                    require(p['window_end_cycle'] == p['end_cycle'], 'Window uses future observation')
                    require(cm_label(p['rul_label_only']) == r['label'], 'C-MAPSS RUL label mismatch')
                else:
                    require(r['provenance']['sample'] >= 200 > 20, 'TEP pre-fault sample')
        require(len({r['group'] for r in test}) == len(test), f'{name}: repeated test group')
        for r in test:
            expected = [f'{display}: {value:.{precision}f}' for display, value, precision in zip(ds['feature_display_names'], r['features'], ds['feature_precisions'])]
            require(expected == r['sensor_lines'], 'Prompt and ML feature mismatch')
            require(r['prompt'] == ds['prompt_header'] + '\n\n' + '\n'.join(expected), 'Noncanonical prompt')
            require(r['id'] not in r['prompt'], 'ID leaked into prompt')
        report[name] = {'train_records': len(train), 'test_records': len(test), 'feature_count': width,
            'train_groups': len({r['group'] for r in train}), 'test_groups': len(test),
            'train_class_counts': dict(Counter(r['label'] for r in train)), 'test_class_counts': dict(Counter(r['label'] for r in test)),
            'group_disjoint': True, 'finite_features': True, 'prompt_feature_equality': True,
            'label_coverage': True, 'test_one_record_per_group': True}
    require(len(data['datasets']['cmapss']['test']) == 60, 'Expected 60 C-MAPSS test engines')
    require(len(data['datasets']['tep']['test']) == 70, 'Expected 70 TEP test runs')
    if verify_raw:
        cm = data['datasets']['cmapss']
        arrays = {split: np.loadtxt(ROOT/f'data/cmapss/raw/{split}_FD001.txt') for split in ['train','test']}
        test_rul = np.loadtxt(ROOT/'data/cmapss/raw/RUL_FD001.txt', dtype=int)
        for split in ['train','test']:
            for r in cm[split]:
                p=r['provenance']; a=arrays[split]; window=a[p['raw_row_indices_zero_based']]
                require(np.all(window[:,0]==p['engine_id']), 'Engine mismatch in raw window')
                require(np.array_equal(window[:,1],np.arange(p['end_cycle']-29,p['end_cycle']+1)), 'Window not consecutive past30')
                require(cm_features(window)==r['features'], 'Features differ from raw past-only window')
                if split=='test':
                    require(p['end_cycle']==a[a[:,0]==p['engine_id'],1].max(), 'Test not last observation')
                    require(p['rul_label_only']==int(test_rul[p['engine_id']-1]), 'Official RUL mismatch')
        report['cmapss']['all_raw_windows_recomputed_exactly']=True
    return report


def baseline_metrics(truth, guesses, labels):
    return {'n_samples': len(truth), 'correct': sum(a==b for a,b in zip(truth,guesses)),
        'accuracy_pct':100*accuracy_score(truth,guesses),
        'macro_f1_pct':100*f1_score(truth,guesses,labels=labels,average='macro',zero_division=0),
        'labels':labels,'confusion_matrix_true_rows':confusion_matrix(truth,guesses,labels=labels).tolist()}


def baselines(data, data_path):
    preds, report = {}, {'dataset_sha256': digest(data_path), 'seed':SEED, 'datasets':{}}
    for name,ds in data['datasets'].items():
        X=np.array([r['features'] for r in ds['train']],dtype=float); y=[r['label'] for r in ds['train']]
        test_X=np.array([r['features'] for r in ds['test']],dtype=float); truth=[r['label'] for r in ds['test']]
        labels=ds['valid_labels']; counts=Counter(y)
        majority=max(labels,key=lambda label:counts[label])
        models={'RF':RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1),
                'SVM':make_pipeline(StandardScaler(),SVC(C=1.0,kernel='rbf',gamma='scale'))}
        preds[name]={}; scores={}
        for label,model in models.items():
            model.fit(X,y); guesses=model.predict(test_X).tolist()
            preds[name][label]=[{'id':r['id'],'predicted':p} for r,p in zip(ds['test'],guesses)]
            scores[label]=baseline_metrics(truth,guesses,labels)
        guesses=[majority]*len(truth)
        preds[name]['Majority']=[{'id':r['id'],'predicted':p} for r,p in zip(ds['test'],guesses)]
        scores['Majority']=baseline_metrics(truth,guesses,labels)
        priors={label:counts[label]/len(y) for label in labels}
        report['datasets'][name]={'metrics':scores, 'training_class_priors':priors,
            'majority_label':majority, 'majority_tie_break':'valid_labels order',
            'prior_random_expected_accuracy_pct':100*sum(priors[label] for label in truth)/len(truth),
            'uniform_random_expected_accuracy_pct':100/len(labels),
            'random_macro_f1':'Not reported: F1 is nonlinear, so accuracy expectation is not an F1 estimate.',
            'train_records':len(y), 'test_records':len(truth),
            'settings':{'RF':{'n_estimators':300,'random_state':SEED,'other_parameters':'scikit-learn defaults'},
                'SVM':{'pipeline':['StandardScaler fitted to train only','SVC'], 'C':1.0,'kernel':'rbf','gamma':'scale','other_parameters':'scikit-learn defaults'}},
            'test_tuning':False, 'feature_input':'Exactly the stored, rounded features used to render LLM prompts.'}
        print(name, {k:round(v['accuracy_pct'],2) for k,v in scores.items()}, flush=True)
    report['versions']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'pyreadr':pyreadr.__version__}
    write_json(OUT/'baselines/predictions.json',preds)
    write_json(OUT/'baselines/scores.json',report)
    write_json(OUT/'baselines/manifest.json',{'dataset_sha256':digest(data_path), 'files':{name:digest(OUT/'baselines'/name) for name in ['predictions.json','scores.json']}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only',action='store_true')
    parser.add_argument('--baselines-only',action='store_true')
    args=parser.parse_args()
    path=OUT/'data/datasets.json'; manifest_path=OUT/'data/manifest.json'
    if path.exists():
        require(manifest_path.exists(),'Existing dataset lacks frozen manifest')
        manifest=json.loads(manifest_path.read_text())
        require(digest(path)==manifest['dataset_sha256'],'Frozen datasets.json changed')
        data=json.loads(path.read_text())
        validate_datasets(data,verify_raw=True)
        print('Reusing frozen datasets.json; no dataset writes',flush=True)
    else:
        require(not args.baselines_only,'Build data first')
        cmapss,cm_sources=build_cmapss()
        tep,tep_sources=build_tep()
        data={'schema_version':1,'experiment_version':'revision_v3','datasets':{'cmapss':cmapss,'tep':tep}}
        validation=validate_datasets(data,verify_raw=True)
        write_json(path,data)
        sources=cm_sources+tep_sources
        manifest={'schema_version':1,'dataset_sha256':digest(path),'seed':SEED,
            'builder':'src/experiments/build_revision_v3.py','builder_sha256':digest(Path(__file__)),
            'source_files':{str(p.relative_to(ROOT)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sources},
            'split_groups':{name:{split:sorted({r['group'] for r in ds[split]}) for split in ['train','test']} for name,ds in data['datasets'].items()},
            'selection_precision_and_methods':{name:ds['methods']|{'feature_precisions':ds['feature_precisions']} for name,ds in data['datasets'].items()},
            'validation':validation,'inference_boundary':'Only prompt/sensor_lines and valid_labels may be passed to models. label, provenance, group and source IDs are evaluation-only.',
            'freeze_policy':'datasets.json is immutable after creation. Reruns verify SHA-256 and reuse it; no overwrite option is provided.'}
        write_json(manifest_path,manifest)
        write_json(OUT/'data/validation.json',validation)
        print('DATA_READY '+str(path)+' SHA256 '+digest(path),flush=True)
    if not args.data_only:
        baselines(data,path)


if __name__=='__main__':
    main()
