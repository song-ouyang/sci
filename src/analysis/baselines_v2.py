"""Stronger, tuned supervised references for the LLM fault-diagnosis panels (baselines v2).

Answers the reviewer criticism that the earlier references (RF 300 trees and an
RBF SVM with C=1, gamma=scale, both untuned) were weak. Everything here is fit on
training data only; test labels are read only to score finished predictions.

Two families are kept strictly apart:

1. ``original_train`` (and the reproduced ``legacy_untuned`` reference): six
   tuned model families trained on exactly the stored, rounded training features
   of revision v3 (TEP late snapshots of the 100 training runs; FD001 windows of
   the 100 training engines) and scored on the same numeric test features the LLM
   receives. This is the like-for-like reference.
2. Fair-transfer variants, which receive training information the LLM does NOT
   get (phase-matched TEP snapshots, labelled FD002 training engines, or FD002
   operating-regime statistics). They bound what supervised learning can do when
   given the target condition; they are not like-for-like comparisons.

Hyperparameters are chosen by GroupKFold cross-validation grouped by training
run / engine (macro-F1), never on a test panel. Run from the project root:

    .venv/bin/python -B src/analysis/baselines_v2.py

Outputs (only new files): results/baselines_v2/{cases.csv, scores.json,
summary.md, tep_early_training_snapshots.csv, fd002_training_cutpoints.csv}.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import pyreadr
import sklearn
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.score_revision_v3 import wilson  # noqa: E402
from src.experiments.build_revision_v3 import (  # noqa: E402
    CM_SENSORS, TEP_COLS, WINDOW, cm_features, cm_label, digest, require, rounded, write_json)

OUT = ROOT / "results/baselines_v2"
V3_DATA = ROOT / "results/revision_v3/data/datasets.json"
IV_ANSWERS = ROOT / "results/independent_validation/answers/records.json"
IV_CASES = ROOT / "results/independent_validation/analysis/cases.csv"
IV_SCORES = ROOT / "results/independent_validation/analysis/scores.json"
V3_BASELINE_PREDICTIONS = ROOT / "results/revision_v3/baselines/predictions.json"
V3_CASES = ROOT / "results/revision_v3/analysis/reuse/clean_correctness.csv"
CM_RAW = ROOT / "data/cmapss/raw"
TEP_RAW = {False: ROOT / "data/tep/raw/TEP_FaultFree_Training.RData",
           True: ROOT / "data/tep/raw/TEP_Faulty_Training.RData"}

# Model seed equals the original revision-v3 seed so that the legacy RF is
# reproduced exactly; all other random choices derive from it with fixed offsets.
SEED = 20260926
CV_FOLDS = 5
EARLY_SAMPLE = 40            # same definition as the independent early test panel
FD002_CUTS_PER_CLASS = 2     # sampled training cut points per engine and RUL class
N_REGIMES = 6                # FD002 operating conditions
TEP_LABELS = ["Normal", "Fault_1", "Fault_2", "Fault_4", "Fault_6", "Fault_7", "Fault_13"]
CM_LABELS = ["Healthy", "Degrading", "Critical"]
TEP_FAULT_OF = {label: 0 if label == "Normal" else int(label.split("_")[1]) for label in TEP_LABELS}
LLM_REFERENCE = {  # where the paired LLM per-case predictions for each panel live
    "tep_late": ("results/independent_validation/analysis/cases.csv", "condition=clean"),
    "tep_early": ("results/independent_validation/analysis/cases.csv", "condition=early"),
    "fd002": ("results/independent_validation/analysis/cases.csv", "condition=fd002"),
    "tep_v3": ("results/revision_v3/analysis/reuse/clean_correctness.csv", "dataset=tep"),
    "fd001_v3": ("results/revision_v3/analysis/reuse/clean_correctness.csv", "dataset=cmapss"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matrix(records):
    return np.array([r["features"] for r in records], dtype=float)


# --------------------------------------------------------------------------- panels
def load_panels(v3, answers):
    """Test panels with the exact IDs, labels and numeric features shown to the LLMs."""
    def pick(rows):
        return [dict(id=r["id"], label=r["label"], group=r["group"], features=r["features"],
                     provenance=r["provenance"]) for r in rows]
    tep_new = [r for r in answers if r["dataset"] == "tep" and r["role"] == "main"]
    panels = {
        "tep_late": pick([r for r in tep_new if r["phase"] == "late"]),
        "tep_early": pick([r for r in tep_new if r["phase"] == "early"]),
        "tep_v3": pick(v3["tep"]["test"]),
        "fd001_v3": pick(v3["cmapss"]["test"]),
        "fd002": pick([r for r in answers if r["dataset"] == "cmapss_fd002"]),
    }
    require(all(r["provenance"]["sample"] == EARLY_SAMPLE for r in panels["tep_early"]), "Early panel sample")
    require({r["group"] for r in panels["tep_early"]} == {r["group"] for r in panels["tep_late"]},
            "Early and late panels must use the same 70 runs")
    return panels


def check_llm_ids(panels):
    """Baseline panels must use exactly the IDs of the LLM case files (for McNemar pairing)."""
    iv = pd.read_csv(IV_CASES)
    v3 = pd.read_csv(V3_CASES)
    report = {}
    for panel, (_, selector) in LLM_REFERENCE.items():
        key, value = selector.split("=")
        frame = iv if panel in ["tep_late", "tep_early", "fd002"] else v3
        frame = frame[frame[key] == value]
        ids = {r["id"] for r in panels[panel]}
        for _, sub in frame.groupby(["model", "repeat"]):
            require(set(sub["id"]) == ids, f"{panel}: IDs differ from LLM case file")
        report[panel] = dict(llm_case_file=LLM_REFERENCE[panel][0], selector=selector,
                             model_repeat_sessions_matched=int(frame.groupby(["model", "repeat"]).ngroups))
    return report


# ----------------------------------------------------------------------- TEP extraction
def extract_tep(requests):
    """Return rounded 52-variable rows for (fault, run, sample) requests from the raw RData.

    Uses the same official Training RData files and 4-decimal rounding as the
    revision-v3 and independent-validation builders.
    """
    out = {}
    requests = set(requests)
    for faulty in [False, True]:
        wanted = {q for q in requests if (q[0] != 0) == faulty}
        if not wanted:
            continue
        print(f"Reading {TEP_RAW[faulty].name}", flush=True)
        frame = next(iter(pyreadr.read_r(str(TEP_RAW[faulty])).values()))
        require(list(frame.columns) == ["faultNumber", "simulationRun", "sample"] + TEP_COLS, "TEP raw schema")
        code = (frame["faultNumber"].to_numpy(dtype=np.int64) * 10**6
                + frame["simulationRun"].to_numpy(dtype=np.int64) * 10**3
                + frame["sample"].to_numpy(dtype=np.int64))
        codes = {f * 10**6 + run * 10**3 + s: (f, run, s) for f, run, s in wanted}
        mask = np.isin(code, list(codes))
        sub = frame.loc[mask, TEP_COLS].to_numpy(dtype=float)
        keys = code[mask]
        require(len(keys) == len(set(keys.tolist())) == len(wanted), "TEP request not uniquely matched")
        for key, row in zip(keys.tolist(), sub):
            out[codes[key]] = rounded(row, [4] * 52)
        del frame, sub
    return out


def tep_training_snapshots(v3, panels):
    """Early (sample 40) snapshots of the 100 TEP training runs for all seven classes.

    Also re-extracts every stored late training row and every independent early/late
    test row from raw data to prove that the extraction matches the frozen features.
    """
    train_runs = sorted({r["provenance"]["simulation_run"] for r in v3["tep"]["train"]})
    require(len(train_runs) == 100, "Expected 100 TEP training runs")
    early_req = [(TEP_FAULT_OF[label], run, EARLY_SAMPLE) for label in TEP_LABELS for run in train_runs]
    check = {}
    for r in v3["tep"]["train"]:
        p = r["provenance"]
        check[(p["fault_number_label_only"], p["simulation_run"], p["sample"])] = r["features"]
    for panel in ["tep_late", "tep_early"]:
        for r in panels[panel]:
            p = r["provenance"]
            check[(p["fault_number"], p["simulation_run"], p["sample"])] = r["features"]
    rows = extract_tep(early_req + list(check))
    mismatches = sum(rows[k] != v for k, v in check.items())
    require(mismatches == 0, f"{mismatches} re-extracted TEP rows differ from frozen features")
    early = [dict(label=label, group=f"TEP/Training/run-{run:03d}", features=rows[(TEP_FAULT_OF[label], run, EARLY_SAMPLE)],
                  provenance=dict(simulation_run=run, sample=EARLY_SAMPLE, fault_number=TEP_FAULT_OF[label]))
             for label in TEP_LABELS for run in train_runs]
    audit = dict(early_training_records=len(early), reextracted_and_matched_exactly=len(check),
                 reextracted_sets=["2100 revision-v3 late training rows", "70 independent late test rows",
                                   "70 independent early (sample 40) test rows"])
    return early, audit


# ---------------------------------------------------------------------------- C-MAPSS
def load_cmapss(name):
    raw = np.loadtxt(CM_RAW / f"{name}.txt")
    require(raw.shape[1] == 26, f"{name}: raw schema")
    return raw


def trajectory(raw, unit):
    traj = raw[raw[:, 0] == unit]
    traj = traj[np.argsort(traj[:, 1])]
    require(np.all(np.diff(traj[:, 1]) == 1), f"Noncontiguous engine {unit}")
    return traj


class RegimeNormalizer:
    """Operating-regime-wise sensor standardization, standard in the C-MAPSS literature.

    Regimes are KMeans clusters of the three standardized operating settings
    (raw columns 2-4), fitted on training rows only; every sensor (columns 5-25)
    is then z-scored with the training mean/std of its regime. With one regime this
    reduces to global per-sensor standardization (FD001 has a single condition).
    """

    def __init__(self, n_regimes, seed=SEED):
        self.n_regimes, self.seed = n_regimes, seed

    def fit(self, raw):
        self.scaler = StandardScaler().fit(raw[:, 2:5])
        z = self.scaler.transform(raw[:, 2:5])
        self.kmeans = KMeans(self.n_regimes, n_init=10, random_state=self.seed).fit(z) if self.n_regimes > 1 else None
        regime = self.regimes(raw)
        sensors = raw[:, 5:26]
        self.mean = np.array([sensors[regime == k].mean(axis=0) for k in range(self.n_regimes)])
        std = np.array([sensors[regime == k].std(axis=0, ddof=0) for k in range(self.n_regimes)])
        self.std = np.where(std < 1e-8, 1.0, std)  # constant-within-regime sensors map to 0
        self.train_regime_counts = np.bincount(regime, minlength=self.n_regimes).tolist()
        return self

    def regimes(self, raw):
        if self.kmeans is None:
            return np.zeros(len(raw), dtype=int)
        return self.kmeans.predict(self.scaler.transform(raw[:, 2:5]))

    def max_centroid_distance(self, raw):
        # Standardized-setting distance of each row to its assigned centroid; small
        # values mean the six conditions are recovered unambiguously.
        if self.kmeans is None:
            return 0.0
        z = self.scaler.transform(raw[:, 2:5])
        return float(np.min(self.kmeans.transform(z), axis=1).max())

    def transform(self, raw):
        out = raw.copy()
        regime = self.regimes(raw)
        out[:, 5:26] = (raw[:, 5:26] - self.mean[regime]) / self.std[regime]
        return out


def fd002_training_cutpoints(raw):
    """Balanced labelled windows from train_FD002 run-to-failure trajectories.

    For every training engine and every RUL class, FD002_CUTS_PER_CLASS observation
    endpoints are drawn without replacement (fixed seed) among those with a complete
    30-cycle past window whose RUL (= failure cycle - endpoint cycle) falls into the
    class. Bins match the test panel: Healthy RUL>80, Degrading 30<RUL<=80,
    Critical RUL<=30. The FD002 test panel is balanced (20/20/20), so classes are
    balanced by construction.
    """
    rng = np.random.default_rng(SEED + 302)
    cuts = []
    for unit in np.unique(raw[:, 0]).astype(int):
        traj = trajectory(raw, unit)
        life = int(traj[-1, 1])
        for label in CM_LABELS:
            ends = [i for i in range(WINDOW - 1, len(traj)) if cm_label(life - int(traj[i, 1])) == label]
            for i in sorted(rng.choice(ends, min(FD002_CUTS_PER_CLASS, len(ends)), replace=False).tolist()):
                cuts.append(dict(engine_id=int(unit), group=f"FD002/train/engine-{unit:03d}", label=label,
                                 end_index=int(i), end_cycle=int(traj[i, 1]), failure_cycle=life,
                                 rul_label_only=life - int(traj[i, 1])))
    return cuts


def window_features(raw, unit, end_index):
    traj = trajectory(raw, unit)
    return cm_features(traj[end_index - WINDOW + 1:end_index + 1])


def last_window_features(raw, unit):
    traj = trajectory(raw, unit)
    return cm_features(traj[-WINDOW:])


def row_overlap(test_raw, test_units, train_raw):
    """Count raw test-window rows (settings + all 21 sensors) that also occur in training."""
    seen = {tuple(np.round(row, 6)) for row in train_raw[:, 2:]}
    hits = 0
    for unit in test_units:
        for row in trajectory(test_raw, unit)[-WINDOW:, 2:]:
            hits += tuple(np.round(row, 6)) in seen
    return hits


# -------------------------------------------------------------------------- models
def method_specs(imbalanced):
    """Small grids; the scikit-learn default (or original reference) is listed first.

    GridSearchCV breaks ties by the first candidate in grid order, so exact CV ties
    (common on the separable TEP late snapshots) fall back towards defaults.
    class_weight is only searched when training classes are imbalanced (FD001).
    """
    cw = {"clf__class_weight": [None, "balanced"]} if imbalanced else {}
    plain = lambda est: Pipeline([("clf", est)])  # noqa: E731
    scaled = lambda est: Pipeline([("scale", StandardScaler()), ("clf", est)])  # noqa: E731
    return {
        "RF": (plain(RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1)),
               {"clf__max_features": ["sqrt", 0.3, 1.0], "clf__min_samples_leaf": [1, 3, 10], **cw}),
        "SVM-RBF": (scaled(SVC(kernel="rbf")),
                    {"clf__C": [1.0, 0.1, 10.0, 100.0], "clf__gamma": ["scale", 0.001, 0.01, 0.1], **cw}),
        "HGB": (plain(HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=SEED)),
                {"clf__learning_rate": [0.1, 0.05], "clf__max_leaf_nodes": [31, 7],
                 "clf__l2_regularization": [0.0, 1.0], **cw}),
        "MLP": (scaled(MLPClassifier(max_iter=2000, random_state=SEED)),
                {"clf__hidden_layer_sizes": [(100,), (64,), (128, 64)], "clf__alpha": [1e-4, 1e-2, 1.0]}),
        "kNN": (scaled(KNeighborsClassifier()),
                {"clf__n_neighbors": [5, 1, 3, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
        "LogReg": (scaled(LogisticRegression(max_iter=10000)),
                   {"clf__C": [1.0, 0.001, 0.01, 0.1, 10.0, 100.0], **cw}),
    }


def legacy_specs():
    """The earlier untuned references, reproduced exactly (for continuity checks)."""
    return {"RF-legacy": RandomForestClassifier(n_estimators=300, random_state=SEED),
            "SVM-legacy": make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))}


def jsonable(params):
    return {k.removeprefix("clf__"): list(v) if isinstance(v, tuple) else v for k, v in params.items()}


def metrics(truth, pred, labels):
    correct = int(sum(a == b for a, b in zip(truth, pred)))
    n = len(truth)
    return dict(n=n, correct=correct, accuracy=correct / n,
                macro_f1=float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
                accuracy_wilson95=wilson(correct, n),
                confusion_matrix=confusion_matrix(truth, pred, labels=labels).tolist(), class_order=labels)


# ------------------------------------------------------------------------ experiments
def build_experiments(v3, panels, early_train, fd002_cuts, raw, normalizers):
    """Each experiment = one training set + the panels it is scored on (same IDs)."""
    tep_train, cm_train = v3["tep"]["train"], v3["cmapss"]["train"]
    lab = lambda rows: [r["label"] for r in rows]  # noqa: E731
    grp = lambda rows: [r["group"] for r in rows]  # noqa: E731
    same = lambda names: {p: matrix(panels[p]) for p in names}  # noqa: E731

    # FD001 training windows and FD001/FD002 test windows after regime normalization.
    n001, n002 = normalizers["FD001"], normalizers["FD002"]
    f001_train = n001.transform(raw["train_FD001"])
    f001_test = n001.transform(raw["test_FD001"])
    f002_train = n002.transform(raw["train_FD002"])
    f002_test = n002.transform(raw["test_FD002"])
    norm001 = [cm_features(trajectory(f001_train, r["provenance"]["engine_id"])[
        r["provenance"]["end_cycle"] - WINDOW:r["provenance"]["end_cycle"]]) for r in cm_train]
    fd001_panel_norm = [last_window_features(f001_test, r["provenance"]["engine_id"]) for r in panels["fd001_v3"]]
    fd002_panel_norm = [last_window_features(f002_test, r["provenance"]["engine_id"]) for r in panels["fd002"]]
    fd002_raw_X = [window_features(raw["train_FD002"], c["engine_id"], c["end_index"]) for c in fd002_cuts]
    fd002_norm_X = [window_features(f002_train, c["engine_id"], c["end_index"]) for c in fd002_cuts]

    tep_panels = ["tep_late", "tep_early", "tep_v3"]
    return [
        dict(variant="original_train", domain="TEP", labels=TEP_LABELS,
             X=matrix(tep_train), y=lab(tep_train), groups=grp(tep_train), panels=same(tep_panels),
             training_source="revision-v3 TEP training: 100 run IDs x 7 classes x samples {200,350,500}"),
        dict(variant="original_train", domain="C-MAPSS", labels=CM_LABELS,
             X=matrix(cm_train), y=lab(cm_train), groups=grp(cm_train), panels=same(["fd001_v3", "fd002"]),
             training_source="revision-v3 FD001 training: 100 engines x up to 5 equally spaced 30-cycle windows"),
        dict(variant="tep_phase_aware", domain="TEP", labels=TEP_LABELS,
             X=np.vstack([matrix(tep_train), matrix(early_train)]), y=lab(tep_train) + lab(early_train),
             groups=grp(tep_train) + grp(early_train), panels=same(tep_panels),
             training_source="revision-v3 TEP training late rows + sample-40 rows of the same 100 training runs x 7 classes"),
        dict(variant="fd002_in_domain_raw", domain="C-MAPSS", labels=CM_LABELS,
             X=np.array(fd002_raw_X), y=[c["label"] for c in fd002_cuts], groups=[c["group"] for c in fd002_cuts],
             panels=same(["fd002"]),
             training_source=f"train_FD002: 260 engines x 3 RUL classes x {FD002_CUTS_PER_CLASS} sampled 30-cycle windows"),
        dict(variant="fd002_in_domain_regime_norm", domain="C-MAPSS", labels=CM_LABELS,
             X=np.array(fd002_norm_X), y=[c["label"] for c in fd002_cuts], groups=[c["group"] for c in fd002_cuts],
             panels={"fd002": np.array(fd002_panel_norm)},
             training_source="Same FD002 windows as fd002_in_domain_raw, computed after per-regime normalization"),
        dict(variant="fd001_regime_norm_transfer", domain="C-MAPSS", labels=CM_LABELS,
             X=np.array(norm001), y=lab(cm_train), groups=grp(cm_train),
             panels={"fd001_v3": np.array(fd001_panel_norm), "fd002": np.array(fd002_panel_norm)},
             training_source="revision-v3 FD001 training windows recomputed after FD001 standardization (1 regime)"),
    ]


VARIANTS = {
    "original_train": dict(
        extra_information_vs_llm=False,
        description="Tuned models trained only on the original revision-v3 training features (TEP late snapshots of "
                    "100 training runs; FD001 windows of 100 training engines) and scored on exactly the numeric "
                    "features rendered in the LLM prompts. Like-for-like reference; hyperparameters by grouped CV."),
    "legacy_untuned": dict(
        extra_information_vs_llm=False,
        description="Exact reproduction of the earlier untuned references (RF 300 trees, seed 20260926; "
                    "StandardScaler + RBF SVM C=1, gamma=scale) on the original training features. Continuity check only."),
    "tep_phase_aware": dict(
        extra_information_vs_llm=True,
        description="FAIR-TRANSFER (extra training information): original TEP late training rows plus the sample-40 "
                    "snapshot (same definition as the early test panel) of the SAME 100 training runs for all 7 classes, "
                    "i.e. labelled examples of the early phase. Run-disjoint from every test panel."),
    "fd002_in_domain_raw": dict(
        extra_information_vs_llm=True,
        description="FAIR-TRANSFER (extra training information): labelled FD002 training engines (official train_FD002, "
                    "disjoint partition from the 60 test engines). Same 60-feature 30-cycle summary as the LLM, no "
                    "regime normalization. Balanced sampled cut points (see training_design)."),
    "fd002_in_domain_regime_norm": dict(
        extra_information_vs_llm=True,
        description="FAIR-TRANSFER (extra training information + different features): as fd002_in_domain_raw, but every "
                    "sensor is z-scored per operating regime (KMeans, 6 regimes, fitted on train_FD002 settings) before "
                    "computing the same 30-cycle summaries. Test features are transformed the same way, so they are NOT "
                    "the numbers the LLM saw."),
    "fd001_regime_norm_transfer": dict(
        extra_information_vs_llm=True,
        description="FAIR-TRANSFER (unlabelled target statistics + different features): classifier trained only on FD001 "
                    "labels (FD001 z-scored with its own training statistics, one regime). FD002 test engines are "
                    "normalized with per-regime statistics estimated from UNLABELLED train_FD002 sensor data (no RUL "
                    "used). Also scored on the FD001 panel (same normalization) as an in-domain check."),
}
PANELS = {
    "tep_late": "70 new TEP runs (10/class), late snapshot at a seeded sample in [200,500] (independent validation, LLM condition 'clean').",
    "tep_early": "Same 70 runs at sample 40 (~60 min after fault injection) (independent validation, LLM condition 'early').",
    "tep_v3": "Original revision-v3 TEP held-out panel: 70 runs (10/class), late snapshot.",
    "fd001_v3": "Original revision-v3 FD001 held-out panel: 60 official test engines (20/RUL class), last 30 cycles.",
    "fd002": "60 FD002 official test engines (20/RUL class), last 30 cycles, FD001 feature representation (LLM condition 'fd002').",
}


def run_tuned(exp, cases, results, tuning, panel_records):
    counts = Counter(exp["y"])
    imbalanced = max(counts.values()) / min(counts.values()) > 1.2
    for method, (pipe, grid) in method_specs(imbalanced).items():
        cv = GroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True, error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(exp["X"], exp["y"], groups=exp["groups"])
        res = search.cv_results_
        best = search.best_index_
        clf = search.best_estimator_.named_steps["clf"]
        info = dict(best_params=jsonable(search.best_params_),
                    cv_macro_f1_mean=float(res["mean_test_score"][best]), cv_macro_f1_sd=float(res["std_test_score"][best]),
                    n_candidates=len(res["params"]),
                    n_tied_best=int(np.sum(np.isclose(res["mean_test_score"], res["mean_test_score"][best]))),
                    class_weight_searched=imbalanced)
        if isinstance(clf, MLPClassifier):
            info["mlp_iterations"] = int(clf.n_iter_)
            info["mlp_converged"] = bool(clf.n_iter_ < clf.max_iter)
        if method in SEED_SENSITIVE:
            info["seed_sensitivity"] = seed_sensitivity(search.best_estimator_, exp, panel_records)
        tuning.setdefault(exp["variant"], {}).setdefault(exp["domain"], {})[method] = info
        print(f"  {exp['variant']:<28} {exp['domain']:<8} {method:<8} cv={info['cv_macro_f1_mean']:.3f} "
              f"ties={info['n_tied_best']} {info['best_params']}", flush=True)
        score_panels(exp, method, search, cases, results, panel_records)


# RF and MLP depend on random_state. HGB is deterministic here (no early stopping,
# fewer rows than its binning subsample), and SVM/kNN/LogReg are deterministic.
SEED_SENSITIVE = ["RF", "MLP"]
EXTRA_SEEDS = [SEED + k for k in range(1, 6)]


def seed_sensitivity(best, exp, panel_records):
    """Refit the CV-chosen configuration with five further seeds (descriptive only).

    The primary reported prediction always uses SEED; these refits only show how
    much panel accuracy moves with the random seed at fixed hyperparameters.
    """
    out = {}
    for seed in EXTRA_SEEDS:
        model = clone(best).set_params(clf__random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(exp["X"], exp["y"])
        for panel, X in exp["panels"].items():
            truth = [r["label"] for r in panel_records[panel]]
            out.setdefault(panel, dict(seeds=EXTRA_SEEDS, accuracy=[]))["accuracy"].append(
                float(np.mean(np.array(model.predict(X)) == np.array(truth))))
    for v in out.values():
        v.update(min=min(v["accuracy"]), max=max(v["accuracy"]), mean=float(np.mean(v["accuracy"])))
    return out


def run_legacy(exp, cases, results, panel_records):
    for method, model in legacy_specs().items():
        model.fit(exp["X"], exp["y"])
        score_panels(dict(exp, variant="legacy_untuned"), method, model, cases, results, panel_records)


def score_panels(exp, method, model, cases, results, panel_records):
    for panel, X in exp["panels"].items():
        recs = panel_records[panel]
        pred = model.predict(X).tolist()
        truth = [r["label"] for r in recs]
        results.append(dict(variant=exp["variant"], domain=exp["domain"], method=method, panel=panel,
                            **metrics(truth, pred, exp["labels"])))
        for r, p in zip(recs, pred):
            cases.append(dict(method=method, variant=exp["variant"], panel=panel, id=r["id"], true_label=r["label"],
                              predicted=p, correct=int(p == r["label"]), source_group=r["group"],
                              llm_case_file=LLM_REFERENCE[panel][0], llm_selector=LLM_REFERENCE[panel][1]))


# ---------------------------------------------------------------------- sanity checks
def leakage_checks(v3, panels, experiments, early_train, fd002_cuts, raw):
    tep_train_groups = {r["group"] for r in v3["tep"]["train"]} | {r["group"] for r in early_train}
    tep_test_groups = {r["group"] for p in ["tep_late", "tep_early", "tep_v3"] for r in panels[p]}
    require(not tep_train_groups & tep_test_groups, "TEP run ID overlap between training and test")
    iv_runs = {r["provenance"]["simulation_run"] for p in ["tep_late", "tep_early"] for r in panels[p]}
    train_runs = {r["provenance"]["simulation_run"] for r in v3["tep"]["train"]}
    require(not iv_runs & train_runs and {r["provenance"]["simulation_run"] for r in early_train} == train_runs,
            "TEP early training snapshots must come only from the 100 training runs")

    fd001_train_groups = {r["group"] for r in v3["cmapss"]["train"]}
    require(all(g.startswith("FD001/train/") for g in fd001_train_groups), "FD001 training partition")
    require(all(r["group"].startswith("FD001/test/") for r in panels["fd001_v3"]), "FD001 test partition")
    require(all(r["group"].startswith("FD002/test/") for r in panels["fd002"]), "FD002 test partition")
    require(all(c["group"].startswith("FD002/train/") for c in fd002_cuts), "FD002 training partition")

    # Recompute the FD002 test features and labels from raw files: identical to the LLM inputs.
    rul = np.loadtxt(CM_RAW / "RUL_FD002.txt", dtype=int)
    for r in panels["fd002"]:
        unit = r["provenance"]["engine_id"]
        require(last_window_features(raw["test_FD002"], unit) == r["features"], "FD002 feature recomputation")
        require(cm_label(int(rul[unit - 1])) == r["label"], "FD002 label recomputation")
    # Recompute FD001 training windows from raw data (they seed the normalized variant).
    for r in v3["cmapss"]["train"]:
        p = r["provenance"]
        traj = trajectory(raw["train_FD001"], p["engine_id"])
        require(cm_features(traj[p["end_cycle"] - WINDOW:p["end_cycle"]]) == r["features"], "FD001 train window")

    # Raw-row level: no test-window row (settings + 21 sensors) occurs in a training partition.
    overlap = dict(
        fd002_test_windows_in_train_FD002=row_overlap(raw["test_FD002"], [r["provenance"]["engine_id"] for r in panels["fd002"]], raw["train_FD002"]),
        fd001_test_windows_in_train_FD001=row_overlap(raw["test_FD001"], [r["provenance"]["engine_id"] for r in panels["fd001_v3"]], raw["train_FD001"]))
    require(not any(overlap.values()), f"Raw row overlap {overlap}")

    # Exact feature-vector duplicates between any training set and the panels it scores.
    duplicates = {}
    for exp in experiments:
        train = {tuple(x) for x in np.asarray(exp["X"]).tolist()}
        for panel, X in exp["panels"].items():
            n = sum(tuple(x) in train for x in np.asarray(X).tolist())
            require(n == 0, f"{exp['variant']}/{panel}: test feature vector identical to a training row")
            duplicates[f"{exp['variant']}/{exp['domain']}/{panel}"] = n
    return dict(
        tep_training_run_ids=len(train_runs), tep_test_run_ids=len({r['provenance']['simulation_run'] for p in ['tep_late', 'tep_early', 'tep_v3'] for r in panels[p]}),
        tep_train_test_run_overlap=0,
        fd002_train_engines=len({c['engine_id'] for c in fd002_cuts}), fd002_test_engines=len(panels["fd002"]),
        fd002_partition_note="train_FD002 and test_FD002 are distinct official partitions; numeric engine IDs are namespaced by partition and denote different engines. Verified additionally by zero raw-row overlap.",
        raw_row_overlap=overlap, exact_feature_duplicates=duplicates,
        fd002_test_features_recomputed_from_raw=True, fd001_training_windows_recomputed_from_raw=True,
        test_labels_used_only_for_scoring=True)


def legacy_reproduction(results, panels):
    """The legacy references must reproduce the stored revision-v3 / independent scores exactly."""
    stored_v3 = read_json(V3_BASELINE_PREDICTIONS)
    stored_iv = {(b["model"], b["condition"]): b for b in read_json(IV_SCORES)["baselines"]}
    report = {}
    for row in results:
        if row["variant"] != "legacy_untuned":
            continue
        old = row["method"].split("-")[0]
        if row["panel"] in ["tep_v3", "fd001_v3"]:
            ds = "tep" if row["panel"] == "tep_v3" else "cmapss"
            truth = {r["id"]: r["label"] for r in panels[row["panel"]]}
            stored = sum(p["predicted"] == truth[p["id"]] for p in stored_v3[ds][old])
        else:
            condition = {"tep_late": "clean", "tep_early": "early", "fd002": "fd002"}[row["panel"]]
            stored = stored_iv[(old, condition)]["correct"]
        require(stored == row["correct"], f"Legacy {row['method']} on {row['panel']} not reproduced")
        report[f"{row['method']}/{row['panel']}"] = dict(correct=row["correct"], stored_correct=int(stored))
    return report


# --------------------------------------------------------------------------- outputs
def llm_reference_table():
    """LLM accuracy per repeat from the existing case files (context only; not recomputed)."""
    iv = pd.read_csv(IV_CASES)
    iv = iv[iv["condition"].isin(["clean", "early", "fd002"])]
    table = iv.groupby(["condition", "model", "repeat"])["correct"].mean().unstack()
    v3 = pd.read_csv(V3_CASES)
    v3 = v3.groupby(["dataset", "model", "repeat"]).apply(
        lambda g: g["correct"].mean() if g["available"].all() else np.nan, include_groups=False).unstack()
    return table, v3


def write_summary(results, tuning, training_design, sanity, class_balance):
    order = ["tep_late", "tep_early", "tep_v3", "fd001_v3", "fd002"]
    methods = ["RF", "SVM-RBF", "HGB", "MLP", "kNN", "LogReg", "RF-legacy", "SVM-legacy"]
    variants = ["original_train", "legacy_untuned", "tep_phase_aware", "fd002_in_domain_raw",
                "fd002_in_domain_regime_norm", "fd001_regime_norm_transfer"]
    cell = {(r["variant"], r["method"], r["panel"]): r for r in results}
    lines = ["# Supervised references v2 (tuned, grouped CV)", "",
             "Generated by `src/analysis/baselines_v2.py`. Cells: accuracy / macro-F1 (%), Wilson 95% CI for accuracy in "
             "`scores.json`. Hyperparameters chosen by 5-fold GroupKFold CV (groups = training run or engine), "
             "macro-F1, training data only. `—` = panel not applicable to that variant.", "",
             "## Like-for-like references (same information and features as the LLM)", ""]

    def table(vs):
        out = ["| variant | method | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
        for v in vs:
            for m in methods:
                if not any((v, m, p) in cell for p in order):
                    continue
                vals = []
                for p in order:
                    r = cell.get((v, m, p))
                    vals.append("—" if r is None else f"{100 * r['accuracy']:.1f} / {100 * r['macro_f1']:.1f}")
                out.append(f"| {v} | {m} | " + " | ".join(vals) + " |")
        return out
    lines += table(["original_train", "legacy_untuned"])
    lines += ["", "## Fair-transfer variants (receive information the LLM does NOT get)", ""]
    lines += table(variants[2:])
    lines += ["", "## Chosen hyperparameters (grouped-CV macro-F1 on training data)", "",
              "| variant | domain | method | CV macro-F1 | tied best | chosen |", "|---|---|---|---|---|---|"]
    for v, domains in tuning.items():
        for d, ms in domains.items():
            for m, info in ms.items():
                lines.append(f"| {v} | {d} | {m} | {100 * info['cv_macro_f1_mean']:.1f} ± {100 * info['cv_macro_f1_sd']:.1f} "
                             f"| {info['n_tied_best']}/{info['n_candidates']} | `{info['best_params']}` |")
    lines += ["", "## Seed sensitivity (RF, MLP: chosen configuration refit with 5 further seeds; accuracy min–max %)", ""]
    for v, domains in tuning.items():
        for d, ms in domains.items():
            for m in SEED_SENSITIVE:
                sens = ms[m]["seed_sensitivity"]
                lines.append(f"- {v}/{d}/{m}: " + "; ".join(
                    f"{p} {100 * s['min']:.1f}–{100 * s['max']:.1f}" for p, s in sens.items()))
    lines += ["", "## Variants", ""]
    lines += [f"- **{k}**: {v['description']}" for k, v in VARIANTS.items()]
    lines += ["", "## Panels", ""] + [f"- **{k}** (n={class_balance['panels'][k]['n']}, balanced "
                                        f"{class_balance['panels'][k]['class_counts']}): {v}" for k, v in PANELS.items()]
    lines += ["", "## Training sets and class balance", ""]
    for key, v in class_balance["training"].items():
        lines.append(f"- {key}: {v['n_records']} records, {v['n_groups']} groups, classes {v['class_counts']}")
    lines += ["", f"FD002 training design: {training_design['fd002']['sampling']}", "",
              "## LLM reference accuracy (from existing case files, per repeat 1/2/3)", ""]
    iv, v3 = llm_reference_table()
    for (condition, model), row in iv.iterrows():
        lines.append(f"- independent `{condition}`, {model}: " + " / ".join(f"{100 * x:.1f}" for x in row.values))
    for (dataset, model), row in v3.iterrows():
        lines.append(f"- revision-v3 `{dataset}` clean, {model}: " + " / ".join("N/A" if np.isnan(x) else f"{100 * x:.1f}" for x in row.values))
    lines += ["", "## Sanity checks", "",
              f"- TEP: {sanity['tep_training_run_ids']} training run IDs vs {sanity['tep_test_run_ids']} test run IDs; overlap 0. "
              "Early training snapshots come only from training runs; 2240 frozen TEP rows re-extracted from raw RData matched exactly.",
              f"- FD002: {sanity['fd002_train_engines']} train_FD002 engines vs {sanity['fd002_test_engines']} test_FD002 engines; "
              f"raw-row overlap of test windows with training partition: {sanity['raw_row_overlap']}. Test features/labels recomputed from raw files.",
              "- No test feature vector is identical to any training row in any variant. Panel IDs equal the LLM case-file IDs.",
              "- Legacy RF/SVM reproduce the stored earlier results exactly on all five panels.",
              "", "## Caveats", "",
              "- Exact CV ties (reported as `tied best`) are broken by grid order, which lists defaults first.",
              "- Reported predictions use one fixed seed. RF and MLP accuracy move by a few cases across seeds (see the seed "
              "sensitivity section); with n=60-70, differences of 1-3 cases between methods are within this noise.",
              "- Fair-transfer variants are upper-bound style references, not equal-information comparisons with the LLM. "
              "tep_phase_aware is trained at exactly the test sample index (40); a deployed model would not know the "
              "time since fault onset. Regime-normalized variants also use features the LLM never sees.",
              "- FD002 in-domain training cut points are sampled uniformly within each RUL bin; the test panel's RUL "
              f"distribution within bins follows the official truncation (training ranges: {training_design['fd002']['rul_range_by_class']}).",
              "- The FD001 original panel is reconstructed exactly from the frozen revision-v3 datasets.json "
              "(same IDs, features, labels); no approximation was needed."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v3 = read_json(V3_DATA)["datasets"]
    answers = read_json(IV_ANSWERS)
    panels = load_panels(v3, answers)
    id_check = check_llm_ids(panels)

    early_train, tep_audit = tep_training_snapshots(v3, panels)
    raw = {name: load_cmapss(name) for name in ["train_FD001", "test_FD001", "train_FD002", "test_FD002"]}
    fd002_cuts = fd002_training_cutpoints(raw["train_FD002"])
    normalizers = {"FD001": RegimeNormalizer(1).fit(raw["train_FD001"]),
                   "FD002": RegimeNormalizer(N_REGIMES).fit(raw["train_FD002"])}
    fd002_test_units = [r["provenance"]["engine_id"] for r in panels["fd002"]]
    test_rows = np.vstack([trajectory(raw["test_FD002"], u) for u in fd002_test_units])
    regime_audit = dict(train_regime_counts=normalizers["FD002"].train_regime_counts,
                        max_train_distance_to_centroid_std_units=normalizers["FD002"].max_centroid_distance(raw["train_FD002"]),
                        max_test_distance_to_centroid_std_units=normalizers["FD002"].max_centroid_distance(test_rows),
                        centroids_in_setting_units=np.round(normalizers["FD002"].scaler.inverse_transform(
                            normalizers["FD002"].kmeans.cluster_centers_), 4).tolist())
    require(regime_audit["max_test_distance_to_centroid_std_units"] < 0.1, "FD002 test rows not near a training regime")

    experiments = build_experiments(v3, panels, early_train, fd002_cuts, raw, normalizers)
    sanity = leakage_checks(v3, panels, experiments, early_train, fd002_cuts, raw)
    sanity.update(tep_extraction=tep_audit, llm_id_match=id_check)

    class_balance = dict(
        panels={p: dict(n=len(rs), class_counts=dict(Counter(r["label"] for r in rs))) for p, rs in panels.items()},
        training={f"{e['variant']}/{e['domain']}": dict(n_records=len(e["y"]), n_groups=len(set(e["groups"])),
                                                         class_counts=dict(Counter(e["y"])))
                  for e in experiments})
    for key, v in class_balance["training"].items():
        print("train", key, v, flush=True)

    cases, results, tuning = [], [], {}
    panel_records = panels
    for exp in experiments:
        if exp["variant"] == "original_train":
            run_legacy(exp, cases, results, panel_records)
        run_tuned(exp, cases, results, tuning, panel_records)
    sanity["legacy_reproduction"] = legacy_reproduction(results, panels)

    training_design = dict(
        tep_phase_aware=dict(sampling=f"All 7 classes x the 100 revision-v3 training runs at sample {EARLY_SAMPLE} "
                                      "(fault injected after sample 20; 3-minute sampling), 4-decimal rounding as in the prompts. "
                                      "Appended to the 2100 late training rows; CV groups = run ID shared across classes.",
                             file="tep_early_training_snapshots.csv"),
        fd002=dict(sampling=" ".join(fd002_training_cutpoints.__doc__.split()),
                   cuts_per_engine_and_class=FD002_CUTS_PER_CLASS, seed=SEED + 302, file="fd002_training_cutpoints.csv",
                   rul_range_by_class={lab: [min(c["rul_label_only"] for c in fd002_cuts if c["label"] == lab),
                                             max(c["rul_label_only"] for c in fd002_cuts if c["label"] == lab)] for lab in CM_LABELS}),
        regime_normalization=dict(method=" ".join(RegimeNormalizer.__doc__.split()),
                                  kmeans=dict(n_clusters=N_REGIMES, n_init=10, random_state=SEED, fitted_on="train_FD002 operating settings only"),
                                  audit=regime_audit, sensors=CM_SENSORS,
                                  fd001="Single regime: global per-sensor standardization with train_FD001 statistics."))

    with (OUT / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    pd.DataFrame([dict(**r["provenance"], label=r["label"], group=r["group"],
                       **dict(zip(TEP_COLS, r["features"]))) for r in early_train]).to_csv(
        OUT / "tep_early_training_snapshots.csv", index=False)
    pd.DataFrame(fd002_cuts).to_csv(OUT / "fd002_training_cutpoints.csv", index=False)

    grids = {m: jsonable({k: v for k, v in g.items()}) for m, (_, g) in method_specs(True).items()}
    scores = dict(
        schema="baselines-v2", created_at=datetime.now(timezone.utc).isoformat(), seed=SEED,
        script="src/analysis/baselines_v2.py", script_sha256=digest(Path(__file__)),
        inputs_sha256={str(p.relative_to(ROOT)): digest(p) for p in [V3_DATA, IV_ANSWERS, IV_CASES, V3_CASES,
                       CM_RAW / "train_FD002.txt", CM_RAW / "test_FD002.txt", CM_RAW / "RUL_FD002.txt",
                       CM_RAW / "train_FD001.txt", CM_RAW / "test_FD001.txt"]},
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scikit_learn=sklearn.__version__, pyreadr=pyreadr.__version__),
        policy="Training data only; hyperparameters by GroupKFold (5 folds, shuffled groups, seed) on training run/engine "
               "IDs, scoring macro-F1, refit on all training rows; each panel scored once. Test labels only for scoring.",
        variants=VARIANTS, panels={p: dict(description=d, llm_case_file=LLM_REFERENCE[p][0],
                                            llm_selector=LLM_REFERENCE[p][1], **class_balance["panels"][p]) for p, d in PANELS.items()},
        methods=dict(
            pipelines={"RF": "RandomForestClassifier(n_estimators=500)", "SVM-RBF": "StandardScaler + SVC(rbf)",
                       "HGB": "HistGradientBoostingClassifier(max_iter=200, early_stopping=False) (gradient-boosting reference in place of XGBoost)",
                       "MLP": "StandardScaler + MLPClassifier(adam, max_iter=2000)", "kNN": "StandardScaler + KNeighborsClassifier",
                       "LogReg": "StandardScaler + multinomial LogisticRegression(lbfgs, L2, max_iter=10000)",
                       "RF-legacy": "RandomForestClassifier(n_estimators=300, seed 20260926), untuned",
                       "SVM-legacy": "StandardScaler + SVC(C=1, gamma='scale'), untuned"},
            grids=grids, grid_note="class_weight is searched only when training classes are imbalanced (max/min > 1.2: FD001 only). "
                                   "Ties broken by first candidate in grid order (defaults first)."),
        training_design=training_design, class_balance=class_balance, tuning=tuning, results=results, sanity_checks=sanity)
    write_json(OUT / "scores.json", scores)
    write_summary(results, tuning, training_design, sanity, class_balance)
    for r in results:
        print(f"{r['variant']:<28} {r['method']:<10} {r['panel']:<9} acc={100 * r['accuracy']:5.1f} "
              f"F1={100 * r['macro_f1']:5.1f}", flush=True)


if __name__ == "__main__":
    main()
