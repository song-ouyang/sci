"""Inferential statistics for the frozen LLM outputs (offline, no model calls).

Three complementary analyses are computed for every contrast between a
reference input (left) and a changed input (right):

* exact McNemar tests on the prespecified primary repeat (Holm-adjusted within
  each family of contrasts);
* a pigeonhole (crossed case x call) bootstrap interval for the accuracy
  difference pooled over all repeats, which propagates both case sampling and
  call-to-call variation (Owen, 2007);
* an exact call-level randomization test in which whole calls, not cases, are
  relabelled. This respects the dependence between labels returned in one call.
  With k calls per side and no blocking there are C(2k, k) relabellings, so the
  smallest attainable two-sided p-value is bounded below by the design.

Label disagreement is summarized as the excess of between-condition over
within-condition disagreement, E = B - W, per case. Repeatability of identical
inputs is reported as Fleiss' kappa over the repeated calls.
"""
from collections import Counter, defaultdict
import csv
import itertools
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/statistics_v2"
SEED = 20260928
N_BOOT = 10000


# ---------------------------------------------------------------- utilities
def mcnemar(left_correct, right_correct):
    """Exact two-sided McNemar test on paired correctness vectors."""
    b = int(np.sum((left_correct == 1) & (right_correct == 0)))
    c = int(np.sum((left_correct == 0) & (right_correct == 1)))
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def pigeonhole_ci(left, right, rng, n_boot=N_BOOT):
    """Percentile interval for mean(left) - mean(right); cases and calls resampled independently.

    ``left`` and ``right`` are case x call correctness matrices with shared case rows.
    """
    n, kl, kr = left.shape[0], left.shape[1], right.shape[1]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n, n)
        cl, cr = rng.integers(0, kl, kl), rng.integers(0, kr, kr)
        stats[b] = left[np.ix_(rows, cl)].mean() - right[np.ix_(rows, cr)].mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def case_bootstrap_ci(values, rng, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def matched_block_ci(left, right, blocks, rng, n_boot=N_BOOT):
    """Exploratory case x matched-block percentile interval.

    Resample the shared case rows, then the execution blocks jointly on both
    sides. Retain all observed calls inside each selected block. This is a
    post-hoc sensitivity analysis conditional on these cases and few blocks,
    not a prespecified population-coverage guarantee.
    """
    if sum(a for a, _ in blocks) != left.shape[1] or sum(b for _, b in blocks) != right.shape[1]:
        raise ValueError("Block sizes do not match call matrices")
    if len(blocks) < 2:
        raise ValueError("Matched-block sensitivity needs at least two blocks")
    ls, rs, li, ri = [], [], 0, 0
    for a, b in blocks:
        ls.append(np.arange(li, li + a)); rs.append(np.arange(ri, ri + b))
        li += a; ri += b
    stats = np.empty(n_boot)
    for index in range(n_boot):
        rows = rng.integers(0, left.shape[0], left.shape[0])
        chosen = rng.integers(0, len(blocks), len(blocks))
        cl = np.concatenate([ls[b] for b in chosen])
        cr = np.concatenate([rs[b] for b in chosen])
        stats[index] = left[np.ix_(rows, cl)].mean() - right[np.ix_(rows, cr)].mean()
    return tuple(float(v) for v in np.percentile(stats, [2.5, 97.5]))


def abba_design_masks(n_blocks=4, n_abba=2):
    """Frozen design: shuffle n_abba ABBA and remaining BAAB sequences."""
    for selected in itertools.combinations(range(n_blocks), n_abba):
        yield np.array([flag for b in range(n_blocks)
                        for flag in ((True, False, False, True) if b in selected
                                     else (False, True, True, False))])


def disagreement(labels_left, labels_right):
    """Per-case within (W) and between (B) disagreement; inputs are case x call label arrays."""
    def within(labels):
        pairs = list(itertools.combinations(range(labels.shape[1]), 2))
        return np.mean([labels[:, i] != labels[:, j] for i, j in pairs], axis=0) if pairs else np.zeros(labels.shape[0])
    w = (within(labels_left) + within(labels_right)) / 2
    b = np.mean([labels_left[:, i] != labels_right[:, j]
                 for i in range(labels_left.shape[1]) for j in range(labels_right.shape[1])], axis=0)
    return w, b


def relabellings(calls_per_block):
    """Yield boolean masks (True = left) for all allowed relabellings of calls.

    ``calls_per_block`` is a list of (n_left, n_right) tuples; calls are only
    exchanged within their own execution block.
    """
    per_block = []
    for n_left, n_right in calls_per_block:
        size = n_left + n_right
        per_block.append([tuple(i in chosen for i in range(size))
                          for chosen in itertools.combinations(range(size), n_left)])
    for combo in itertools.product(*per_block):
        yield np.array([flag for block in combo for flag in block])


def call_randomization(correct, labels, is_left, blocks=None, masks=None):
    """Exact call-level randomization p-values for accuracy difference and excess disagreement.

    ``correct``/``labels`` are case x call arrays over all calls of both
    conditions; ``is_left`` marks the observed assignment; ``blocks`` lists the
    (n_left, n_right) layout in the same column order.
    """
    def stats(mask):
        acc = correct[:, mask].mean() - correct[:, ~mask].mean()
        w, b = disagreement(labels[:, mask], labels[:, ~mask])
        return acc, float(np.mean(b - w))
    observed = stats(is_left)
    null = np.array([stats(mask) for mask in (relabellings(blocks) if masks is None else masks)])
    eps = 1e-12
    p_acc = float(np.mean(np.abs(null[:, 0]) >= abs(observed[0]) - eps))
    p_exc = float(np.mean(null[:, 1] >= observed[1] - eps))  # one-sided: excess disagreement
    # Smallest two-sided p-value this relabelling set can produce for the accuracy statistic.
    extreme = np.max(np.abs(null[:, 0]))
    p_min = float(np.mean(np.abs(null[:, 0]) >= extreme - eps)) if extreme > 0 else 1.0
    return p_acc, p_exc, len(null), p_min


def fleiss_kappa(labels):
    """Fleiss' kappa for a case x rater label array."""
    labels = np.asarray(labels)
    n, k = labels.shape
    categories = sorted(set(labels.ravel()))
    counts = np.array([[np.sum(row == c) for c in categories] for row in labels], dtype=float)
    p_i = (np.sum(counts ** 2, axis=1) - k) / (k * (k - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n * k)
    p_e = np.sum(p_j ** 2)
    return 1.0 if np.isclose(p_e, 1.0) else float((p_bar - p_e) / (1 - p_e))


def contrast(name, family, left, right, truth, ids, blocks, primary, rng):
    """Full analysis of one contrast.

    ``left``/``right`` are lists of {id: label} dicts (one per call, in block order);
    ``primary`` is the index of the prespecified primary call pair.
    """
    ids = [i for i in ids if all(i in call for call in left + right)]
    lab_l = np.array([[call[i] for call in left] for i in ids])
    lab_r = np.array([[call[i] for call in right] for i in ids])
    y = np.array([truth[i] for i in ids])
    cor_l, cor_r = (lab_l == y[:, None]).astype(int), (lab_r == y[:, None]).astype(int)
    b, c, p_mc = mcnemar(cor_l[:, primary], cor_r[:, primary])
    lo, hi = pigeonhole_ci(cor_l, cor_r, rng)
    crossed_lo, crossed_hi = lo, hi
    method = "crossed cases x independent call columns"
    paired_lo = paired_hi = None
    if len(blocks) > 1:
        # Independent seed: adding this sensitivity does not perturb the legacy
        # crossed or case-only intervals for any later contrast.
        seed = int.from_bytes(hashlib.sha256(f"{SEED}:{family}:{name}".encode()).digest()[:8], "big")
        paired_lo, paired_hi = matched_block_ci(cor_l, cor_r, blocks, np.random.default_rng(seed))
        lo, hi = paired_lo, paired_hi
        method = "post-hoc case x matched-block sensitivity"
    w, bb = disagreement(lab_l, lab_r)
    e_lo, e_hi = case_bootstrap_ci(bb - w, rng)
    # Column order for the randomization test follows ``blocks``.
    order, li, ri, mask = [], 0, 0, []
    for n_left, n_right in blocks:
        order += [("L", li + j) for j in range(n_left)] + [("R", ri + j) for j in range(n_right)]
        mask += [True] * n_left + [False] * n_right
        li, ri = li + n_left, ri + n_right
    stacked_lab = np.column_stack([lab_l[:, j] if s == "L" else lab_r[:, j] for s, j in order])
    stacked_cor = np.column_stack([cor_l[:, j] if s == "L" else cor_r[:, j] for s, j in order])
    p_acc, p_exc, n_relabel, p_min = call_randomization(stacked_cor, stacked_lab, np.array(mask), blocks)
    stable_diff = int(np.sum([(len(set(lab_l[k])) == 1) and (len(set(lab_r[k])) == 1) and lab_l[k, 0] != lab_r[k, 0]
                              for k in range(len(ids))]))
    return {"family": family, "contrast": name, "n_cases": len(ids), "calls_left": lab_l.shape[1],
            "calls_right": lab_r.shape[1],
            "acc_left": float(cor_l.mean()), "acc_right": float(cor_r.mean()),
            "acc_diff": float(cor_l.mean() - cor_r.mean()), "acc_diff_ci_low": lo, "acc_diff_ci_high": hi,
            "acc_diff_ci_method": method,
            "crossed_ci_low": crossed_lo, "crossed_ci_high": crossed_hi,
            "matched_block_ci_low": paired_lo, "matched_block_ci_high": paired_hi,
            "execution_blocks": len(blocks),
            "primary_mcnemar_b": b, "primary_mcnemar_c": c, "primary_mcnemar_p": p_mc,
            "within_disagreement": float(w.mean()), "between_disagreement": float(bb.mean()),
            "excess_disagreement": float((bb - w).mean()), "excess_ci_low": e_lo, "excess_ci_high": e_hi,
            "call_perm_p_acc": p_acc, "call_perm_p_excess": p_exc, "call_relabellings": n_relabel,
            "call_test_method": "within-block exchangeability sensitivity" if len(blocks) > 1 else "whole-call exchangeability test",
            "min_attainable_p_acc": p_min,
            "stable_different_cases": stable_diff,
            "kappa_left": fleiss_kappa(lab_l) if lab_l.shape[1] > 1 else None,
            "kappa_right": fleiss_kappa(lab_r) if lab_r.shape[1] > 1 else None}


def finish_family(rows):
    adjusted = holm(np.array([r["primary_mcnemar_p"] for r in rows]))
    for row, p in zip(rows, adjusted):
        row["primary_mcnemar_p_holm"] = float(p)
    return rows


# ---------------------------------------------------------------- datasets
def tep_independent(rng):
    table = defaultdict(dict)
    truth = {}
    with open(ROOT / "results/independent_validation/analysis/cases.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            table[(r["model"], r["condition"], int(r["repeat"]))][r["id"]] = r["predicted"]
            truth[r["id"]] = r["true_label"]
    rows, kappas = [], []
    mapping = [("rotation", "Cyclic rotation"), ("random_1", "Random derangement 1"),
               ("random_2", "Random derangement 2"), ("random_3", "Random derangement 3"),
               ("same_unit_swap", "Same-unit swap"), ("pair_shuffle", "Correct-pair reorder")]
    context = [("a_front", "a_back", "A: front vs back"), ("b_front", "b_back", "B: front vs back"),
               ("a_front", "b_front", "Front: A vs B"), ("a_back", "b_back", "Back: A vs B"),
               ("focal_only", "clean", "Focal only vs within 70")]
    for model in ["gpt-5.6-sol", "gpt-6-astra"]:
        calls = lambda cond: [table[(model, cond, k)] for k in (1, 2, 3)]
        main_ids = sorted(table[(model, "clean", 1)])
        focal_ids = sorted(table[(model, "focal_only", 1)])
        family = []
        for cond, label in mapping:
            row = contrast(label, f"{model}/mapping", calls("clean"), calls(cond), truth, main_ids, [(3, 3)], 0, rng)
            family.append({"model": model, **row})
        rows += finish_family(family)
        family = []
        for left, right, label in context:
            row = contrast(label, f"{model}/context", calls(left), calls(right), truth, focal_ids, [(3, 3)], 0, rng)
            family.append({"model": model, **row})
        rows += finish_family(family)
        for cond in ["clean", "rotation", "random_1", "random_2", "random_3", "same_unit_swap", "pair_shuffle",
                     "early", "fd002"]:
            ids = sorted(table[(model, cond, 1)])
            lab = np.array([[table[(model, cond, k)][i] for k in (1, 2, 3)] for i in ids])
            kappas.append({"model": model, "condition": cond, "n_cases": len(ids), "calls": 3,
                           "fleiss_kappa": fleiss_kappa(lab),
                           "all_calls_agree": float(np.mean([len(set(x)) == 1 for x in lab]))})
        for cond in ["focal_only", "a_front", "a_back", "b_front", "b_back"]:
            lab = np.array([[table[(model, cond, k)][i] for k in (1, 2, 3)] for i in focal_ids])
            kappas.append({"model": model, "condition": cond, "n_cases": len(focal_ids), "calls": 3,
                           "fleiss_kappa": fleiss_kappa(lab),
                           "all_calls_agree": float(np.mean([len(set(x)) == 1 for x in lab]))})
    return rows, kappas


def load_predictions(folder, name):
    path = folder / f"{name}_results.json"
    if not path.exists():
        return None
    return {r["id"]: r["predicted"] for r in json.loads(path.read_text(encoding="utf-8"))}


def abba(rng):
    base = ROOT / "results/context_blocked_v1"
    folder = base / "predictions/gpt-5.6-sol"
    records = json.loads((base / "answers/records.json").read_text(encoding="utf-8"))
    truth = {r["id"]: r["label"] for r in records}
    focal = sorted(r["id"] for r in records if r["focal"])
    schedule = json.loads((base / "schedule.json").read_text(encoding="utf-8"))["blocks"]
    front, back, blocks = [], [], []
    for block in schedule:
        f = [load_predictions(folder, n) for n in block if n.endswith("_a")]
        r = [load_predictions(folder, n) for n in block if n.endswith("_b")]
        if None in f or None in r:
            continue
        front += f
        back += r
        blocks.append((len(f), len(r)))
    row = contrast("Focal first vs focal last (ABBA/BAAB)", "sol/abba", front, back, truth, focal,
                   blocks, 0, rng)
    if len(blocks) != 4:
        raise ValueError("Frozen ABBA design requires all four complete blocks")
    # The experiment randomized two ABBA and two BAAB sequences, not arbitrary
    # two-of-four call labels independently within each block.
    names = [name for block in schedule for name in block]
    lab = np.array([[load_predictions(folder, name)[i] for name in names] for i in focal])
    cor = (lab == np.array([truth[i] for i in focal])[:, None]).astype(int)
    observed = np.array([name.endswith("_a") for name in names])
    p_acc, p_exc, n_design, p_min = call_randomization(cor, lab, observed, masks=abba_design_masks())
    row.update({"exchangeability_p_acc_sensitivity": row["call_perm_p_acc"],
                "exchangeability_p_excess_sensitivity": row["call_perm_p_excess"],
                "exchangeability_relabellings_sensitivity": row["call_relabellings"],
                "call_perm_p_acc": p_acc, "call_perm_p_excess": p_exc,
                "call_relabellings": n_design, "min_attainable_p_acc": p_min,
                "call_test_method": "exact frozen design: two ABBA and two BAAB across four blocks"})
    # Per-block disagreement for the table in the manuscript.
    per_block = []
    for k in range(len(blocks)):
        f, r = front[2 * k:2 * k + 2], back[2 * k:2 * k + 2]
        per_block.append({"block": k + 1,
                          "within_front": sum(f[0][i] != f[1][i] for i in focal),
                          "within_back": sum(r[0][i] != r[1][i] for i in focal),
                          "cross": [sum(a[i] != b[i] for i in focal) for a in f for b in r]})
    return finish_family([{"model": "gpt-5.6-sol", **row}]), per_block


def hydraulic(rng):
    v1 = ROOT / "results/hydraulic_validation_v1"
    comp = ROOT / "results/hydraulic_validation_v1_completion"
    panels = json.loads((v1 / "answers/records.json").read_text(encoding="utf-8"))["panels"]
    truth = {r["id"]: r["true_label"] for p in panels.values() for r in p}
    focal = sorted(r["id"] for r in panels["focal"])
    sources = {1: v1, 2: v1, 3: comp, 4: comp, 5: comp}

    def call(cond, rep):
        return load_predictions(sources[rep] / "predictions/gpt-5.6-sol", f"{cond}_r{rep}")

    available = [rep for rep in range(1, 6)
                 if all(call(c, rep) is not None for c in
                        ["clean", "clean_repeat", "pair_reverse", "descriptor_rotation", "a_front",
                         "a_front_repeat", "a_back", "b_front", "b_back"])]
    # Byte-identical duplicates are pooled as the reference where they exist, which
    # enlarges the within-block randomization set (C(3,1) per block instead of 2).
    contrasts = [(["clean", "clean_repeat"], ["descriptor_rotation"], "Descriptor rotation"),
                 (["clean", "clean_repeat"], ["pair_reverse"], "Intact-pair reversal"),
                 (["clean"], ["clean_repeat"], "Identical duplicate (focal)"),
                 (["a_front"], ["a_front_repeat"], "Identical duplicate (focal+A)"),
                 (["a_front", "a_front_repeat"], ["a_back"], "A: front vs back"),
                 (["b_front"], ["b_back"], "B: front vs back"),
                 (["a_front", "a_front_repeat"], ["b_front"], "Front: A vs B"),
                 (["a_back"], ["b_back"], "Back: A vs B"),
                 (["clean", "clean_repeat"], ["a_front", "a_front_repeat"], "Focal only vs focal+A (front)")]
    rows = []
    for lefts, rights, label in contrasts:
        rows.append({"model": "gpt-5.6-sol",
                     **contrast(label, "sol/hydraulic", [call(c, r) for r in available for c in lefts],
                                [call(c, r) for r in available for c in rights], truth, focal,
                                [(len(lefts), len(rights))] * len(available), 0, rng)})
    singles = {}
    for rep in range(1, 6):
        folder = sources[rep] / "predictions/gpt-5.6-sol"
        for cid in focal:
            pred = load_predictions(folder, f"single_{cid}_r{rep}")
            if pred is not None:
                singles[cid] = pred[cid]
    # A "collapsed" response assigns one label to at least 16 of the 18 focal cycles.
    collapse = []
    for cond in ["clean", "clean_repeat", "pair_reverse", "descriptor_rotation", "a_front", "a_front_repeat",
                 "a_back", "b_front", "b_back"]:
        for rep in available:
            pred = call(cond, rep)
            top = Counter(pred[i] for i in focal).most_common(1)[0]
            collapse.append({"condition": cond, "block": rep, "collapsed": top[1] >= 16, "modal_label": top[0],
                             "modal_count": top[1], "correct": sum(pred[i] == truth[i] for i in focal)})

    def fisher(cond_a, cond_b):
        a = [c["collapsed"] for c in collapse if c["condition"] in cond_a]
        b = [c["collapsed"] for c in collapse if c["condition"] in cond_b]
        table = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
        return {"a": cond_a, "b": cond_b, "table": table, "p_two_sided": float(fisher_exact(table)[1])}
    canonical = ["clean", "clean_repeat"]
    collapse_tests = [fisher(canonical, ["pair_reverse"]), fisher(canonical, ["descriptor_rotation"]),
                      fisher(canonical, ["a_front", "a_front_repeat", "a_back", "b_front", "b_back"])]
    kept = [c["correct"] for c in collapse if not c["collapsed"]]
    dropped = [c["correct"] for c in collapse if c["collapsed"]]
    per_block = []
    for rep in available:
        entry = {"block": rep}
        for cond in ["clean", "clean_repeat", "pair_reverse", "descriptor_rotation", "a_front", "a_front_repeat",
                     "a_back", "b_front", "b_back"]:
            pred = call(cond, rep)
            entry[cond] = sum(pred[i] == truth[i] for i in focal)
        per_block.append(entry)
    return finish_family(rows), {"blocks_used": available, "per_block_correct": per_block,
                                 "collapse_calls": collapse, "collapse_tests": collapse_tests,
                                 "accuracy_noncollapsed": sum(kept) / (18 * len(kept)) if kept else None,
                                 "accuracy_collapsed": sum(dropped) / (18 * len(dropped)) if dropped else None,
                                 "n_noncollapsed": len(kept), "n_collapsed": len(dropped),
                                 "singleton_valid": len(singles),
                                 "singleton_correct": sum(singles[i] == truth[i] for i in singles)}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tep_rows, kappas = tep_independent(rng)
    abba_rows, abba_blocks = abba(rng)
    hyd_rows, hyd_info = hydraulic(rng)
    write_csv(OUT / "tep_contrasts.csv", tep_rows)
    write_csv(OUT / "tep_repeatability.csv", kappas)
    write_csv(OUT / "abba_contrast.csv", abba_rows)
    write_csv(OUT / "hydraulic_contrasts.csv", hyd_rows)
    summary = {"seed": SEED, "bootstrap_replicates": N_BOOT,
               "revision": "offline statistical correction 2026-09-30",
               "interval_scope": "TEP crossed case/call intervals; blocked contrasts use post-hoc case/matched-block sensitivity with legacy crossed intervals retained. E intervals resample cases conditional on observed calls. Few blocks limit population interpretation.",
               "tep": tep_rows, "tep_repeatability": kappas, "abba": abba_rows, "abba_blocks": abba_blocks,
               "hydraulic": hyd_rows, "hydraulic_info": hyd_info}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in tep_rows + abba_rows + hyd_rows:
        print(f"{row['family']:<22} {row['contrast']:<32} n={row['n_cases']:>2} "
              f"dAcc={100 * row['acc_diff']:6.1f} [{100 * row['acc_diff_ci_low']:6.1f},{100 * row['acc_diff_ci_high']:6.1f}] "
              f"McN p={row['primary_mcnemar_p']:.2g} holm={row['primary_mcnemar_p_holm']:.2g} "
              f"E={row['excess_disagreement']:.3f} [{row['excess_ci_low']:.3f},{row['excess_ci_high']:.3f}] "
              f"perm p_acc={row['call_perm_p_acc']:.3f} p_exc={row['call_perm_p_excess']:.3f} (R={row['call_relabellings']}) "
              f"D={row['stable_different_cases']}")
    print("hydraulic blocks used:", hyd_info["blocks_used"])


if __name__ == "__main__":
    main()
