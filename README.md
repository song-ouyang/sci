# Known-answer screening for industrial fault diagnosis with large language models

Reproducibility materials for **Known-answer screening for industrial fault diagnosis with large language models: prospective evaluation and failure analysis**, by Song Ouyang, Xin Lu and Yan Yu.

This is the corrected 30 September 2026 revision prepared for Quality and Reliability Engineering International (subscription route). Publication of this repository does not claim journal submission, peer-review acceptance or publication. The versioned snapshot is tagged `qrei-2026-09-30`.

## Principal evidence and limits

A locally frozen 27-call hydraulic study compared a known-answer screen with first-call and voting policies. The screen released 123 correct and 3 incorrect labels and withheld 36 of 162 window decisions. A post hoc concentration-only check selected exactly the same calls. Thus the original study does not establish incremental window benefit from reference answers.

Before a later 18-call follow-up we fixed both screening rules, new focal/reference source blocks, the call schedule and scoring code. Across six windows, the known-answer policy released 100 correct and 8 incorrect labels, versus 88 correct and 20 incorrect for Single and the concentration-only check. Coverage was 100%; offline replay required 1.5 rather than 1 call per window. All 18 calls were actually collected. These 108 window decisions reuse 18 focal cycles from nine source blocks. Gains occur in two of six windows, and all eight residual errors repeat two weak-leakage cycles from one block. One unchanged rig, and jointly changed panels, period and client version, limit generalization.

Historical analysis corrections use actual dispatch chronology, the six allocations allowed by the ABBA/BAAB design, and matched execution blocks. Original frozen protocols and responses are unchanged. This is an empirical evaluation of a bounded acceptance check, not a universal diagnostic guarantee.

## Files

| Path | Content |
|---|---|
| `manuscript/main.pdf`, `manuscript/supplement.pdf` | Current reading copies |
| `results/industrial_interface_focused_overleaf/` | Canonical editorial LaTeX, bibliography, tables and figures; `manuscript/` is its reading mirror |
| `results/canary_hydraulic_v1/` | Original 27-call prospective study |
| `results/canary_controls_v2/` | Post hoc concentration control and threshold sensitivity |
| `results/canary_stability_v2/` | Later 18-call frozen follow-up, source audits, requests and policy records |
| `results/statistics_v2/`, `results/error_structure_v3/` | Corrected statistics, chronology, error structure and policy comparisons |
| `results/cc_statistical_correction_v4/` | Correction provenance and comparison with reconstructed historical calculations |
| `results/reference_audit_20260930/` | Reference checks, corrections and source-access limitations |
| `results/independent_validation/`, `results/context_blocked_v1/` | Simulation-source study and blocked replication |
| `results/hydraulic_validation_v1/`, `results/hydraulic_validation_v1_completion/` | Earlier measured hydraulic studies |
| `results/policy_hydraulic_v1/`, `results/policy_paderborn_v1/` | Position/check studies on hydraulic and bearing records |
| `results/baselines_v2/`, `results/bearing_baseline_v1/` | Same-feature supervised references |
| `src/`, `tests/` | Versioned preparation, execution, scoring, audit and reconstruction code |
| `data/` | Processed features and provenance; original raw datasets are not redistributed |
| `paper/archive/pre_cc_correction_20260930/src/analysis/` | Preserved analysis code needed to reconstruct the explicitly labelled historical comparison |
| `PUBLICATION_MANIFEST.json` | SHA-256 inventory of this revision, including retained historical files and immutable evidence |

Other retained results are historical/exploratory evidence; their protocols and appendices specify their role. Date-stamped verification reports describe the files identified by their hashes, not all later revisions. A computational reference audit is not independent human peer review.

## Offline reproduction

Use Python with the packages in `requirements.txt`. No model credentials, new inference or raw-data download is required to rebuild the reported corrected analyses from frozen outputs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/analysis/reproduce_public_revision.py
```

This verifies immutable evidence, rebuilds historical corrections and both control analyses, independently checks raw-response arithmetic, renders follow-up tables, and refreshes the `manuscript/` mirror. To also compile both reading PDFs with an existing Tectonic installation and available TeX cache:

```bash
python src/analysis/reproduce_public_revision.py --compile
```

Alternatively compile `main.tex` and `supplement.tex` separately in `manuscript/` using a normal LaTeX/BibTeX installation. Document compilation is separate from numeric reconstruction.

Relevant regression checks:

```bash
python -m unittest tests.test_cc_statistical_correction_v4 tests.test_canary_controls_v2 tests.test_canary_stability_v2 tests.test_run_hydraulic_v1
```

Do not execute study `prepare`, `run` or `freeze` commands to reproduce existing results. Those commands belong to new experiments and must use a separately versioned directory. The frozen scoring code and response manifests must not be edited.

The exported research files exclude submission correspondence, local credentials, `.env` files, virtual environments and downloaded third-party papers. Some frozen logs retain historical absolute local paths, because their exact bytes are part of the audit record.

## Dataset sources and terms

- Expanded Tennessee Eastman Process simulation: Rieth et al., Harvard Dataverse, https://doi.org/10.7910/DVN/6C3JR1.
- C-MAPSS turbofan degradation: NASA Prognostics Data Repository; see the manuscript dataset reference and original NASA terms.
- Hydraulic condition monitoring: UCI, https://doi.org/10.24432/C5CW21 (CC BY 4.0).
- Paderborn bearing data: Paderborn University KAt data center; see Lessmeier et al. in the manuscript and the original data-source terms.

Underlying datasets retain their original terms. This revision does not assign a new license to third-party data or publications. No downloaded third-party full-text literature is redistributed with the reference audit.

## Citation

```bibtex
@misc{ouyang2026knownanswer,
  author = {Ouyang, Song and Lu, Xin and Yu, Yan},
  title = {Known-answer screening for industrial fault diagnosis with large language models: prospective evaluation and failure analysis},
  year = {2026},
  note = {Reproducibility revision qrei-2026-09-30; manuscript prepared for submission},
  url = {https://github.com/song-ouyang/sci/tree/qrei-2026-09-30}
}
```

AI-assisted code, checks and language work are disclosed in the manuscript. The Scientific Agent Skills software/method reference is included there with the other verified bibliographic records; it is not evidence for the experimental findings.
