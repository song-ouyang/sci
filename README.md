# sci: error structure and known-answer screening for LLM-based industrial fault diagnosis

Code, fixed inputs and frozen model outputs for the paper

> **Error structure and known-answer screening for LLM-based industrial fault diagnosis: a repeated-call study**
> Song Ouyang (2026). Manuscript under review.

The study evaluates general-purpose LLM services as diagnostic interfaces for industrial sensor data. Each measured value is serialized as a text field (`ID | name | unit | value`). The study injects descriptor-mapping faults, meaning-preserving reorderings and context changes, submits every input several times, and tests each effect against call-to-call variation.

## What is in this repository

| Path | Contents | Used in the paper |
|---|---|---|
| `manuscript/` | LaTeX sources (`main.tex`, `supplement.tex`, `references.bib`, `figures/`) | the full paper |
| `src/experiments/` | input preparation and single-attempt inference runners (Codex CLI) | Sections 3–4 |
| `src/analysis/` | scoring, statistics, baselines and figures (offline, no model calls) | Sections 4–5 |
| `tests/` | unit tests for parsers, runners and audits | — |
| `data/` | processed features only (TEP, C-MAPSS, hydraulic rig summaries) | Section 4.1 |
| `results/independent_validation/` | source-disjoint TEP/FD002 study: protocol, prompts, 84 frozen responses, per-case CSV | Tables 4–6, Figs. 2–4, 6 |
| `results/context_blocked_v1/` | ABBA/BAAB time-balanced replication (16 calls) | Table 6, Fig. 4b |
| `results/hydraulic_validation_v1/` | measured hydraulic rig, blocks R1–R2 (and the interrupted R3) | Table 7, Fig. 5 |
| `results/hydraulic_validation_v1_completion/` | versioned completion of blocks R3–R5 (37/37 valid) | Table 7, Fig. 5 |
| `results/statistics_v2/` | McNemar, crossed bootstrap, randomization tests, kappa, collapse, voting, call timing, LLM-vs-baseline tests | Tables 4–7, Figs. 3–6, Suppl. App. F |
| `results/baselines_v2/` | tuned and target-informed supervised baselines, per-case predictions | Table 4, Suppl. App. H |
| `results/call_timing_v2/` | client-side request timing audit | Fig. 4a |
| `results/policy_hydraulic_v1/`, `results/policy_paderborn_v1/` | paired-position and decision-check studies on new hydraulic blocks and the Paderborn bearing rig | Table 8, Suppl. App. I |
| `results/bearing_baseline_v1/` | same-feature supervised bearing benchmark | Section 5.1, Suppl. App. J |
| `results/canary_hydraulic_v1/` | prospective known-answer (canary) screening study, frozen before calls | Table 8, Fig. 6b |
| `results/error_structure_v3/` | error structure, single-call audit, canary screening and decision policies across all studies | Fig. 6, Table 8, Suppl. App. F |
| `results/revision_v3/`, `mechanism_extension/`, `domestic_extension_v2/` | preliminary 130-source panel and cross-provider checks | Supplementary Sections 2–4 |
| `results/claude_*` | non-controlled historical records, kept only for Supplementary Appendix B | supplement only |

Every study directory follows the same layout:

```
protocol.json / protocol.sha256   fixed before any inference
inference/prompts/*.txt           exact payloads sent to the service
schedule.json                     call order and execution blocks
predictions/<model>/              raw responses, event logs, metadata, parsed labels
predictions/frozen_manifest.json  SHA-256 of every output, written before scoring
answers/records.json              ground truth, opened only after freezing
analysis/                         scores and per-case CSV files
```

## Reproducing the analyses

All analyses run offline from the frozen outputs; no model access is needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/analysis/statistics_v2.py      # inferential statistics  -> results/statistics_v2/
python src/analysis/voting_time_v2.py     # voting and call timing  -> results/statistics_v2/
python src/analysis/baselines_v2.py       # supervised baselines    -> results/baselines_v2/ (about 8 min)
python src/analysis/llm_vs_baselines_v2.py  # paired LLM-vs-baseline tests
python src/analysis/stats_tables_v2.py    # supplementary tables (Appendix F)
python src/analysis/error_structure_v3.py      # error structure and canary screening
python src/analysis/single_call_audit_v3.py    # single-call vs call-level evidence
python src/analysis/decision_policies_v3.py    # decision policies across studies
python src/analysis/score_canary_hydraulic_v1.py  # prospective study (frozen scoring rules)
python src/analysis/structure_tables_v3.py     # supplementary tables for the above
python src/analysis/figures_v2.py         # figures                 -> manuscript/figures/v2/
cd manuscript && tectonic main.tex         # compile the paper
```

Re-running inference is possible with `src/experiments/run_hydraulic_v1.py run --root <study dir>` and the Codex CLI (`codex-cli 0.155.0-alpha.9.2`) authenticated to an account with access to the requested models. Each runner refuses to overwrite frozen outputs; use a new study directory.

## Data sources and licences

| Dataset | Source | Licence / terms |
|---|---|---|
| Tennessee Eastman Process (expanded simulation) | Rieth et al. (2017), Harvard Dataverse, doi:10.7910/DVN/6C3JR1 | see the original dataset page |
| C-MAPSS turbofan degradation | Saxena & Goebel (2008), NASA Prognostics Data Repository | NASA open data |
| Condition monitoring of hydraulic systems | Helwig et al. (2015), UCI, doi:10.24432/C5CW21 | CC BY 4.0 |
| Paderborn bearing data | Lessmeier et al. (2016), Paderborn University KAt data center | see the original dataset page |

Raw datasets are **not** redistributed. `download_datasets.py` fetches them from the original sources.

Suggested licences for this repository: MIT for code; CC BY 4.0 for derived data, prompts and model outputs.

## Excluded on purpose

- raw third-party datasets (download them with `download_datasets.py`);
- API keys and `.env` files (never commit them);
- `.venv/`, `__pycache__/`, packaged `.zip` review archives;
- superseded early batches (`results/batches*`, `results/gpt_models*`), which predate the frozen protocols.

Note: some frozen metadata and log files record absolute local paths from the machine that ran the experiments. They are left unchanged because their SHA-256 hashes are part of the frozen manifests.

## Citation

```bibtex
@misc{ouyang2026repeatedcall,
  author = {Ouyang, Song},
  title  = {Error structure and known-answer screening for {LLM}-based industrial fault diagnosis: a repeated-call study},
  year   = {2026},
  note   = {Code and data: https://github.com/song-ouyang/sci}
}
```
