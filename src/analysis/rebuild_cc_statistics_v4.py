"""Offline correction workflow: verify immutable evidence, rebuild, test, log provenance.

Run from the repository: .venv/bin/python src/analysis/rebuild_cc_statistics_v4.py
No model runner is imported or launched. Original frozen protocols, response
files, and frozen scoring code are read only. Figures/tables are derived files.
"""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cc_statistical_correction_v4"
ARCHIVE = ROOT / "paper/archive/pre_cc_correction_20260930"
STUDIES = ["independent_validation", "context_blocked_v1", "hydraulic_validation_v1",
           "hydraulic_validation_v1_completion", "policy_hydraulic_v1", "policy_paderborn_v1",
           "canary_hydraulic_v1"]
MODULES = ["statistics_v2", "error_structure_v3", "decision_policies_v3", "single_call_audit_v3",
           "stats_tables_v2", "structure_tables_v3"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_inventory():
    files, checks = {}, []
    for study in STUDIES:
        base = ROOT / "results" / study
        protocol_path = base / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        manifest_path = base / "predictions/frozen_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        expected = {base / "predictions" / name: digest for name, digest in manifest["files"].items()}
        expected[protocol_path] = manifest["protocol_sha256"]
        expected.update({ROOT / name: digest for name, digest in protocol.get("code_sha256", {}).items()})
        for path, wanted in expected.items():
            got = sha(path) if path.exists() else None
            checks.append({"study": study, "path": str(path.relative_to(ROOT)), "expected": wanted,
                           "actual": got, "matches": got == wanted})
            if got is not None:
                files[str(path.relative_to(ROOT))] = got
        # Hash local analysis inputs and the declarations themselves, too. These
        # are logged as inputs, not retroactively described as frozen artifacts.
        for pattern in ["protocol.json", "schedule.json", "answers/*.json", "analysis/*.csv", "analysis/scores.json"]:
            for path in base.glob(pattern):
                files[str(path.relative_to(ROOT))] = sha(path)
        files[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
    return {"files": files, "freeze_checks": checks,
            "mismatches": [c for c in checks if not c["matches"]]}


def run(arguments):
    env = dict(os.environ, MPLBACKEND="Agg", SOURCE_DATE_EPOCH="1790726400")
    proc = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env, text=True, capture_output=True)
    log_name = arguments[-1].replace("/", "_").replace(".", "_")
    (OUT / f"run_{log_name}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"Failed {arguments}: {proc.stderr[-1500:]}")


def summary():
    stats = json.loads((ROOT / "results/statistics_v2/summary.json").read_text())
    audit = json.loads((ROOT / "results/error_structure_v3/single_call_audit.json").read_text())
    policies = read_csv(ROOT / "results/error_structure_v3/decision_policies.csv")
    # The pre-correction archive preserved source and manuscript tables, not
    # every intermediate CSV. Reconstruct its old replay in this new directory
    # from the archived source; never overwrite an archived/frozen artifact.
    def module_from_archive(name):
        spec = importlib.util.spec_from_file_location("legacy_" + name, ARCHIVE / f"src/analysis/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = ROOT
        return module
    legacy_structure = module_from_archive("error_structure_v3")
    key = "src.analysis.error_structure_v3"
    saved = sys.modules.get(key)
    try:
        sys.modules[key] = legacy_structure
        legacy_policy = module_from_archive("decision_policies_v3")
        legacy_policy.OUT = OUT / "legacy_replay"
        legacy_policy.OUT.mkdir(exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            legacy_policy.main()
        old = read_csv(legacy_policy.OUT / "decision_policies.csv")
    finally:
        if saved is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved
    old_by = {(r["study"], r["policy"]): r for r in old}
    changed = [{"before": old_by[(r["study"], r["policy"])], "after": r} for r in policies
               if (r["study"], r["policy"]) in old_by and any(
                   r[k] != old_by[(r["study"], r["policy"])][k]
                   for k in ["accepted_correct", "accepted_error", "abstained", "calls_per_window"])]
    groups = read_csv(ROOT / "results/error_structure_v3/groups.csv")
    return {"scope": "Offline correction; not new experimental evidence or independent human peer review.",
            "primary_version": "results/industrial_interface_focused_overleaf/main.tex (CC version)",
            "archive": str(ARCHIVE.relative_to(ROOT)), "abba": stats["abba"][0],
            "legacy_comparison_source": "Old CSV reconstructed locally from archived source into cc_statistical_correction_v4/legacy_replay; this reconstructed CSV is not an original frozen artifact.",
            "blocked_interval_sensitivity": stats["hydraulic"], "single_call_audit": audit,
            "decision_policies": policies, "changed_policy_rows": changed,
            "always_wrong_breakdown": [{k: g[k] for k in ["study", "service", "group", "n_cases", "calls", "stable_error",
                                        "stable_error_same_label", "stable_error_varying_labels", "plurality_correct",
                                        "plurality_wrong", "plurality_withheld_ties"]}
                                      for g in groups if int(g["stable_error"])],
            "corrections": [
                "All first-call and first-canary-pass choices use timezone-aware started_at metadata and stable call IDs; no repeat-number fallback.",
                "ABBA exact primary null has six allowable allocations, with old 1296-assignment exchangeability result retained as sensitivity.",
                "Case x matched-block bootstrap is explicitly a post-hoc sensitivity; old independently sampled call intervals remain in CSV.",
                "19-contrast single-call comparison shows raw and original-family Holm thresholds separately; neither is a false-positive-rate estimate.",
                "Always-wrong cases are split by label agreement. Figure6 is same-data descriptive accounting, with bound G<=100(1-s).",
                "Plurality uses actual k and explicit tie rules. Prospective frozen tie-first and exploratory tie-withhold results coexist.",
                "Retry summaries use the same full windows as policy replay; fixed three-call chunks were removed from ancillary summaries.",
                "Retrospective tau=.8 does not establish preregistration; prospective 8/9 accepts one wrong canary."
            ]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = frozen_inventory()
    if before["mismatches"]:
        (OUT / "frozen_input_audit.json").write_text(json.dumps(before, indent=2) + "\n")
        raise RuntimeError(f"Frozen input mismatches: {len(before['mismatches'])}; inspect frozen_input_audit.json")
    run(["-m", "unittest", "tests.test_cc_statistical_correction_v4", "-v"])
    for module in MODULES:
        run([f"src/analysis/{module}.py"])
    run(["src/analysis/figures_v2.py", "framework", "forest", "timing", "hydraulic", "structure"])
    after = frozen_inventory()
    if before != after:
        raise RuntimeError("Immutable inputs changed during offline rebuild")
    (OUT / "frozen_input_audit.json").write_text(json.dumps(before, indent=2) + "\n")
    result = summary()
    (OUT / "correction_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    output_paths = list((ROOT / "results/statistics_v2").glob("*.csv")) + list((ROOT / "results/statistics_v2").glob("*.json"))
    output_paths += list((ROOT / "results/error_structure_v3").glob("*.csv")) + list((ROOT / "results/error_structure_v3").glob("*.json"))
    overleaf = ROOT / "results/industrial_interface_focused_overleaf"
    output_paths += [overleaf / "supp_stats_tables.tex", overleaf / "supp_structure_tables.tex"]
    output_paths += [overleaf / f"figures/v2/{name}.pdf" for name in ["framework", "forest", "timing", "hydraulic", "error_structure"]]
    source_paths = [ROOT / f"src/analysis/{name}.py" for name in MODULES + ["figures_v2", "rebuild_cc_statistics_v4"]]
    source_paths += [ROOT / "tests/test_cc_statistical_correction_v4.py"]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": ".venv/bin/python src/analysis/rebuild_cc_statistics_v4.py",
                  "original_frozen_inputs_unchanged": True, "freeze_checks": len(before["freeze_checks"]),
                  "unique_input_files": len(before["files"]), "regression_tests": 6,
                  "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
                  "output_sha256": {str(p.relative_to(ROOT)): sha(p) for p in output_paths},
                  "input_audit_sha256": sha(OUT / "frozen_input_audit.json"),
                  "correction_summary_sha256": sha(OUT / "correction_summary.json")}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "freeze_checks": provenance["freeze_checks"],
                      "original_frozen_inputs_unchanged": True, "regression_tests": 6,
                      "raw_audit": [audit_value for audit_value in [result["single_call_audit"]["raw_significant_without_call_criterion"], result["single_call_audit"]["raw_significant"]]],
                      "holm_audit": [result["single_call_audit"]["holm_significant_without_call_criterion"], result["single_call_audit"]["holm_significant"]]}, indent=2))


if __name__ == "__main__":
    main()
