"""Fail-closed scoring entry for the unauditable first Claude extension.

The historical numerical scorer is preserved in the dated audit archive. This
entry never opens answer-key dataset files: its collection lacks a verified inference and
pre-score freeze chain. Shape-valid predictions do not establish eligibility.
A corrected rerun requires a separately versioned, provider-native adapter.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.audit_revision_v3_claude_extension import audit_extension, write_reports

EXT_DIR = ROOT / "results/revision_v3_claude_extension"


def write_unavailable_scores(report, base):
    """Invalidate derived metrics without modifying original inference artifacts."""
    scores, rows = {}, []
    for session in report["sessions"]:
        model, group = session["model"], session["group"]
        reasons = [item["code"] for item in session["issues"]]
        row = {"model": model, "group": group,
               "status": "not_eligible_for_primary_comparison",
               "audit_status": session["status"],
               "eligible_for_primary_comparison": False,
               "accuracy": None, "macro_f1": None,
               "accuracy_95ci": [None, None], "macro_f1_95ci": [None, None],
               "issue_codes": reasons}
        scores.setdefault(model, {})[group] = row
        rows.append({k:(";".join(v) if k == "issue_codes" else v)
                     for k,v in row.items() if not k.endswith("95ci")})
    output = base / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"audit_recorded_at": datetime.now(timezone.utc).isoformat(),
               "status": "not_eligible_for_primary_comparison",
               "answer_key_files_opened": False,
               "note": "N/A reflects missing or conflicting experiment provenance, not zero accuracy. No classification metrics were computed by this entry.",
               "audit_report": "../audit/report.json",
               "historical_archive": "results/claude_extension_audit_archive/2026-09-27/",
               "scores": scores}
    (output / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Claude extension: primary-comparison eligibility blocked", "",
             f"All {len(report['sessions'])} submitted model/group records are unavailable for the controlled primary comparison.",
             "This does not establish that every prediction is incorrect or copied. The stored",
             "artifacts cannot verify inference provenance, full-input delivery and isolation.",
             "Removing a historical contamination flag does not restore eligibility. Original artifacts are preserved.",
             "", "See [integrity audit](../audit/report.md). The old numerical summaries are",
             "preserved in `results/claude_extension_audit_archive/2026-09-27/` and are not verified scores.",
             "", "| Claimed model | Session | Primary accuracy / F1 | Audit status |",
             "|---|---|---|---|"]
    lines += [f"| {s['model']} | {s['group']} | N/A | {s['status']} |"
              for s in report["sessions"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="Read-only audit; return nonzero when eligibility is blocked")
    args = parser.parse_args(argv)
    report = audit_extension(EXT_DIR, ROOT / "results/revision_v3")
    if any(s["eligible_for_primary_comparison"] for s in report["sessions"]):
        raise RuntimeError("This legacy entry cannot certify a new experiment. Use a separately reviewed provider-native scorer.")
    if not args.validate_only:
        write_reports(report, EXT_DIR)
        write_unavailable_scores(report, EXT_DIR)
    print(f"BLOCKED: 0/{len(report['sessions'])} records eligible for the primary comparison. "
          "Answer-key dataset files were not opened. See the Claude extension integrity audit.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
