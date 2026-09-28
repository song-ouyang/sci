"""Freeze all planned v3 terminal outcomes and logs without opening labels.

Successful predictions and semantic failures are both retained. Failed sessions
must not have repaired predictions or subsequent semantic retries.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.score_revision_v3 import inventory_files, outcome_counts, require, verify_frozen_runs


def freeze(base, runner_path=None):
    _, _, _, audit = verify_frozen_runs(base, runner_path, require_frozen=False)
    output = base / "predictions"
    target = output / "frozen_manifest.json"
    require(inventory_files(output) == audit["files_sha256"], "Artifacts changed during freeze audit")
    manifest = {"schema": "revision-v3-outcomes-frozen-v2",
                "frozen_at": audit["frozen_at"] or datetime.now(timezone.utc).isoformat(),
                "files": audit["files_sha256"], "protocol_sha256": audit["protocol_sha256"],
                "input_sha256": audit["input_sha256"], "runner_sha256": audit["runner_sha256"],
                "failure_reporting_amendment_sha256": audit["failure_reporting_amendment_sha256"],
                "transport_audit_note_sha256": audit["transport_audit_note_sha256"],
                "recovered_stream_reconnection_details": audit["recovered_stream_reconnection_details"],
                **outcome_counts(audit), "session_failures": audit["session_failures"], "answer_keys_read": False}
    raw = json.dumps(manifest, indent=2) + "\n"
    if target.exists():
        require(target.read_text() == raw, "Frozen outcomes/logs changed; refusing to refresh manifest")
    else:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(raw)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/revision_v3")
    args = parser.parse_args(argv)
    manifest = freeze(args.root.resolve())
    print(f"Frozen {manifest['planned_sessions']} planned outcomes: {manifest['completed_sessions']} valid, "
          f"{manifest['failed_sessions']} semantic failures; {manifest['total_valid_predictions']}/"
          f"{manifest['expected_predictions']} valid predictions. Answer labels not read.")


if __name__ == "__main__":
    main()
