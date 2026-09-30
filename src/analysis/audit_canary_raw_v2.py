"""Independently reconstruct the two canary studies from frozen raw arrays.

This post-execution verification is separate from the pre-execution-frozen
scorer. It performs no inference and changes no protocol or raw response.
Only matching C/R arms enter the earlier-period comparison. On an existing
output it requires byte-identical reconstruction before writing it again.
"""
import collections
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run_hydraulic_v1 import audit_outputs


def reconstruct():
    out = ROOT / "results/canary_stability_v2/analysis"
    allreports = {}
    for study in ["canary_stability_v2", "canary_hydraulic_v1"]:
        b = ROOT / "results" / study
        audit_outputs(b, require_frozen=True)
        protocol = json.loads((b / "protocol.json").read_text())
        truthrows = json.loads((b / "answers/records.json").read_text())["records"]
        truth = {r["id"]: r["true_label"] for r in truthrows}
        focal = [r["id"] for r in truthrows if r["role"] == "focal"]
        canary = [r["id"] for r in truthrows if r["role"] == "canary"]
        info = {r["id"]: r for r in truthrows}
        schedule = json.loads((b / "schedule.json").read_text())["blocks"]
        folder = b / "predictions" / protocol["model"]
        calls, windows, errors = [], [], []
        for block, names in enumerate(schedule, 1):
            for slot, n in enumerate(names, 1):
                if n[0] not in "CR":
                    continue
                raw = (folder / f"{n}_attempt1_response.txt").read_text().strip()
                if raw.startswith("```"):
                    raw = "\n".join(raw.splitlines()[1:-1])
                predlist = json.loads(raw)
                pred = {r["id"]: r["predicted"] for r in predlist}
                assert len(pred) == len(predlist) == 27 and set(pred) == set(truth)
                assert predlist == json.loads((folder / f"{n}_results.json").read_text())
                meta = json.loads((folder / f"{n}_metadata.json").read_text())
                dispatch = json.loads((folder / f"{n}_dispatch.json").read_text())
                a = meta["attempts"][0]
                assert meta["status"] == "complete" and len(meta["attempts"]) == 1
                calls.append({"group": n, "block": block, "slot": slot, "arm": n[0],
                    "correct": sum(pred[i] == truth[i] for i in focal),
                    "canary_correct": sum(pred[i] == truth[i] for i in canary),
                    "modal": max(collections.Counter(pred.values()).values()),
                    "started_at": dispatch["started_at"], "completed_at": meta["updated_at"],
                    "tokens": a["usage"]["input_tokens"] + a["usage"]["output_tokens"],
                    "seconds": a["elapsed_seconds"], "pred": pred})
        for block in range(1, 4):
            for arm in "CR":
                seq = sorted([r for r in calls if r["block"] == block and r["arm"] == arm], key=lambda x: x["slot"])
                assert len(seq) == 3
                assert [r["group"] for r in seq] == [r["group"] for r in sorted(seq, key=lambda r: datetime.fromisoformat(r["started_at"]))]
                for policy in ["Single", "Known_answer", "Collapse_90", "Vote_first_tie", "Vote_abstain_tie"]:
                    sel, req = None, seq
                    if policy == "Single":
                        sel, req = seq[0], seq[:1]
                    elif policy == "Known_answer":
                        sel = next((r for r in seq if r["canary_correct"] >= 8), None)
                        req = seq[:seq.index(sel) + 1] if sel else seq
                    elif policy == "Collapse_90":
                        sel = next((r for r in seq if r["modal"] * 10 < 27 * 9), None)
                        req = seq[:seq.index(sel) + 1] if sel else seq
                    if policy.startswith("Vote"):
                        chosen = {}
                        for i in focal:
                            cc = collections.Counter(r["pred"][i] for r in seq)
                            maximum = max(cc.values())
                            top = [k for k, v in cc.items() if v == maximum]
                            chosen[i] = top[0] if len(top) == 1 else seq[0]["pred"][i] if policy == "Vote_first_tie" else None
                    else:
                        chosen = sel["pred"] if sel else {i: None for i in focal}
                    row = {"block": block, "arm": arm, "policy": policy, "selected": sel["group"] if sel else "",
                        "correct": sum(chosen[i] == truth[i] for i in focal),
                        "error": sum(chosen[i] is not None and chosen[i] != truth[i] for i in focal),
                        "withheld": sum(chosen[i] is None for i in focal), "calls": len(req),
                        "tokens": sum(r["tokens"] for r in req), "seconds": sum(r["seconds"] for r in req)}
                    assert row["correct"] + row["error"] + row["withheld"] == 18
                    windows.append(row)
                    if policy == "Known_answer":
                        for i in focal:
                            if chosen[i] is not None and chosen[i] != truth[i]:
                                errors.append({"arm": arm, "block": block, "selected": row["selected"], "id": i,
                                    "cycle": info[i]["cycle_1based"], "source_block": info[i]["block_id"],
                                    "true": truth[i], "predicted": chosen[i]})
        if study == "canary_stability_v2":
            with (out / "policy_windows.csv").open() as f:
                scored = list(csv.DictReader(f))
            with (out / "policy_cases.csv").open() as f:
                cases = list(csv.DictReader(f))
            for w in windows:
                prior = next(r for r in scored if r["policy"] == w["policy"] and int(r["block"]) == w["block"] and r["arm"] == w["arm"])
                cc = [r for r in cases if r["policy"] == w["policy"] and int(r["block"]) == w["block"] and r["arm"] == w["arm"]]
                assert [sum(r["action"] == a for r in cc) for a in ["accepted_correct", "accepted_error", "rejected"]] == [w["correct"], w["error"], w["withheld"]]
                assert prior["selected_call"] == w["selected"] and int(prior["planned_requests"]) == w["calls"] and int(prior["known_tokens"]) == w["tokens"]
            for bnum in (1, 2):
                assert max(datetime.fromisoformat(r["completed_at"]) for r in calls if r["block"] == bnum) < min(datetime.fromisoformat(r["started_at"]) for r in calls if r["block"] == bnum + 1)
        totals = {p: {k: sum(r[k] for r in windows if r["policy"] == p) for k in ["correct", "error", "withheld", "calls", "tokens", "seconds"]}
                  for p in ["Single", "Known_answer", "Collapse_90", "Vote_first_tie", "Vote_abstain_tie"]}
        allreports[study] = {"independent_direct_raw_recomputation": True, "arms": ["C", "R"],
                           "calls": [{k: v for k, v in r.items() if k != "pred"} for r in calls],
                           "windows": windows, "totals": totals, "retained_errors": errors}
    path = out / "independent_recomputation.json"
    data = (json.dumps(allreports, indent=2) + "\n").encode()
    existed = path.exists()
    if existed:
        assert path.read_bytes() == data, "Independent reconstruction differs from the existing JSON"
    path.write_bytes(data)
    print(json.dumps({"path": str(path.relative_to(ROOT)), "existing_bytes_identical": existed,
                      "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    reconstruct()
