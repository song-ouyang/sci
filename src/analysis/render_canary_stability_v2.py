"""Render audited v2 study results without changing the frozen scorer or outputs.

Reads the independent direct-raw reconstruction alongside the frozen scorer's
summary. This rendering stage is subsequent to outcome observation.
"""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/canary_stability_v2"
OUT = ROOT / "results/industrial_interface_focused_overleaf"

def tab(caption, label, spec, header, rows):
    return "\n".join([r"\begin{table}[htbp]\centering\small",r"\caption{"+caption+"}",
       r"\label{"+label+"}",r"\begin{tabular}{@{}"+spec+r"@{}}\toprule",header+r" \\\midrule",
       *[" & ".join(map(str,r))+r" \\" for r in rows],r"\bottomrule\end{tabular}\end{table}"])

def texid(value):
    return r"\texttt{"+value.replace("_",r"\_")+"}"

def render():
    summary = json.loads((BASE/"analysis/summary.json").read_text())
    independent = json.loads((BASE/"analysis/independent_recomputation.json").read_text())
    new = independent["canary_stability_v2"]
    old = independent["canary_hydraulic_v1"]
    p = summary["policies"]
    for policy, expected in new["totals"].items():
        for k,original in [("correct","accepted_correct"),("error","accepted_error"),("withheld","rejected"),("calls","requests"),("tokens","known_tokens")]:
            assert expected[k] == p[policy][original]
    assert len(new["calls"]) == 18 and sum(r["tokens"] for r in new["calls"]) == 408408
    assert len({r["source_block"] for r in new["retained_errors"]}) == 1
    assert {r["cycle"] for r in new["retained_errors"]} == {552,555}
    assert len(new["retained_errors"]) == 8
    assert max(r["modal"] for r in new["calls"]) == 15
    assert min(r["modal"] for r in new["calls"]) == 9
    labels = [("Single","Single"),("Known_answer","Known answer"),("Collapse_90",r"Collapse only (90\%)"),("Vote_first_tie","Three-call vote")]
    main_table = tab(r"Later-period follow-up on six C/R windows. All four policies cover 108 window decisions on the same 18 focal cycles; calls are offline required requests, not actual adaptive savings.",
        "tab:canary_stability_main","lrrrr","Policy & Correct & Error & Withheld & Calls/window",
        [[name,p[key]["accepted_correct"],p[key]["accepted_error"],p[key]["rejected"],f'{p[key]["calls_per_window"]:.2f}'] for key,name in labels])
    main = r"""\paragraph{Prospectively fixed later-period follow-up.}
After specifying the collapse control, we froze the same screening thresholds and scorer for a fresh 18-request follow-up on 30 September. The unchanged hydraulic rig supplies 18 new focal cycles from nine previously unused source blocks (two spaced cycles per block), plus nine new canaries from nine other unused blocks. Canonical (C) and intact-pair reversal (R) each receive three byte-identical calls in each of three chronological blocks. The source panel, canary panel, service period and client version change jointly; this is not a separate causal test of any one change or a new-device replication.

""" + main_table + r"""

Known-answer screening retains 100 correct and eight incorrect labels among 108 window decisions (92.6\% correct yield and 7.4\% accepted-case error), compared with 88 correct and 20 incorrect for Single and the 90\% collapse control. All three policies cover every window decision. This descriptive gain requires nine calls rather than six (1.50 versus 1.00 per window), corresponding to 206,117 rather than 135,535 observed input-plus-output tokens in the offline replay. The full study actually collected all 18 calls and used 408,408 tokens.

Every new call passes the collapse rule: its modal class occupies only 9--15 of 27 labels. Known-answer screening rejects ten of those calls. Its window-level gain is concentrated in C/block~2 (+10 correct) and R/block~3 (+2); the other four windows are unchanged. Thus reference labels provide discrimination beyond this fixed collapse check in the new sample, but these six dependent windows do not establish a population benefit. All eight retained errors repeat two weak-leakage cycles from one source block, classified as no leakage in four selected calls. The guard therefore does not assure correctness for every source even when its reference check passes. Supplementary Appendix~M gives all 18 calls, six windows, retained errors, and comparisons restricted to the earlier study's matching C/R arms.
"""
    (OUT/"canary_stability_main.tex").write_text(main)

    call_rows = [[r["block"],r["slot"],texid(r["group"]),f'{r["canary_correct"]}/9',f'{r["correct"]}/18',f'{r["modal"]}/27',"Pass" if r["canary_correct"]>=8 else "Reject",f'{r["tokens"]:,}'] for r in new["calls"]]
    call_table = tab(r"Every later-period request in frozen schedule order. Slot is an integer within a chronological block, not the copy suffix in the call name. All 18 responses are valid. Every call passes the 90\% collapse check; the gate column reports the known-answer check only.",
                    "tab:canary_stability_calls","rrlrrrlr","Block & Slot & Call & Canary & Focal & Modal & Gate & Tokens",call_rows)
    window_rows = []
    for block in range(1,4):
        for arm in "CR":
            ws={r["policy"]:r for r in new["windows"] if r["block"]==block and r["arm"]==arm}
            window_rows.append([f"{arm}/{block}",texid(ws["Single"]["selected"]),texid(ws["Known_answer"]["selected"]),
                                ws["Single"]["correct"],ws["Known_answer"]["correct"],ws["Vote_first_tie"]["correct"],ws["Known_answer"]["calls"]])
    window_table = tab(r"All six later-period windows, each containing the same 18 focal cycles. Entries S, K and V are correct counts for Single (also collapse), known-answer screening and the first-call-tie three-call vote. No output is withheld by these three primary policies; errors equal $18-$correct. K calls is its offline required-call count.",
        "tab:canary_stability_windows","ll lrrrr","Window & First call & K selected call & S & K & V & K calls",window_rows)
    cross = [r for r in summary["gate_cross_tab"] if r["collapse_threshold"]=="9/10"]
    cross_table = tab(r"Later-period call-level acceptance cross-tab. These counts use all focal outputs in each cell, not only outputs selected by a window policy. The same 18 focal cycles recur in every request.",
        "tab:canary_stability_cross","llrrrr","Known gate & Collapse gate & Calls & Focal outputs & Correct & Error",
        [["Pass" if r["known_pass"] else "Reject","Pass" if r["collapse_pass"] else "Reject",r["calls"],r["focal_outputs"],r["focal_correct"],r["focal_errors"]] for r in cross])
    policy_table = tab(r"Later-period policy and tie sensitivity. Coverage and correct yield use all 108 planned window decisions. Token and duration columns sum recorded resources for the calls a policy requires, rather than actual savings from stopping collection.",
        "tab:canary_stability_policies","lrrrrrr","Policy & Correct & Error & Withheld & Calls & Tokens & Seconds",
        [[name,p[k]["accepted_correct"],p[k]["accepted_error"],p[k]["rejected"],p[k]["requests"],f'{p[k]["known_tokens"]:,}',f'{p[k]["client_seconds_sum"]:.3f}'] for k,name in labels+[("Vote_abstain_tie","Vote, tie abstention")]])
    comparison_table = tab(r"Descriptive comparison on matching C/R arms only. Each period has six windows and 108 planned window decisions, but different focal and canary records; these are not paired source observations. The original hidden-swap H arm is excluded from this table.",
        "tab:canary_stability_periods","llrrrr","Period & Policy & Correct & Error & Withheld & Calls",
        [[period,name,data["totals"][k]["correct"],data["totals"][k]["error"],data["totals"][k]["withheld"],data["totals"][k]["calls"]] for period,data in [("29 Sep",old),("30 Sep",new)] for k,name in labels])
    error_table = tab(r"Every erroneous focal output retained by the later-period known-answer window policy. The eight errors are repeated predictions of two cycles from source block~43, not eight distinct cases. W denotes weak leakage and N no leakage.",
        "tab:canary_stability_errors","llllrrl","Arm & Block & Selected call & Record ID & Cycle & Source block & Truth/pred.",
        [[r["arm"],r["block"],texid(r["selected"]),texid(r["id"]),r["cycle"],r["source_block"],"W/N"] for r in new["retained_errors"]])
    supplement = r"""\clearpage
\section{Prospectively fixed later-period screening follow-up}\label{app:canary_stability}
\subsection{Design, new sources and frozen rules}
This follow-up was designed after the original prospective canary study and the post hoc collapse analysis. Its own input records, prompts, chronological schedule, endpoints, thresholds, runner and scoring code were fixed before any follow-up prediction. The protocol was created at 05:06:42 UTC on 30 September 2026; first dispatch was 05:07:14 UTC. Every planned request completed by 05:20:33 UTC, and the audited response inventory was frozen at 05:21:00 UTC before answer-key scoring. These timestamps establish prospective fixation for this follow-up only; they do not retroactively preregister the earlier control.

The design uses the same 28 rounded hydraulic features and the same physical rig. All selected blocks and cycles are excluded from the three prior hydraulic source selections (the completed hydraulic stage reused its original panel). Nine new focal source blocks each supply two spaced stable cycles, selected at integer one-third and two-third positions in the stable-cycle list. Nine other new blocks each supply a middle stable cycle as a canary. These provide 18 focal and nine canary cycles, but only nine focal block units, not 18 independent sources. All 27 serialized feature vectors are distinct. Selection uses source annotations and a fixed hash, never new predictions. Both panels cover the nine pump-state by cooler-condition strata; the fixed hash assigns valve/accumulator settings 100/100 to the focal panel and 90/100 to the canary panel. These joint source-condition differences further limit causal attribution to any one change.

One fixed interleaving and anonymous IDs are used in every request. Roles and reference labels remain outside the inference inputs. There are two arms: C keeps canonical field order and R reverses intact descriptor--value pairs. Each of three chronological blocks contains three byte-identical calls per arm, in a seeded order, for 18 calls and 486 classification slots (324 focal outputs and 162 canary outputs). The runner waits for the entire block to finish before starting the next and permits at most two concurrent requests. Its isolation settings suppress project context, answer access, tools and web use. The requested model and reasoning setting remain GPT-5.6 Sol/medium; the client changes from 0.155.0-alpha.9.2 to 0.159.0. Source panel, reference panel, service period and client version change together; their effects cannot be separated.

The known-answer rule accepts at least 8/9 correct canaries. The label-free rule rejects a call if at least 90\% of all 27 returned labels agree (25 or more); fixed sensitivity thresholds are 85\%, 95\% and 100\%. Each policy uses the first passing call in its arm/block window, with at most three requests and complete withholding if none passes. Single uses the first scheduled call. The three-call vote takes the modal focal label and breaks an all-label tie using the first call, with tie abstention reported separately. Invalid calls cannot pass a gate, and incomplete voting withholds the affected outputs. All planned windows retain their denominator. No threshold is adjusted after outcomes are seen. Every scheduled call is collected regardless of gating; policy costs and adaptive decisions are replayed offline.

The session uses the existing subscription CLI without a paid API. Limits are 18 jobs, one invocation per job, two workers, a 900-second per-call timeout and a 3,600-second session budget; no new request starts when less than one full timeout remains. A 600,000 observed input-plus-output-token threshold stops further dispatch, with up to two in-flight calls potentially exceeding it. Unknown usage, request failure or quota failure also stops dispatch; none occurred. There were no retries, replacements, invalid responses or unattempted jobs. The original usage-based estimate was 402,481 tokens; actual use was 408,408. Summed client duration was 1,377.925 seconds; it exceeds elapsed session time because requests overlap.

\subsection{Complete request and window results}
"""+call_table+"\n\n"+window_table+r"""

Integer schedule positions agree with dispatch timestamps in each arm/block window, and all chronological block barriers hold. An independent reconstruction reads the original response arrays, checks them against the saved parsed arrays and recomputes the rules directly; it reproduces every primary window's output counts, selected request and required resources. This check is separate from the unchanged, pre-execution-frozen scorer.

"""+policy_table+r"""

Known-answer screening yields 100/108 correct window decisions (92.6\%) with eight errors and no withholding, versus 88/108 correct (81.5\%) and 20 errors for Single and the collapse gate. Its accepted-case error is 8/108 (7.4\%); all three policies have 100\% decision coverage. The ordinary vote yields 84 correct and 24 errors; tie abstention instead yields 84 correct, 22 errors and two withheld, for 98.1\% coverage and 22/106 (20.8\%) accepted-case error. Known-answer screening requires nine calls in replay (1.50 per window), versus six for Single/collapse and 18 for either vote. The actual experiment collected all 18 calls, so these are not realized savings from adaptive collection, monetary costs or end-to-end latency. Summed client durations likewise do not equal wall-clock time.

The gain is uneven. C gains ten correct decisions (38/54 to 48/54), entirely in block~2; R gains two (50/54 to 52/54), entirely in block~3. The remaining four windows use the same first call and show no change. Aggregated over both arms, the three temporal blocks change from 34, 24 and 30 correct decisions under Single to 34, 34 and 32 under known-answer screening. This supports a descriptive gain in this session, not consistent improvement in every temporal block or a calibrated population effect.

"""+cross_table+r"""

Every call passes all four pre-fixed collapse thresholds because modal counts range only from 9 to 15 of 27. Thus collapse screening reproduces Single at 85\%, 90\%, 95\% and 100\%. The known-answer rule passes eight calls and rejects ten. Its passing calls contain 134 correct and ten incorrect focal outputs among 144 (6.9\% error), compared with 101 correct and 79 incorrect among 180 rejected-call outputs (43.9\% error). Across all 18 calls there are 235 correct and 89 incorrect focal outputs among 324. These call-level quantities differ from the selected-window outcomes: call acceptance is 8/18, whereas every window contains a passing call. Ten errors across all passing calls and eight selected-window errors are different denominators and must not be conflated.

\subsection{Retained errors and comparison with the earlier period}
"""+error_table+r"""

All eight retained errors concern the same two weak-leakage cycles, 552 and 555 from source block~43, labelled no leakage in the selected calls for C/blocks~1--3 and R/block~3. The selected calls in R/blocks~1--2 classify both correctly. This is a direct description of residual source-specific failure; it does not identify a sensor, mechanism or model reasoning cause. Passing the reference threshold therefore leaves a repeated failure on one source block.

"""+comparison_table+r"""

The matching-arm comparison restricts the original study to C/R, as specified before the new results; it never compares the old nine-window C/H/R denominator directly with the new six-window C/R denominator. On the old C/R windows, known-answer and collapse checks coincide (88 correct, two incorrect and 18 withheld). In the new period, only the known-answer check changes the first-call decisions, while accepting every window. Its greater coverage also retains more absolute errors than in the earlier period. The panels are different and the observations are dependent; these descriptive differences cannot be attributed separately to time, the client update, source selection or reference selection.

The evidence is narrower than a general screening guarantee. Reference labels provide useful discrimination beyond the fixed collapse check for two of these six windows, and the same unchanged threshold improves correct yield in both arms in aggregate. That is one same-rig follow-up with nine focal block units and three short chronological blocks. It does not establish performance across devices, future service versions or a population of operating conditions; no independent-record $p$-value or population confidence interval is assigned. All prompts retain the canary records, so the comparison addresses the incremental use of known labels at the decision stage, not removal of canaries from the prompt or the validity of their physical labels.

\subsection{Reproduction and provenance}
The complete protocol, source exclusions, input and code hashes, raw and parsed responses, dispatch timestamps, usage, output-freeze inventory and final tables are under \nolinkurl{results/canary_stability_v2}. The pre-execution-frozen scorer is \nolinkurl{src/analysis/canary_controls_v2.py}; its interpretation flag distinguishes the prospective follow-up from the earlier post hoc control. The full protocol checksum is retained in \nolinkurl{results/canary_stability_v2/protocol.sha256}. Offline reproduction requires the retained local artifacts and makes no model calls:
\begin{quote}\small\ttfamily\raggedright
python src/experiments/run\_hydraulic\_v1.py audit --root results/canary\_stability\_v2\\
python src/analysis/canary\_controls\_v2.py --root results/canary\_stability\_v2 --out results/canary\_stability\_v2/analysis
\end{quote}
Eight gate/source-selection tests and ten inherited runner tests passed before protocol preparation. Those tests use mocked execution and do not constitute additional service calls. Later direct-raw recomputation, restricted earlier-period comparison and rendering are retained as subsequent verification, not changes to the frozen scoring rules.
\FloatBarrier
"""
    (OUT/"supp_canary_stability.tex").write_text(supplement)
    print("Wrote canary_stability_main.tex and supp_canary_stability.tex; frozen source artifacts unchanged.")

if __name__ == "__main__":
    render()
