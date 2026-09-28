"""Build synchronized manuscripts and five figures from verified frozen v3 results."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/revision_v3"
PAPER = ROOT / "paper"
MODELS = ["gpt-5.5", "gpt-5.6-sol", "gpt-6-astra"]
NAMES = {"gpt-5.5": "GPT-5.5", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-6-astra": "GPT-6 Astra",
         "RF": "Random Forest", "SVM": "RBF SVM", "Majority": "Training majority"}
DS = {"cmapss": "C-MAPSS", "tep": "TEP"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(x):
    if x is None:
        return 'N/A'
    return f"{100*x:.1f}"


def interval(value):
    if value is None or any(x is None for x in value):
        return 'N/A'
    return "[" + ", ".join(percent(x) for x in value) + "]"


def span(values):
    low, high = min(values), max(values)
    return percent(low) if low == high else percent(low) + '--' + percent(high)


def latex_table(caption, label, cols, header, rows, small=True):
    return ("\\begin{table}[htbp]\n\\centering\n" + ("\\small\n" if small else "")
            + "\\caption{" + caption + "}\n\\label{" + label + "}\n"
            + "\\begin{tabular}{@{}" + cols + "@{}}\n\\toprule\n"
            + " & ".join(header) + " \\\\\n\\midrule\n"
            + "\n".join(" & ".join(row) + " \\\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def latex_figure(filename, label, caption):
    width = '0.85\\linewidth' if filename == 'class_recall' else '\\linewidth'
    return ("\\begin{figure}[htbp]\n\\centering\n"
            + "\\includegraphics[width=" + width + "]{figures/revision_v3/" + filename + ".pdf}\n"
            + "\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{figure}")


def main():
    score_path = BASE / "analysis/scores.json"
    report = json.loads(score_path.read_text())
    if report["audit"]["status"] != "verified" or report["audit"]["planned_sessions"] != 27:
        raise ValueError("Only all 27 audited scheduled outcomes can generate this manuscript")
    manifest = BASE / "predictions/frozen_manifest.json"
    if sha(manifest) != report["audit"]["frozen_manifest_sha256"]:
        raise ValueError("Analysis does not refer to the current frozen prediction manifest")
    frozen = json.loads(manifest.read_text())
    for name, digest in frozen["files"].items():
        if sha(BASE / "predictions" / name) != digest:
            raise ValueError(f"Prediction provenance changed: {name}")
    if sha(BASE / "data/datasets.json") != report['evaluation_provenance']['datasets_sha256']:
        raise ValueError('Dataset differs from the scored evaluation')
    if sha(BASE / "baselines/predictions.json") != report['evaluation_provenance']['baseline_predictions_sha256']:
        raise ValueError('Baseline predictions differ from the scored evaluation')
    baseline_manifest = json.loads((BASE / 'baselines/manifest.json').read_text())
    for name, digest in baseline_manifest['files'].items():
        if sha(BASE / 'baselines' / name) != digest:
            raise ValueError(f'Baseline provenance changed: {name}')
    data = json.loads((BASE / "data/datasets.json").read_text())["datasets"]
    baseline = json.loads((BASE / "baselines/scores.json").read_text())
    rows = report["metrics"]

    def metric(model, dataset, condition="clean", repeat=1):
        return next(x for x in rows if x["model"] == model and x["dataset"] == dataset
                    and x["condition"] == condition and x["repeat"] == (repeat if model in MODELS else None))

    paired = [x for x in report["paired_contrasts"] if x["contrast"] == "clean_minus_descriptor_mapping_fault"]
    def contrast(model, repeat=1):
        return next(x for x in paired if x["a"].startswith(model + "/") and x["repeat"] == repeat)

    cm = [metric(m, "cmapss") for m in MODELS if metric(m,'cmapss')['available']]
    tep = [metric(m, "tep") for m in MODELS if metric(m,'tep')['available']]
    corrupt = [metric(m, "tep", "misnamed") for m in MODELS]
    contrasts = [contrast(m) for m in MODELS]
    text = {}
    text["ABSTRACT_RESULTS"] = (
        "Primary TEP accuracy losses are "
        f"{percent(contrasts[0]['accuracy_difference'])} percentage points for GPT-5.5 "
        f"(conditional 95\\% interval {interval(contrasts[0]['accuracy_difference_group_bootstrap95'])}) and "
        f"{percent(contrasts[1]['accuracy_difference'])} for GPT-5.6 Sol "
        f"({interval(contrasts[1]['accuracy_difference_group_bootstrap95'])}); GPT-6 Astra has unchanged labels."
    )
    text['WORKFLOW_FIGURE'] = latex_figure(
        'evaluation_workflow', 'fig:workflow',
        "Original-study evaluation workflow and the symbolic TEP descriptor rotation. Source units are disjoint across training and testing; LLM outputs share a grouped context. "
        "Supervised methods fit labeled training data, while LLMs receive task definitions and sensor descriptions. "
        "The intervention shifts complete descriptors by one position while retaining the numerical sequence; it does not alter C-MAPSS inputs.")
    text['PRIMARY_FIGURE'] = latex_figure(
        'primary_accuracy', 'fig:primary',
        "Original-study primary clean-input accuracy with 95\\% Wilson intervals on the common test observations. N/A denotes the failed whole-batch response. "
        "LLM values use repeat 1; RF and SVM are each fit once. Intervals use the independent-source-unit approximation and are conditional on the observed grouped responses. "
        "The training-majority references are given in Table~\\ref{tab:primary}.")
    text['MAPPING_FIGURE'] = latex_figure(
        'mapping_sensitivity', 'fig:mapping',
        "TEP sensitivity to the fixed descriptor rotation. Left: clean-minus-corrupted accuracy differences, with paired group-bootstrap 95\\% intervals for repeat 1 and separate points for repeats 2 and 3. "
        "Right: the proportion of labels that change between paired conditions in each repeat. "
        "All comparisons use the same 70 runs; repetitions are separate sessions. Zero effect and zero flips for Astra describe this input set, not equivalence under other corruptions.")
    text['CLASS_FIGURE'] = latex_figure(
        'class_recall', 'fig:recall',
        "Descriptive class-level recall from the frozen primary responses. Cells show recall (\\%); supports are 20 engines per C-MAPSS category and 10 runs per TEP condition. "
        "N/A retains the invalid C-MAPSS batch. The paired TEP rows compare clean and rotated-descriptor conditions. "
        "Small class denominators make these error profiles exploratory; full confusion counts and intervals are supplied with the analysis.")
    text["DATA_TABLE"] = latex_table(
        "Evaluation units and representations. Test counts refer to unique engines or run IDs, not repeated LLM outputs. TEP training contains seven scenarios per selected run ID.",
        "tab:data", "lrrrr", ["Task", "Features", "Train records", "Train groups", "Test groups"],
        [[DS[name], str(len(ds['feature_names'])), str(len(ds['train'])), str(len({r['group'] for r in ds['train']})), str(len(ds['test']))] for name, ds in data.items()])
    audit = report['audit']
    text["COVERAGE"] = (
        f"All {audit['planned_sessions']} scheduled outcomes are retained: {audit['completed_sessions']} complete valid responses and "
        f"{audit['failed_sessions']} response-validation {'failure' if audit['failed_sessions']==1 else 'failures'}, yielding {audit['total_valid_predictions']:,} labels eligible for classification scoring "
        f"from {audit['expected_predictions']:,} requested classifications. The logs contain {audit['tools_observed']} tool events "
        f"and {audit['diagnostic_errors_count']} diagnostic {'event' if audit['diagnostic_errors_count']==1 else 'events'}. The actual run used {audit['total_attempt_sessions']} attempt sessions. "
        "No response was manually corrected or selected by its score. Invalid batches remain visible as N/A, not as zero accuracy, and no failed primary is replaced by a later repeat. "
        "Table~\\ref{tab:primary} reports the primary first-session results and the common-test baselines; Figure~\\ref{fig:primary} visualizes their accuracy intervals."
    )
    for failure in report['session_failures']:
        text['COVERAGE'] += (f" {NAMES[failure['model']]} on {DS[failure['dataset']]} repeat {failure['repeat']} "
                             f"returned {failure['actual_raw_count']} objects where {failure['expected_n']} were required. "
                             "The whole batch failed the completeness rule; its partial labels are not scored.")
    if audit.get('recovered_stream_reconnections'):
        text['COVERAGE'] += (" The logged stream reconnection was a recovered transport event, "
                             "followed by a complete valid response in the same CLI session; it is retained in the audit rather than treated as a classification failure.")
    table_rows = []
    for dataset in ['cmapss', 'tep']:
        for model in MODELS + ['RF', 'SVM', 'Majority']:
            r = metric(model, dataset)
            table_rows.append([DS[dataset], NAMES[model], f"{r['correct']}/{r['n']}" if r['available'] else 'N/A', percent(r['accuracy']), interval(r['accuracy_wilson95']), percent(r['macro_f1'])])
    text['PRIMARY_TABLE'] = latex_table(
        "Original-study clean-input performance on the identical held-out observations. Accuracy and macro F1 are percentages. LLMs use the prespecified repeat 1; supervised models are fit once. N/A indicates failed batch validation, not zero accuracy. Accuracy intervals are 95\\% Wilson summaries under the independent-source-unit approximation.",
        "tab:primary", "llrrrr", ["Task", "Method", "Correct", "Acc.", "95\\% interval", "Macro F1"], table_rows)
    descriptions = []
    for dataset in ['cmapss', 'tep']:
        values = ', '.join(f"{NAMES[m]} {percent(metric(m,dataset)['accuracy'])}\\%" if metric(m,dataset)['available'] else f"{NAMES[m]} unavailable" for m in MODELS)
        descriptions.append(f"For {DS[dataset]}, first-session LLM accuracies are {values}. "
                            f"The matched RF and SVM reach {percent(metric('RF',dataset)['accuracy'])}\\% and "
                            f"{percent(metric('SVM',dataset)['accuracy'])}\\%, respectively.")
    perfect_tep_baselines = all(metric(m,'tep')['correct'] == metric(m,'tep')['n'] for m in ['RF','SVM'])
    text['PRIMARY_RESULTS'] = '\n\n'.join(descriptions) + (
        "\n\nUniform and training-prior random guessing have expected accuracies of "
        f"{percent(baseline['datasets']['cmapss']['uniform_random_expected_accuracy_pct']/100)}\\% on C-MAPSS and "
        f"{percent(baseline['datasets']['tep']['uniform_random_expected_accuracy_pct']/100)}\\% on this balanced TEP test subset. "
        "These reference expectations do not demonstrate an operationally useful classifier. "
        + ("The perfect TEP supervised result has a Wilson interval with a lower bound below 100\\%; it does not imply zero error on future runs."
           if perfect_tep_baselines else "The supervised results describe this held-out subset and should not be extrapolated to future operating regimes.")
    )
    table_rows = []
    for model in MODELS:
        a, b, c = metric(model, 'tep'), metric(model, 'tep', 'misnamed'), contrast(model)
        table_rows.append([NAMES[model], percent(a['accuracy']), percent(b['accuracy']), percent(c['accuracy_difference']), interval(c['accuracy_difference_group_bootstrap95']), f"{c['flip_count']}/{c['n']}"])
    text['PAIRED_TABLE'] = latex_table(
        "Paired TEP mapping stress test, repeat 1. Accuracy and differences are in percent and percentage points, respectively. Positive differences favor clean inputs. Intervals resample paired source groups conditional on the fixed batch responses. Flips count changes in predicted labels.",
        "tab:paired", "lrrrrr", ["Model", "Clean", "Corrupted", "$\\Delta$ acc.", "95\\% paired interval", "Flips"], table_rows)
    text['PAIRED_RESULTS'] = (
        "Table~\\ref{tab:paired} shows the primary paired contrast, while Figure~\\ref{fig:mapping} places it alongside the repeated effects and prediction flips. "
        + ' '.join(f"For {NAMES[m]}, macro F1 changes from {percent(metric(m,'tep')['macro_f1'])}\\% to "
                   f"{percent(metric(m,'tep','misnamed')['macro_f1'])}\\% under the descriptor rotation "
                   f"(clean-minus-corrupted difference {percent(contrast(m)['macro_f1_difference'])} pp)." for m in MODELS)
        + " Full accuracy, F1, flip-rate intervals, and all repeated paired contrasts are retained in the machine-readable analysis."
    )
    repeat_rows = []
    for dataset, condition in [('cmapss','clean'), ('tep','clean'), ('tep','misnamed')]:
        for model in MODELS:
            rr = [metric(model, dataset, condition, repeat) for repeat in [1,2,3]]
            available = [r['accuracy'] for r in rr if r['available']]
            repeat_rows.append([DS[dataset] + (' corrupted' if condition == 'misnamed' else ' clean'), NAMES[model],
                                *[percent(r['accuracy']) for r in rr], percent(max(available)-min(available)) if len(available)>1 else 'N/A', str(len(available))])
    text['REPEAT_TABLE'] = latex_table(
        "Accuracy (\\%) for scheduled fresh sessions with identical within-condition payloads. Range is maximum minus minimum over the valid sessions, in percentage points. Valid gives their count. N/A is retained for failed validation; repetitions do not increase the held-out sample size.",
        "tab:repeat", "llrrrrr", ["Task/condition", "Model", "R1", "R2", "R3", "Range", "Valid"], repeat_rows)
    ranges = [x['accuracy']['range'] for x in report['repeat_variability'] if x['accuracy']['range'] is not None]
    text['REPEAT_RESULTS'] = (
        f"The available model--condition accuracy ranges span {percent(min(ranges))}--{percent(max(ranges))} pp. "
        "Ranges based on two valid sessions should not be compared with three-session ranges as a stability ranking. "
        "These are observed ranges across three sessions, not confidence intervals over the service's future behavior. "
        "A stable aggregate score may still conceal different sample-level decisions. Figure~\\ref{fig:repeat} displays all available scheduled-session accuracies alongside the common-test baselines."
    )
    repeat_flip_rates = [pair['flip_rate'] for row in report['repeat_variability'] for pair in row['prediction_disagreement']]
    text['REPEAT_RESULTS'] += (f" Across the {len(repeat_flip_rates)} valid within-condition pairs of repeated sessions, prediction disagreement ranges from "
                              f"{percent(min(repeat_flip_rates))}\\% to {percent(max(repeat_flip_rates))}\\%. "
                              "These compare repeated predictions of the same observations; they are distinct from the clean-versus-corrupted flips in Table~\\ref{tab:paired}.")
    error_rows = []
    for model in MODELS + ['RF', 'SVM']:
        a, b = metric(model, 'cmapss'), metric(model, 'tep')
        normal = b['per_class']['Normal']
        error_rows.append([NAMES[model], f"{a['miss_numerator']}/{a['miss_denominator']}" if a['available'] else 'N/A', percent(a['miss_rate']),
                           f"{b['miss_numerator']}/{b['miss_denominator']}", percent(b['miss_rate']),
                           f"{normal['fn']}/{normal['support']}"])
    text['ERROR_TABLE'] = latex_table(
        "Operational error summaries for clean inputs, repeat 1 for LLMs. C-MAPSS miss means a Critical engine assigned either other class. TEP miss means a fault assigned Normal; false alarm means a Normal observation assigned a fault. These denominators differ from overall accuracy.",
        "tab:errors", "lrrrrr", ["Method", "Critical miss", "Rate (\\%)", "Fault miss", "Rate (\\%)", "False alarm"], error_rows)
    text['ERROR_RESULTS'] = (
        "Table~\\ref{tab:errors} separates fault-to-Normal errors from confusion among fault types; Figure~\\ref{fig:recall} shows all class-level recalls. "
        "Only ten Normal TEP observations and twenty Critical C-MAPSS engines enter the respective class-specific denominators. "
        "Their intervals and full confusion matrices are provided with the analysis; zero observed misses in this subset should not be interpreted as a safety guarantee."
    )
    sol, forest = metric('gpt-5.6-sol','cmapss'), metric('RF','cmapss')
    if sol['available']:
        text['ERROR_RESULTS'] += (f" Overall accuracy and Critical recall need not rank methods identically: GPT-5.6 Sol misses "
                                 f"{sol['miss_numerator']}/{sol['miss_denominator']} Critical engines, compared with "
                                 f"{forest['miss_numerator']}/{forest['miss_denominator']} for RF in the primary comparison. "
                                 "This small class-specific result is descriptive and does not establish a safety advantage.")
    clean_f2 = metric('gpt-5.5', 'tep')['per_class']['Fault_2']
    corrupt_f2 = metric('gpt-5.5', 'tep', 'misnamed')['per_class']['Fault_2']
    text['ERROR_RESULTS'] += (
        f"\n\nThe mapping intervention can have opposite effects across fault classes. In GPT-5.5's primary responses, Fault 2 recall changes from "
        f"{clean_f2['tp']}/{clean_f2['support']} to {corrupt_f2['tp']}/{corrupt_f2['support']} under the rotation, "
        "even though overall accuracy falls. This descriptive pattern helps explain why net accuracy change and prediction disagreement are complementary. "
        "The case exports retain each transition for follow-up analysis.")
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=True)
    colors = {'clean':'#1b526b','misnamed':'#c06926'}
    for ax, dataset in zip(axes, ['cmapss','tep']):
        conditions = ['clean'] if dataset == 'cmapss' else ['clean','misnamed']
        for condition in conditions:
            shift = (-.12 if condition == 'clean' else .12) if dataset == 'tep' else 0
            for repeat, offset in zip([1,2,3],[-.035,0,.035]):
                ax.scatter(np.arange(3)+shift+offset, [100*metric(m,dataset,condition,repeat)['accuracy'] if metric(m,dataset,condition,repeat)['available'] else np.nan for m in MODELS],
                           color=colors[condition], marker=['o','s','^'][repeat-1], s=38, alpha=.9,
                           label=('Clean' if condition=='clean' else 'Mapping fault') if repeat==1 else None, zorder=4)
        for model, style, color in [('RF','--','#758369'),('SVM',':','#999999')]:
            ax.axhline(100*metric(model,dataset)['accuracy'],linestyle=style,color=color,linewidth=1.2,label=NAMES[model])
        tick_names = []
        for model in MODELS:
            missing = [str(r) for r in [1,2,3] if not metric(model,dataset,'clean',r)['available']]
            tick_names.append(NAMES[model] + ('\nN/A: R'+','.join(missing) if missing else ''))
        ax.set_xticks(range(3), tick_names,fontsize=9)
        ax.set_ylim(-3,105); ax.set_xlim(-.45,2.45); ax.set_ylabel('Accuracy (%)');ax.set_title(DS[dataset])
        ax.grid(axis='y',alpha=.15); ax.legend(loc='lower left',fontsize=8,frameon=True,framealpha=.93,ncol=2)
    figures = PAPER/'figures/revision_v3'; figures.mkdir(parents=True,exist_ok=True)
    fig.savefig(figures/'repeat_accuracy.pdf');fig.savefig(figures/'repeat_accuracy.png',dpi=220);plt.close(fig)
    figure_data = BASE/'analysis/figure_data'; figure_data.mkdir(parents=True, exist_ok=True)
    with (figure_data/'repeat_accuracy.csv').open('w', newline='') as stream:
        fields = ['model','dataset','condition','repeat','available','n','n_scored','correct','accuracy_pct','scores_sha256']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            if row['model'] == 'Majority':
                continue
            writer.writerow({**{k:row[k] for k in fields[:-2]},
                             'accuracy_pct':None if row['accuracy'] is None else 100*row['accuracy'],
                             'scores_sha256':sha(score_path)})
    text['FIGURE'] = (
        "\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/revision_v3/repeat_accuracy.pdf}\n"
        "\\caption{All available scheduled-session accuracies (circles, squares, triangles for repeats 1, 2, 3). N/A marks the failed batch. Points may overlap. Horizontal lines show the common-test RF and SVM baselines. "
        + ("Their TEP lines coincide. " if metric('RF','tep')['accuracy'] == metric('SVM','tep')['accuracy'] else "")
        + "This plot shows repeated scores, not independent test replications.}\n\\label{fig:repeat}\n\\end{figure}"
    )
    all_effects = [contrast(m, r)['accuracy_difference'] for m in MODELS for r in [1,2,3]]
    if all(effect == 0 for effect in all_effects):
        mapping_statement = "The clean and corrupted TEP conditions have equal accuracy in all nine model--repeat comparisons; equal scores do not imply identical predictions."
    elif min(all_effects) > 0:
        mapping_statement = "Clean TEP accuracy exceeds the corrupted condition in all nine model--repeat comparisons."
    elif max(all_effects) < 0:
        mapping_statement = "Corrupted TEP accuracy exceeds the clean condition in all nine model--repeat comparisons. This unexpected result warrants investigation rather than a claim that incorrect mappings are beneficial in general."
    elif min(all_effects) >= 0:
        mapping_statement = "Clean TEP accuracy is at least as high as corrupted accuracy in all nine model--repeat comparisons, with ties in some sessions."
    elif max(all_effects) <= 0:
        mapping_statement = "Corrupted TEP accuracy is at least as high as clean accuracy in all nine model--repeat comparisons, with ties in some sessions. This does not establish a general benefit of incorrect mappings."
    else:
        mapping_statement = "The direction of the TEP accuracy contrast is not uniform across all nine model--repeat comparisons. A single aggregate difference therefore does not support a universal claim about mapping errors."
    text['INTERPRETATION'] = (
        mapping_statement + " The complete contrast set is retained regardless of its direction. "
        + ("The supervised baselines already classify all selected late-stage TEP snapshots correctly, leaving no measured accuracy advantage for the LLM interface on this subset. " if perfect_tep_baselines
           else "The supervised comparison uses the same selected late-stage TEP snapshots; differences do not establish superiority beyond that test set. ")
        +
        "Its possible advantages in explanation, interaction, or transfer are separate questions that this label-only experiment does not measure."
    )
    zero_flip_models = [m for m in MODELS if all(contrast(m,r)['flip_count'] == 0 for r in [1,2,3])]
    if zero_flip_models:
        text['INTERPRETATION'] += ("\n\n" + ', '.join(NAMES[m] for m in zero_flip_models)
            + " gives identical clean and corrupted predictions in all three paired repetitions. This contrasts with the responses of the other evaluated services and prevents a claim that the mapping fault universally reduces accuracy. "
              "Invariance under an incorrect descriptor mapping also does not prove that descriptors are used correctly. Reliance on retained numerical patterns, canonical variable order, or recognition and compensation for an obvious mapping shift are possible explanations; this experiment does not distinguish them. "
              "The result motivates separate tests of value order and individual fields rather than an unsupported claim about internal reasoning or benchmark memorization.")
    text['CONCLUSION_RESULTS'] = (
        f"The original primary descriptor rotation reduces TEP accuracy by {percent(contrasts[0]['accuracy_difference'])} "
        f"and {percent(contrasts[1]['accuracy_difference'])} percentage points for GPT-5.5 and GPT-5.6 Sol, "
        "while GPT-6 Astra preserves its labels in all three original paired sessions."
    )
    for script in ['draw_revision_v3_workflow.py', 'plot_revision_v3_diagnostics.py']:
        subprocess.run([sys.executable, str(ROOT/'src/analysis'/script)], check=True)
    # Optional post-v3 extension is generated only from fully frozen, audited results.
    extension_script = ROOT/'src/analysis/render_mechanism_extension.py'
    subprocess.run([sys.executable, str(extension_script)], check=True)
    for key, value in json.loads((PAPER/'mechanism_extension_text.json').read_text()).items():
        text[key] = value
    subprocess.run([sys.executable, str(ROOT/'src/analysis/render_domestic_extension.py')], check=True)
    text.update(json.loads((PAPER/'domestic_extension_text.json').read_text()))
    subprocess.run([sys.executable, str(ROOT/'src/analysis/render_claude_reported_summary.py')], check=True)
    text.update(json.loads((PAPER/'claude_reported_text.json').read_text()))
    subprocess.run([sys.executable, str(ROOT/'src/analysis/summarize_usable_predictions.py')], check=True)
    text['USABLE_RESULTS'] = (ROOT/'results/cross_provider_revision/usable_predictions/paper_insert.tex').read_text()
    subprocess.run([sys.executable, str(ROOT/'src/analysis/render_independent_validation.py')], check=True)
    text.update(json.loads((PAPER/'independent_validation_text.json').read_text()))
    template = (PAPER/'revision_v3_template.tex').read_text()
    markers = set(re.findall(r'@@([A-Z_]+)@@', template))
    if markers != set(text):
        raise ValueError(f"Template/render keys mismatch: {markers ^ set(text)}")
    for marker, value in text.items():
        template = template.replace('@@'+marker+'@@',value)
    bibkeys=set(re.findall(r'@\w+\{([^,]+),',(PAPER/'references.bib').read_text()))
    cited=set(key.strip() for group in re.findall(r'\\cite[pt]?\{([^}]+)\}',template) for key in group.split(','))
    if cited-bibkeys:
        raise ValueError(f"Undefined bibliography keys: {sorted(cited-bibkeys)}")
    for filename in ['main_inline.tex','main.tex','main_compile.tex']:
        (PAPER/filename).write_text(template)
    provenance = {'analysis_sha256':sha(score_path),'template_sha256':sha(PAPER/'revision_v3_template.tex'),
                  'references_sha256':sha(PAPER/'references.bib'),
                  'renderer_sha256':sha(Path(__file__)),'main_inline_sha256':sha(PAPER/'main_inline.tex'),
                  'main_sha256':sha(PAPER/'main.tex'),'main_compile_sha256':sha(PAPER/'main_compile.tex'),'frozen_predictions_manifest_sha256':sha(manifest),
                  'all_template_markers_resolved':True,'all_citation_keys_defined':True}
    extension_scores = ROOT/'results/mechanism_extension/analysis/scores.json'
    provenance['extension_files'] = {}
    if extension_scores.exists():
        for name in ['results/mechanism_extension/protocol.json',
                     'results/mechanism_extension/predictions/frozen_manifest.json',
                     'results/mechanism_extension/analysis/scores.json',
                     'src/analysis/render_mechanism_extension.py',
                     'paper/mechanism_extension_text.json',
                     'src/analysis/plot_mechanism_context.py',
                     'paper/figures/mechanism_extension/matched_context.pdf',
                     'paper/figures/mechanism_extension/matched_context.png',
                     'results/mechanism_extension/analysis/context_figure_data.csv']:
            provenance['extension_files'][name] = sha(ROOT/name)
    provenance['domestic_files'] = {name:sha(ROOT/name) for name in [
        'results/domestic_extension_v2/plan.json',
        'results/domestic_extension_v2/frozen_manifest.json',
        'results/domestic_extension_v2/analysis/scores.json',
        'src/analysis/audit_domestic_extension.py',
        'src/analysis/render_domestic_extension.py',
        'paper/domestic_extension_text.json']}
    provenance['claude_reported_files'] = {name:sha(ROOT/name) for name in [
        'results/claude_descriptive_recheck_2026-09-28/snapshot_manifest.json',
        'results/claude_descriptive_recheck_2026-09-28/descriptive_metrics.json',
        'results/claude_conversation_exploratory_v1/audit_2026-09-28/snapshot_manifest.json',
        'results/claude_conversation_exploratory_v1/audit_2026-09-28/corrected_descriptive_scores.json',
        'src/analysis/recheck_claude_descriptive.py',
        'src/analysis/audit_conversation_exploratory.py',
        'src/analysis/render_claude_reported_summary.py',
        'paper/claude_reported_text.json']}
    provenance['figure_builders'] = {name:sha(ROOT/'src/analysis'/name)
                                    for name in ['draw_revision_v3_workflow.py','plot_revision_v3_diagnostics.py']}
    provenance['usable_output_files'] = {name:sha(ROOT/name) for name in [
        'src/analysis/summarize_usable_predictions.py',
        'results/cross_provider_revision/usable_predictions/summary.json',
        'results/cross_provider_revision/usable_predictions/summary.csv',
        'results/cross_provider_revision/usable_predictions/paper_insert.tex']}
    provenance['independent_files'] = {name:sha(ROOT/name) for name in [
        'results/independent_validation/protocol.json',
        'results/independent_validation/data_audit.json',
        'results/independent_validation/predictions/frozen_manifest.json',
        'results/independent_validation/analysis/scores.json',
        'results/independent_validation/analysis/cases.csv',
        'results/independent_validation/analysis/figure_data.csv',
        'src/analysis/render_independent_validation.py',
        'paper/independent_validation_text.json',
        'paper/figures/independent_validation/independent_controls.pdf',
        'paper/figures/independent_validation/independent_controls.png']}
    provenance['figure_files'] = {str(p.relative_to(ROOT)):sha(p) for p in sorted(figures.glob('*')) if p.suffix in {'.pdf','.png'}}
    provenance['figure_data_files'] = {str(p.relative_to(ROOT)):sha(p) for p in sorted(figure_data.glob('*.csv'))}
    (BASE/'analysis/paper_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Generated synchronized manuscripts, original-study figures and audited follow-up inserts from verified scores.')


if __name__ == '__main__':
    main()
