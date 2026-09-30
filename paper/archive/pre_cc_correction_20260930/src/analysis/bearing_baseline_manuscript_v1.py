"""Source-bound reporting of the separately frozen supervised bearing reference."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/bearing_baseline_v1'

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def esc(x): return str(x).replace('_', r'\_')
def pct(x): return f'{100*x:.1f}'
def table(caption, label, columns, heading, rows, note=''):
    lines=[r'\begin{table}[htbp]\centering\small', r'\caption{'+caption+'}', r'\label{'+label+'}', r'\begin{tabular}{@{}'+columns+r'@{}}\toprule', heading+r' \\\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule\end{tabular}']
    if note: lines += [r'\par\smallskip\footnotesize '+note]
    return '\n'.join(lines+[r'\end{table}'])

METHODS = r'''\subsection{Same-feature supervised bearing reference}
After the bearing language-model outcomes were known, a separately frozen follow-up retained their 36 focal recordings from nine physical bearings as the test set. Its development data use only the six original background specimens: K001/K004 (healthy), KA15/KA22 (outer-race fault) and KI16/KI18 (inner-race fault). Each contributes five complete recordings under each of four operating regimes, selected by the existing outcome-independent hash ordering, giving 120 development records. Twelve earlier background recordings are reused and 108 are added. The test specimens remain K002/K003/K005, KA04/KA16/KA30 and KI04/KI14/KI21. No physical bearing or raw recording crosses the development/test boundary; increasing repeated recordings adds no physical specimens.

Both sides use exactly the original 28 numerical features, column order and six-decimal serialization. Test strings and record membership are unchanged; development features use the same full-record formulas. No test-bearing windows, additional test recordings, frequency-band selection or outcome-based source substitution is introduced. Bearing identity, regime code, filename and reference label are excluded from predictor features. This is a result-informed diagnostic benchmark, not an untouched prospective test: the test labels are excluded from conventional fitting and tuning, but prior language-model outcomes motivated the follow-up. The conventional classifiers receive development labels and more development recordings; the language-model calls did not receive those labels. Equal numerical representation does not equalize supervision, prior exposure or resources.

An RBF SVM uses training-fitted standardization and the six combinations of $C\in\{0.1,1,10\}$ and $\gamma\in\{\mathrm{scale},0.1\}$. A 200-tree random forest uses the four combinations of maximum features in $\{\mathrm{sqrt},1.0\}$ and minimum leaf size in $\{1,3\}$, with seed 20260928. For each class, one of its two development bearings is held out for validation and the other is used for training. All eight combinations are enumerated; each trains on three bearings/60 records and validates on the other three/60. Preprocessing is fitted inside each training fold. Mean validation accuracy selects parameters separately for each model; ties select the first candidate in the fixed ordering. The eight overlapping splits are not eight independent replications. Both classifiers are reported, with no model choice based on test results. Final primary fits use all 120 development records.

Four secondary fits per classifier jointly hold out the test specimens and one operating regime. Each removes that regime from all development data, retunes on the remaining 90 records with the same eight bearing-disjoint combinations (45 training/45 validation records), and tests only the nine original focal records from the omitted regime. No primary-fit parameters are carried over. Their combined predictions cover the same 36 records once, using four fitted models. The contrast with the primary fit changes regime support, training size and selected parameters; it is not an isolated causal effect of regime shift. A training-majority constant, with healthy first in ties, and the uniform-prior expected accuracy of one third provide simple references.

The data rules, finite grids, folds, code and input hashes are frozen before fitting; final test predictions and full development-selection outputs are frozen before test evaluation. All 12 existing bearing language-model calls are compared separately on those same 36 IDs. We report accuracy, macro F1, confusion matrices, class recall and each test bearing's correct count, without treating repeated records or calls as independent devices or adding a population confidence interval. Fixed-classifier row reversal, singleton prediction and appended-background checks verify software behavior, not learned physical semantics. Appendix G retains the split details, all selected configurations, specimen/regime results and individual language-model calls; all candidate/fold predictions are available in the reconstruction package.
'''


def make_baseline_fragments():
    p=read(BASE/'protocol.json'); s=read(BASE/'analysis/scores.json'); f=read(BASE/'prediction_manifest.json')
    assert s['protocol_sha256']==f['protocol_sha256']==sha(BASE/'protocol.json')
    assert s['prediction_manifest_sha256']==sha(BASE/'prediction_manifest.json')
    for rel,h in f['files_sha256'].items(): assert sha(BASE/'fit'/rel)==h,rel
    for rel,h in p['source_sha256'].items(): assert sha(ROOT/rel)==h,rel
    b=s['baseline']; selected=read(BASE/'fit/selections.json'); checks=read(BASE/'fit/structural_checks.json')
    assert all(all(x[k] for k in ['reversed_row_order','single_row_calls','unlabeled_background_appended']) for x in checks)
    rows=[]
    for task,label in [('primary','Seen regimes'),('loro_combined','Regime held out')]:
        for model in ['SVM','RF']:
            m=b[task+'/'+model];lo,hi=m['bearing_correct_count_range']
            rows.append([label,model,f"{m['correct']}/{m['total']}",pct(m['accuracy']),f"{m['macro_f1']:.3f}",f'{lo}--{hi}/4'])
    c=b['primary/Constant']; rows.append(['Either','Constant',f"{c['correct']}/{c['total']}",pct(c['accuracy']),f"{c['macro_f1']:.3f}",'0--4/4'])
    primary_table=table('Same-feature supervised reference on the nine fixed test bearings. Seen-regime fits use 120 records from six other bearings; regime-held-out fits use 90 records per omitted regime and separate retuning. All rows use the same 36 test recordings.','tab:bearing_baseline','llrrrr',r'Development & Model & Correct & Accuracy (\%) & Macro F1 & Per bearing',rows,'The four held-out-regime fits share specimens and training records. These are descriptive results on one rig; the constant tie rule predicts healthy. The frozen language-model calls have no corresponding supervised-development budget.')
    lm=s['llm_calls']; lr=[m['correct'] for m in lm.values()]
    result=[r'\subsection{Representation utility on held-out bearing specimens}',primary_table,
        f"On the unchanged test records, primary SVM and RF obtain {b['primary/SVM']['correct']}/36 and {b['primary/RF']['correct']}/36 correct diagnoses, respectively; the constant obtains {c['correct']}/36. With both specimens and operating regime held out, their combined counts are {b['loro_combined/SVM']['correct']}/36 and {b['loro_combined/RF']['correct']}/36. Individual-bearing outcomes are retained rather than treating the 36 records as 36 devices.",
        f"The 12 frozen language-model calls each evaluate the same 36 focal records and yield {min(lr)}--{max(lr)} correct labels per call. All F/R conditions, both rounds and all three repeats remain reported; the maximum is not selected as a replacement primary result. The supervised results provide a representation reference with additional development labels, not an equal-resource accuracy contest.",
        'All ten fitted classifier/task combinations preserve their own predictions under row reversal, single-row invocation and appended background records. This is expected for these fixed row-wise estimators; it is not evidence that they learned a physical invariance or that a language-model intervention effect has been isolated.']
    result.append(f"The selected primary development-fold means are {pct(next(x['cv_mean_accuracy'] for x in selected if x['task']=='primary' and x['model']=='SVM'))}\\% for SVM and {pct(next(x['cv_mean_accuracy'] for x in selected if x['task']=='primary' and x['model']=='RF'))}\\% for RF. These parameter-selection scores fail to carry over to the nine fixed test specimens. A post hoc reading of the prespecified class/specimen tables shows that primary SVM correctly identifies only one of 12 inner-race records; primary RF identifies none of the healthy or inner-race records, and all four of its correct predictions concern KA30. These localized outcomes limit interpretations of aggregate accuracy.")
    abstract=(f"Using the same 28 features and nine held-out bearing specimens, supervised SVM/RF references obtain {b['primary/SVM']['correct']}/36 and {b['primary/RF']['correct']}/36, below the constant's {c['correct']}/36 despite access to six other specimens' labels. This result-informed check leaves feature and learning limitations unresolved.")
    # Interpretative wording is completed after frozen scoring, not chosen by retuning.
    discussion=(ROOT/'paper/focused_submission/bearing_baseline_interpretation.tex').read_text()
    conclusion=(f"The same-feature bearing reference adds {b['primary/SVM']['correct']}/36 SVM and {b['primary/RF']['correct']}/36 RF correct predictions on held-out specimens, with {b['loro_combined/SVM']['correct']}/36 and {b['loro_combined/RF']['correct']}/36 when development also excludes the target regime. This bounds interpretation of the failed language-model task while preserving the supervision and one-rig limitations.")
    supplement=[r'\section{Same-feature supervised bearing benchmark}',r'\label{sec:bearing_baseline_supp}',r'This appendix documents the result-informed benchmark performed after the frozen bearing language-model outcomes. It adds 108 development recordings from the same six background specimens, no new test specimens and no new language-model requests. It does not retrospectively preregister the earlier study.',METHODS,primary_table.replace('tab:bearing_baseline','tab:supp_bearing_baseline')]
    configs=[]
    for x in selected:
        label='Primary' if x['task']=='primary' else esc(x['held_out_regime'])
        params=x['params']
        description=(f"$C={params['C']}$, $\\gamma={params['gamma']}$" if x['model']=='SVM' else f"features={params['max_features']}, leaf={params['min_samples_leaf']}")
        configs.append([label,x['model'],description,pct(x['cv_mean_accuracy']),str(x['train_records'])])
    supplement += [table('Development-only parameter selection. Each mean uses the same eight overlapping specimen-disjoint splits within that task. Validation outcomes guide parameters but never change the frozen grid or test set.','tab:bearing_configs','lllrr',r'Task / omitted regime & Model & Selected configuration & CV (\%) & Train $n$',configs),r'\FloatBarrier']
    regimes=[]
    for g in p['regimes']:
        for model in ['SVM','RF']:
            a=next(x for x in b['primary/'+model]['by_regime'] if x['regime']==g);z=b['loro_'+g+'/'+model]
            regimes.append([esc(g),model,f"{a['correct']}/9",f"{z['correct']}/9",f"{z['macro_f1']:.3f}"])
    supplement += [table('Per-regime test results on the same nine unseen specimens. Each primary model is fitted once; each omitted-regime model is separately retuned and fitted on the other three regimes.','tab:bearing_regimes','llrrr','Test regime & Model & Primary correct & Held-out correct & Held-out F1',regimes)]
    bearings=[]
    for spec in b['primary/SVM']['by_bearing']:
        name=spec['bearing']; row=[name]
        for task in ['primary','loro_combined']:
            for model in ['SVM','RF']:
                v=next(x for x in b[task+'/'+model]['by_bearing'] if x['bearing']==name);row.append(f"{v['correct']}/4")
        bearings.append(row)
    supplement += [table('Complete test-specimen outcomes, four records per bearing. The combined held-out-regime column uses one prediction from each of four task-specific models for each specimen.','tab:bearing_specimens','lrrrr','Bearing & Primary SVM & Primary RF & Held-out SVM & Held-out RF',bearings)]
    confusion=[]
    for task,label in [('primary','Primary'),('loro_combined','Held-out')]:
        for model in ['SVM','RF']:
            cm=b[task+'/'+model]['confusion_matrix_truth_rows']
            for i,cl in enumerate(['Healthy','Outer','Inner']):
                confusion.append([label,model,cl,*cm[i],pct(cm[i][i]/sum(cm[i]))])
    supplement += [table('Confusion matrices with true class in rows, plus class recall. Every model/setting has 12 test records per class.','tab:bearing_confusion','lllrrrr',r'Task & Model & Truth & Pred. H & Pred. O & Pred. I & Recall (\%)',confusion),r'\FloatBarrier']
    llm_rows=[]
    for call,m in sorted(lm.items()):
        llm_rows.append([m['round'],m['condition'],m['repeat'],f"{m['correct']}/36",pct(m['accuracy']),f"{m['macro_f1']:.3f}"])
    supplement += [table('All frozen bearing language-model calls on the unchanged 36 focal records. Repeats are repeated predictions of the same specimens, not independent test samples.','tab:bearing_llm_calls','llrrrr',r'Round & Condition & Repeat & Correct & Accuracy (\%) & Macro F1',llm_rows)]
    run=read(BASE/'fit/run.json')
    supplement.append(f"Computation uses scikit-learn {run['environment']['packages']['scikit-learn']}. The fixed protocol executes 400 development-fold fits and ten final fits. Candidate/fold predictions, all parameter means, selected configurations, source hashes and client timings are preserved. Training and prediction timings are local descriptive measurements, not a hardware-matched cost comparison with remote service calls. No record-level significance tests, label-permutation searches or feature-ranking claims are added.")
    audit=read(ROOT/'data/bearing_baseline_v1/audit.json')
    assert audit['original48_recomputed_all28_strings_equal'] and audit['development_six_decimal_changed_records']==0
    supplement.append(r'A preprocessing assertion exposed one floating-point Nyquist boundary in a newly selected recording: its terminal rFFT frequency was represented as 32000.000000000004 Hz and the legacy inclusive upper bound omitted a relative power of approximately $7.95\times10^{-9}$. A separately frozen extraction amendment includes all mathematically valid Nyquist bins. It changes one unrounded final-band value and no six-decimal development feature; all 48 original records were recomputed with all 28 strings unchanged. The failed extraction, fixed member list and both code versions remain archived. No recording was replaced and no test outcome guided this correction.')
    supplement.append(discussion)
    supplement.append(r'\FloatBarrier')
    return {'BASELINE_METHODS':METHODS,'BASELINE_RESULTS':'\n\n'.join(result),'BASELINE_ABSTRACT':abstract,'BASELINE_DISCUSSION':discussion,'BASELINE_CONCLUSION':conclusion},'\n\n'.join(supplement)
