"""Render the fourth stage exclusively from frozen, audited responses."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.independent_validation import audit_outputs,BASE,digest
from src.analysis.render_revision_v3_paper import latex_table,percent,interval,span,NAMES

METHODS=r'''\subsection{Source-disjoint follow-up and crossed controls}
After inspecting the first three stages, we froze an additional protocol before its inference. GPT-5.6 Sol and GPT-6 Astra were selected because their earlier responses differed; this selection does not support a population-wide service ranking. The fixed seed is 20260928. Fourteen conditions, two services and three fresh sessions per condition give 84 requests and 4,854 requested labels. The schedule interleaves conditions, including contemporaneous clean controls, with four concurrent requests overall. Each job permits one invocation, no semantic retry or manual repair, and a 900-second timeout. The tool-disabled CLI, medium reasoning setting and whole-batch validation are retained. Every terminal response is frozen before scoring; repeat 1 is primary.

The TEP panel contains 70 new main runs, ten per class, plus two disjoint 21-run background panels A and B, each with three runs per class. Globally, run IDs are excluded across fault scenarios if used in the original training/test split or the historical 50-, 100- or 200-per-class evenly spaced selections. This conservative exclusion removes 365 of the 500 available IDs; 112 of the remaining 135 are selected without inspecting predictions. The preparation audit additionally matches the 1,400 legacy derived rows to raw source coordinates and numerical values. Thus, the main and background panels neither overlap each other nor reuse those prior sources.

The main panel retains one late snapshot per run at a seeded sample index from 200--500. A stratified subset of 21 focal observations covers all seven classes with three observations each. Five context conditions present these identical focal observations alone, before A, after A, before B, or after B. The A/B comparison holds focal position and total size 42 fixed; the front/back comparison holds background identity fixed. The focal-only panel has size 21. Its comparison with size 42 or with the focal subset extracted from the 70-case clean response changes composition as well as size, and cannot isolate a pure size effect. No class quotas or reference labels are supplied to the inference service.

Seven mapping conditions use the identical 70 late observations: clean descriptors, the original one-position rotation, three fixed random derangements of all 52 descriptors, a local swap of reactor-level and separator-level descriptors (both percentages), and a seeded reordering that preserves each correct descriptor--value pair. The five incorrect mappings retain numerical strings and their positions. Correct-pair reordering changes their positions but preserves physical correspondence. The fixed permutations and complete inputs are included in the protocol artifacts; each is reported separately.

Two additional conditions challenge the original task setting. Earlier TEP observations use sample 40 from the same 70 runs, approximately 60 minutes after fault injection, paired with the late snapshots by run. These add observations, not sources, and do not measure early-alarm delay. The FD002 panel selects 60 previously unused test engines, 20 per RUL category, from the official six-operating-condition, one-fault-mode subset. Six of its 259 engines have fewer than 30 available cycles and are excluded; the eligible category counts are 123 Healthy, 69 Degrading and 61 Critical. The final 30 available cycles use the original FD001-selected channels, rounding and 60-feature summary without target-domain adaptation. Operating-setting columns remain excluded as in FD001, so regime variation may affect these summaries. RF and SVM are fit only on the original training features and evaluated once on new late TEP, earlier TEP and FD002 observations. This is a fixed-representation transfer stress test, not a tuned FD002 benchmark.

The stage adds 172 source units and 242 observations, including the 70 same-run earlier snapshots. Its precision rationale is fixed before inference: independent-binomial worst-case 95\% half-widths are approximately 11.4 pp for 70 cases, 12.3 pp for 60 and 20 pp for 21 focal cases. These are feasibility benchmarks, potentially optimistic under grouped generation, not a significance-based power calculation. All repeated outcomes, paired flips and conditional paired bootstrap intervals (10,000 source-pair resamples, seed 20260928) are retained; no permutation or comparison is selected by its score.'''

CONDITIONS={'clean':'Clean','rotation':'Rotation','random_1':'Random 1','random_2':'Random 2','random_3':'Random 3','same_unit_swap':'Same-unit swap','pair_shuffle':'Correct-pair reorder','early':'TEP earlier','fd002':'FD002','focal_only':'Focal only (21)','a_front':'Focal before A (42)','a_back':'Focal after A (42)','b_front':'Focal before B (42)','b_back':'Focal after B (42)'}

def main():
    p,g,verified,_=audit_outputs(BASE)
    s=json.loads((BASE/'analysis/scores.json').read_text())
    if s['protocol_sha256']!=digest(BASE/'protocol.json') or s['frozen_manifest_sha256']!=digest(BASE/'predictions/frozen_manifest.json'):
        raise ValueError('Stale independent-source analysis')
    if digest(BASE/'data_audit.json')!=p['data_audit_sha256']:
        raise ValueError('Source audit changed')
    models=p['models'];cv=s['coverage']
    def get(m,c,r=1,source='metrics'):
        return next(x for x in s[source] if x['model']==m and x['condition']==c and x.get('repeat',r)==r)
    def pair(m,c,r=1):return next(x for x in s['paired'] if x['model']==m and x['repeat']==r and x['left']=='clean' and x['right']==c)
    text={'INDEPENDENT_METHODS':METHODS}
    text['STAGE_TABLE']=latex_table('Evidence stages. Valid labels count repeated classifications, not independent assets. The first three stages reuse 130 sources; the last adds 112 TEP runs and 60 FD002 engines. Both historical and tool-enabled conversational Claude records are excluded.','tab:stages','lrrr',['Stage','Requests','Valid','Labels valid/planned'],[
        ['Original paired study','27','26','1,740/1,800'],['Representation/context follow-up','84','84','2,604/2,604'],['Additional-provider API study','18','15','990/1,200'],['Source-disjoint follow-up',str(cv['planned']),str(cv['valid']),f"{cv['valid_labels']:,}/{cv['requested_labels']:,}"],['Total (302 unique sources)','213',str(125+cv['valid']),f"{5334+cv['valid_labels']:,}/10,458"]])
    rows=[]
    for m in models:
        for c in list(CONDITIONS)[:7]:
            rr=[get(m,c,r) for r in [1,2,3]];q=pair(m,c) if c!='clean' else None
            rows.append([NAMES[m].replace('GPT-',''),CONDITIONS[c],*[percent(x['accuracy']) for x in rr],percent(rr[0]['macro_f1']),f"{q['flips']}/{q['paired_n']}" if q and q['flips'] is not None else ('--' if c=='clean' else 'N/A')])
    results=r'\subsection{Source-disjoint follow-up: mapping, context and transfer}'+'\n'
    results+=f"The fourth stage yields {cv['valid']}/{cv['planned']} valid responses and {cv['valid_labels']:,}/{cv['requested_labels']:,} eligible labels. All scheduled outcomes are retained. "
    results+=r'Tables~\ref{tab:ind_mapping}--\ref{tab:ind_transfer} show the planned conditions; Figure~\ref{fig:independent} summarizes their repeated outcomes. All comparisons below use the new source panel, separately from the preceding discovery stages.'+'\n\n'
    results+=latex_table('New TEP main panel (70 runs). Accuracy for each scheduled repeat and primary macro F1 are percentages. Primary flips compare the condition with its contemporaneous clean response. Correct-pair reordering is a physically matched control, not an incorrect mapping.','tab:ind_mapping','llrrrrr',['Service','Condition','R1','R2','R3','R1 F1','R1 flips'],rows)
    for m in models:
        q=pair(m,'rotation');random=[get(m,c)['accuracy'] for c in ['random_1','random_2','random_3']];vals=[x for x in random if x is not None]
        results+=f"For {NAMES[m]}, the primary clean-to-rotation accuracy difference is {percent(q['difference'])} pp (conditional 95\\% interval {interval(q['conditional_ci'])}); "
        results+=f"primary accuracies across the three random mappings span {span(vals)}\\%. " if vals else 'all primary random-mapping responses are unavailable. '
    results+='These finite realizations do not estimate a distribution over all mapping errors.\n\n'
    rows=[]
    for m in models:
        for c in ['clean','focal_only','a_front','a_back','b_front','b_back']:
            rr=[get(m,c,r,'focal') for r in [1,2,3]]
            rows.append([NAMES[m].replace('GPT-',''), 'Within clean (70)' if c=='clean' else CONDITIONS[c],*[f"{x['correct']}/21" if x['correct'] is not None else 'N/A' for x in rr]])
    results+=latex_table('The same 21 new focal observations in six contexts, covering all seven classes. Cells give correct focal labels, not whole-request accuracy. Parentheses denote total request size; the additional cases in A/B are not included in the numerator.','tab:ind_context','llrrr',['Service','Focal context','R1','R2','R3'],rows)
    results+='Across the six prespecified comparisons of each type (two per repeat), '
    for m in models:
        parts=[]
        for kind in ['background','position']:
            ps=[x for x in s['paired'] if x['model']==m and x['kind']==kind and x['flips'] is not None]
            if ps:parts.append(f"{min(x['flips'] for x in ps)}--{max(x['flips'] for x in ps)} out of 21 for {kind}")
        results+=NAMES[m]+' shows '+' and '.join(parts)+' label flips. '
    results+='These comparisons vary one context dimension at a time, while repeated calls still include service variability. They do not identify the model\'s internal mechanism.\n\n'
    rows=[]
    for c in ['clean','early','fd002']:
        for m in models+['RF','SVM']:
            r=get(m,c) if m in models else get(m,c,source='baselines')
            vals='/'.join(percent(get(m,c,i)['accuracy']) for i in [1,2,3]) if m in models else percent(r['accuracy'])
            rows.append(['TEP late' if c=='clean' else CONDITIONS[c],NAMES[m].replace('GPT-',''),vals,percent(r['macro_f1']),interval(r['accuracy_wilson95'])])
    results+=latex_table('New-source classification and fixed-representation transfer. LLM accuracy lists R1/R2/R3; RF/SVM are each fit once on original training data. F1 and Wilson intervals use R1 for LLMs. All values are percentages; intervals are conditional independent-unit approximations. Earlier TEP shares the 70 late-stage runs; FD002 has 60 different engines.','tab:ind_transfer','llrrr',['Panel','Method','Acc. (R1/R2/R3)','R1 F1','R1 95\\% interval'],rows)
    results+=r'No FD002 observation is used to tune channels, preprocessing or classifier hyperparameters. The TEP phase contrast pairs earlier and later snapshots by the same run; their classifications are not pooled as 140 independent sources.'+'\n\n'
    sol_early=get('gpt-5.6-sol','early');astra_early=get('gpt-6-astra','early')
    results+=f"Primary earlier-TEP accuracies are {percent(sol_early['accuracy'])}\\% for Sol and {percent(astra_early['accuracy'])}\\% for Astra, compared with {percent(get('RF','early',source='baselines')['accuracy'])}\\% for RF and {percent(get('SVM','early',source='baselines')['accuracy'])}\\% for SVM. These baselines are trained on the original late-snapshot representation; this difference is conditional on that training-to-test phase shift, not evidence against earlier-phase-trained classifiers. On FD002, the primary service accuracies are {percent(get('gpt-5.6-sol','fd002')['accuracy'])}\\% and {percent(get('gpt-6-astra','fd002')['accuracy'])}\\%, while both unadapted baselines give {percent(get('RF','fd002',source='baselines')['accuracy'])}\\%. No optimized regime-aware reference is evaluated.\n\n"
    figure_dir=ROOT/'paper/figures/independent_validation';figure_dir.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7,3.6),constrained_layout=True)
    series=[]
    for ax,conditions,source,title in [(axes[0],list(CONDITIONS)[:7],'metrics','(a) New TEP runs: mapping controls'),(axes[1],['clean','focal_only','a_front','a_back','b_front','b_back'],'focal','(b) Identical 21 focal observations')]:
        for mi,m in enumerate(models):
            for repeat,marker in [(1,'o'),(2,'s'),(3,'^')]:
                vals=[get(m,c,repeat,source)['accuracy'] for c in conditions]
                xx=np.arange(len(conditions))+(mi-.5)*.22+(repeat-2)*.045
                ax.scatter(xx,[100*v if v is not None else np.nan for v in vals],color=['#176586','#af511e'][mi],marker=marker,s=27,label=NAMES[m]+f' R{repeat}')
                series += [dict(panel=source,model=m,condition=c,repeat=repeat,accuracy=v) for c,v in zip(conditions,vals)]
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(([CONDITIONS[c].replace('Correct-pair reorder','Pair reorder') for c in conditions] if source=='metrics' else ['Within 70','Focal only','Before A','After A','Before B','After B']),rotation=43,ha='right')
        ax.set_ylim(-3,105);ax.set_ylabel('Accuracy (%)');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();order=[0,3,1,4,2,5];fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='outside lower center',ncol=3,fontsize=8)
    for suffix in ['pdf','png']:fig.savefig(figure_dir/f'independent_controls.{suffix}',dpi=200)
    plt.close(fig)
    with (BASE/'analysis/figure_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(series[0]));w.writeheader();w.writerows(series)
    results+=r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/independent_validation/independent_controls.pdf}
\caption{All three scheduled repeats of the new-source mapping and context panels. Color identifies service; circles, squares and triangles identify repeats 1, 2 and 3. Panel (a) uses 70 new TEP runs per condition. Panel (b) scores only the same 21 focal cases in each request. These repeated points describe observed calls, not confidence intervals or added independent sources.}
\label{fig:independent}
\end{figure}'''
    text['INDEPENDENT_RESULTS']=results
    text['INDEPENDENT_ABSTRACT']=(f"A further {cv['planned']}-request follow-up adds 172 disjoint sources, multiple mappings, crossed context controls and FD002 transfer. Across all stages, {125+cv['valid']}/213 responses are valid on 302 source units. Primary FD002 accuracy is {percent(get('gpt-5.6-sol','fd002')['accuracy'])}\\% for Sol and {percent(get('gpt-6-astra','fd002')['accuracy'])}\\% for Astra.")
    text['INDEPENDENT_DISCUSSION']=r'''\subsection{What the source-disjoint follow-up adds}
The additional stage addresses two distinct gaps: replicating the interface tests on previously unused simulator runs, and challenging the fixed FD001 representation on a six-operating-condition subset. The 21-case focal panel also includes every selected TEP class, unlike the original seven-case probe. Its crossed A/B and front/back comparisons narrow the context interpretation by holding request size fixed. Nevertheless, each observed contrast contains variation from separate service calls. Three repeats are insufficient to estimate stable average causal effects over prompts, providers or times.

The new focal panel also qualifies the earlier seven-case result: Sol classifies all 21 focal-only cases correctly in the first two repeats, whereas its primary focal score falls to 9/21 when background B precedes them. Thus, fewer accompanying cases do not consistently imply worse classification. Astra retains perfect new full-batch mapping scores but makes one error in each focal-only repeat. These are configuration-specific patterns rather than a universal batch-size law.

Mapping realizations are operationally different interventions. A random reassignment can be less regular than the cyclic rotation; a same-unit local swap changes only two identities; correct-pair reordering preserves physical correspondence. Their separate outcomes qualify any general claim about ``sensor-name robustness.'' The phase and FD002 contrasts likewise characterize the specified snapshot representation rather than early warning, diagnostic reasoning or optimized transfer learning. Any discrepancy from the original results is retained as evidence of scope dependence, not repaired by replacing observations or choosing a later repeat.'''
    text['INDEPENDENT_CONCLUSION']='The source-disjoint stage extends the evidence to 302 total source units with multiple mappings and crossed focal contexts, while separating within-simulator replication from fixed-representation FD002 transfer.'
    (ROOT/'paper/independent_validation_text.json').write_text(json.dumps(text,indent=2)+'\n')
    print('Rendered source-disjoint follow-up from audited frozen responses')

if __name__=='__main__':main()
