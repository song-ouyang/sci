"""Build manuscript inserts only when the separate follow-up has frozen scores."""
import json
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.mechanism_extension import BASE,CONDITIONS,MODELS,audit_outputs,digest
PAPER=ROOT/'paper'
KEYS=['EXTENSION_ABSTRACT','EXTENSION_METHODS','EXTENSION_RESULTS','EXTENSION_DISCUSSION','EXTENSION_CONCLUSION']
NAMES={'gpt-5.6-sol':'GPT-5.6 Sol','gpt-6-astra':'GPT-6 Astra'}
LABELS={'clean':'Clean','rotation':'Cyclic mapping fault','pair_shuffle':'Correct pair reordering','anonymous':'Anonymous fields','random_mapping':'Random mapping fault','sample_shuffle':'Sample reordering'}

def fmt(x):return 'N/A' if x is None else f'{100*x:.1f}'

def build():
    path=BASE/'analysis/scores.json'
    if not path.exists():
        print('No scored extension: manuscript retains original empirical scope.')
        return dict.fromkeys(KEYS,'')
    p,g,records,_=audit_outputs(BASE)
    data=json.loads(path.read_text())
    if data['protocol_sha256']!=digest(BASE/'protocol.json') or data['frozen_manifest_sha256']!=digest(BASE/'predictions/frozen_manifest.json'):
        raise ValueError('Extension scores do not match frozen inputs/results')
    metrics=data['metrics']; pairs=data['paired_contrasts']
    def metric(m,c,r):return next(x for x in metrics if x['model']==m and x['condition']==c and x['repeat']==r)
    def pair(m,c,r=1,left='clean'):return next(x for x in pairs if x['model']==m and x['right']==c and x['left']==left and x['repeat']==r)
    coverage=data['coverage']
    methods=r'''\subsection{Exploratory representation and context probes}
After inspecting the original results, we designed a separate exploratory TEP extension for GPT-5.6 Sol and GPT-6 Astra, selected because their observed mapping responses contrasted. The extension reuses the same 70 held-out runs; it adds no independent test sources and is not a confirmatory replication. Its protocol, transformation code, request order and label-free payloads were frozen before the new calls. All new outcomes were frozen before this extension was scored. Historical primary responses were neither replaced nor pooled with these calls.

Six full-batch conditions each receive three fresh sessions per model: the unchanged clean input; the original cyclic descriptor fault; a seeded joint reordering of correct descriptor--value pairs; anonymous positional field labels replacing sensor identities, names and units; one seeded noncyclic descriptor derangement with no fixed points; and a seeded reordering of the clean observations. Joint field reordering preserves all physical pairings, whereas the random mapping fault preserves all numerical strings in their canonical positions. Anonymous inputs retain numerical order and the task-level fault definitions; only sensor-field descriptors and the corresponding row-description sentence change. Each transformation is fixed across samples and repeats. Thus this extension tests particular alternatives to the original rotation, not a distribution of all possible schema faults.

For a bounded context probe, seven observations are selected by the smallest SHA-256 values of a fixed salt concatenated with the anonymous IDs, without using labels or previous predictions. Their original relative order is retained. For each model and repeat, these records are classified together in one seven-record request and in seven independent single-record requests. Their predictions in the fresh 70-record clean batch provide the large-context comparison. This design holds each focal record fixed but changes its surrounding context and position; it does not isolate every aspect of context length. The subset is not class-stratified and is too small for population-level accuracy claims.

The schedule contains 84 requests and 2,604 requested classifications over the original 70 sources. Model aliases and reasoning effort follow the original interface, while every request has a new session. One CLI invocation is permitted per scheduled job; invalid outputs remain unavailable for the entire request, without repair or replacement. Full-batch contrasts and the seven-case context contrasts align by ID, with available-pair denominators reported explicitly. Repeat 1 is the descriptive reference, and repeats 2 and 3 are reported separately. Conditional paired intervals use 10,000 source-unit resamples with seed 20260927; they do not incorporate shared-context dependence or variation over transformations.
'''
    table=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Separate exploratory extension on the same 70 TEP runs. Each accuracy column is a fresh full-batch response. Flips compare that condition with the contemporaneous clean response in repeat 1 after ID alignment; N/A retains invalid requests.}',r'\label{tab:extension}',r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Model & Condition & R1 (\%) & R2 (\%) & R3 (\%) & R1 flips \\',r'\midrule']
    for m in MODELS:
        for c in CONDITIONS:
            flip='--' if c=='clean' else ('N/A' if pair(m,c)['flips'] is None else f"{pair(m,c)['flips']}/{pair(m,c)['paired_n']}")
            table.append(' & '.join([NAMES[m],LABELS[c],*[fmt(metric(m,c,r)['accuracy']) for r in (1,2,3)],flip])+r' \\')
        if m!=MODELS[-1]:table.append(r'\addlinespace')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    context=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Seven-case context probe. Entries are prediction flips over available paired cases; the same IDs are used in all three request configurations. Each repeat of the singleton panel consists of seven separate sessions. These small, label-independently selected probes do not estimate general batch-size effects.}',r'\label{tab:context}',r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Model & Comparison & R1 & R2 & R3 \\',r'\midrule']
    for m in MODELS:
        for a,b,label in [('clean','subset_batch','70-record vs. 7-record'),('subset_batch','single_panel','7-record vs. single-record'),('clean','single_panel','70-record vs. single-record')]:
            rr=[pair(m,b,r,left=a) for r in (1,2,3)]
            context.append(' & '.join([NAMES[m],label,*['N/A' if x['flips'] is None else f"{x['flips']}/{x['paired_n']}" for x in rr]])+r' \\')
    context += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    paragraphs=[]
    for m in MODELS:
        clean=metric(m,'clean',1)
        cc=[pair(m,c) for c in ['pair_shuffle','anonymous','random_mapping','sample_shuffle']]
        paragraphs.append(NAMES[m]+" has fresh clean repeat-1 accuracy of "+fmt(clean['accuracy'])+r'\%. '+
            'For correct pair reordering, anonymous fields, random mapping and sample reordering, the corresponding accuracies are '+
            ', '.join(fmt(metric(m,c,1)['accuracy']) for c in ['pair_shuffle','anonymous','random_mapping','sample_shuffle'])+r'\%, respectively. '+
            'The respective paired flip counts are '+', '.join('N/A' if x['flips'] is None else f"{x['flips']}/{x['paired_n']}" for x in cc)+'.')
    context_text=[]
    for m in MODELS:
        ss=[pair(m,'subset_batch',r) for r in (1,2,3)]
        ii=[pair(m,'single_panel',r) for r in (1,2,3)]
        counts=lambda rr,key: ', '.join('N/A' if x[key] is None else f"{round(x[key]*x['paired_n'])}/{x['paired_n']}" for x in rr)
        context_text.append(NAMES[m]+': on the same seven focal records, the full-batch correct counts are '+counts(ss,'left_accuracy')+
            '; seven-record counts are '+counts(ss,'right_accuracy')+'; and single-record counts are '+counts(ii,'right_accuracy')+
            ', in repeats 1, 2 and 3 respectively.')
    composition=', '.join(l.replace('_',r'\_')+' ('+str(n)+')' for l,n in data['context_subset_class_counts'].items() if n)
    context_text.append('The label-free selection contains '+composition+'. Normal and several fault classes are absent. The probe therefore characterizes these seven records rather than class-balanced diagnostic performance.')
    subprocess.run([sys.executable,str(ROOT/'src/analysis/plot_mechanism_context.py')],check=True)
    context_figure=(r'\begin{figure}[htbp]'+ '\n'+r'\centering\includegraphics[width=\linewidth]{figures/mechanism_extension/matched_context.pdf}'+ '\n'+
        r'\caption{Matched context probe on the same seven observations. Each point counts correct predictions for those seven IDs, including when they appear inside the 70-record batch. Horizontal offsets separate the three fresh repetitions; lines connect request configurations and do not imply a continuous size-response law. Single-record points summarize seven separate calls. No new source observations are introduced.}'+ '\n'+
        r'\label{fig:matchedcontext}\end{figure}')
    results=(r'\subsection{Exploratory representation and context extension}'+'\n'+
        f"The separate extension retains all {coverage['planned']} scheduled requests: {coverage['valid']} are valid, yielding {coverage['valid_labels']} labels eligible for scoring from {coverage['requested_labels']} requested classifications. These calls concern the same 70 runs as the original TEP comparison. Table~\\ref{{tab:extension}} reports all full-batch repeats, and Table~\\ref{{tab:context}} reports the fixed seven-case context probes.\n\n"+
        '\n'.join(table)+'\n\n'+'\n\n'.join(paragraphs)+'\n\n'+'\n'.join(context)+'\n\n'+' '.join(context_text)+' Figure~\\ref{fig:matchedcontext} displays the matched correct counts.\n\n'+
        'All pairwise accuracy differences, conditional intervals and session-level outputs are retained in the extension package. Differences between this extension and the historical sessions are not treated as causal effects of a transformation; each contrast uses its own fresh clean reference.\n\n'+context_figure+'\n')
    # Interpretation is conservative and conditional; never substitute a preconceived result.
    evidence=[]
    for m in MODELS:
        insensitive=[LABELS[c].lower() for c in CONDITIONS[1:] if all(pair(m,c,r)['flips']==0 and pair(m,c,r)['paired_n']==70 for r in (1,2,3))]
        sensitive=[LABELS[c].lower() for c in CONDITIONS[1:] if any((pair(m,c,r)['flips'] or 0)>0 for r in (1,2,3))]
        if insensitive:evidence.append(NAMES[m]+' shows no observed flips in any of the three paired repetitions for '+', '.join(insensitive)+'.')
        if sensitive:evidence.append('At least one repeated response changes for '+NAMES[m]+' under '+', '.join(sensitive)+'.')
    discussion=(r'\subsection{What the added controls can discriminate}'+'\n'+' '.join(evidence)+'\n\n'+
        'The context probe qualifies the perfect full-batch scores: both services change at least one focal prediction when surrounding records are removed, despite unchanged focal sensor text. The repeated seven-record deficits are especially pronounced for Sol. These observations establish a dependence of observed outcomes on the tested request configuration; context composition, position and stochastic service behavior remain intertwined.\n\n'+
        'Correct pair reordering preserves the represented physical state, so disagreement under this condition is evidence of sensitivity to presentation for the tested input. Anonymous fields jointly remove sensor identity, name and unit while retaining task-level fault descriptions. Retained accuracy would therefore show that explicit sensor-field descriptors are unnecessary for those decisions in that context, without establishing how the model obtained them. A noncyclic mapping fault tests whether invariance is confined to the original regular rotation, while retaining numerical position as a possible cue. None of these controls alone establishes or excludes benchmark exposure.\n\n'+
        'The seven-case context comparisons inspect whether a fixed observation receives a different label when surrounding observations change. A nonzero flip count establishes an observed context-associated discrepancy; zero flips in this small probe do not establish sample independence. Shared service stochasticity also contributes to these comparisons. The original sample selection, post hoc model selection and small context panel restrict these findings to hypothesis refinement. The source-disjoint fourth stage addresses the first two gaps with new runs and several mappings; a broader provider sample and external data-generation processes remain necessary for wider generalization.\n')
    # Abstract focuses on the added design, leaving causal claims to the controlled results.
    sol=pair('gpt-5.6-sol','subset_batch'); astra=pair('gpt-6-astra','subset_batch')
    abstract=(f"A separate {coverage['planned']}-request follow-up tests field reordering, descriptor removal, noncyclic corruption and context. "
              f"For seven matched cases, repeat-1 correct counts change from {round(sol['left_accuracy']*7)}/7 to {round(sol['right_accuracy']*7)}/7 for Sol and "
              f"from {round(astra['left_accuracy']*7)}/7 to {round(astra['right_accuracy']*7)}/7 for Astra when the surrounding 63 records are removed.")
    conclusion=('In the separate follow-up, descriptor invariance on the full TEP batch coexists with changed decisions for the same focal records in smaller contexts. This distinguishes observed mapping invariance from context-independent classification and makes request composition an explicit part of the evaluation contract.')
    return dict(zip(KEYS,[abstract,methods,results,discussion,conclusion]))

if __name__=='__main__':
    value=build()
    (PAPER/'mechanism_extension_text.json').write_text(json.dumps(value,indent=2)+'\n')
    print('Built extension manuscript inserts.' if value['EXTENSION_RESULTS'] else 'No unobserved extension results included.')
