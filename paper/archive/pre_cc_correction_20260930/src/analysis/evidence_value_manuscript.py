"""Render the new descriptive audits without changing frozen observations."""
from pathlib import Path
import csv, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[2]
VALUE=ROOT/'results/decision_value_audit_2026-09-29'
EVIDENCE=ROOT/'results/evidence_boundary_audit_2026-09-30.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def esc(x):return str(x).replace('_',r'\_')
def table(caption,label,columns,header,rows):
    return '\n'.join([r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{@{}'+columns+r'@{}}\toprule',header+r' \\\midrule',*[ ' & '.join(map(str,row))+r' \\' for row in rows],r'\bottomrule\end{tabular}\end{table}'])

def make_evidence_value_fragments():
    e=read(EVIDENCE);v=read(VALUE/'audit.json')
    assert e['analysis_code_sha256']==sha(ROOT/'src/analysis/audit_evidence_boundaries.py')
    assert v['script_sha256']==sha(ROOT/'src/analysis/audit_decision_value.py')
    for rel,h in e['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    for study in v['studies'].values():
        for rel,h in study['source_hashes'].items():assert sha(ROOT/rel)==h,rel
    for name,h in v['output_sha256'].items():assert sha(VALUE/name)==h,name
    assert all(v['checks'].values()) and v['new_model_calls']==e['new_model_calls']==0
    assert e['context_blocked']['cross_position_flip_range']==[0,6]
    assert e['context_blocked']['within_condition_flip_range']==[0,8]
    classes=e['tep']['sol_rotation']['class_rows']
    assert sum(r['stable_different'] for r in classes)==20
    assert sum(r['denominator'] for r in classes)==70
    transitions={(x['clean_label'],x['rotation_label']):x['count'] for r in classes for x in r['stable_transitions']}
    assert transitions=={('Normal','Fault_1'):4,('Fault_1','Fault_7'):10,('Fault_7','Fault_1'):5,('Fault_13','Fault_1'):1}
    mrows=[]
    for r in classes:
        destination=', '.join(esc(x['rotation_label'])+' ('+str(x['count'])+')' for x in r['stable_transitions']) or '--'
        mrows.append([esc(r['true_label']),r['denominator'],r['stable_different'],destination,r['stable_same'],r['any_side_unstable']])
    mappingtable=table('Post hoc localization of the specified Sol cyclic-mapping contrast. D: each condition has one internally constant label across its three calls, and the two labels differ. S: all six labels agree. U: at least one condition varies. Every D case is correct in all clean calls and wrong in all rotated calls. Zero D does not imply robustness; U cases remain visible.','tab:mapping_localization','lrrlrr',r'True class & $n$ & D & Rotated label (D count) & S & U',mrows)
    mapping=r'''A result-informed class-transition audit localizes the 20 repeat-stable discrepancies (Table~\ref{tab:mapping_localization}). Ten Fault\_1 observations change consistently to Fault\_7, five Fault\_7 observations to Fault\_1, four Normal observations to Fault\_1 and one Fault\_13 observation to Fault\_1. Thus 15 of the 20 cases involve the reciprocal Fault\_1/Fault\_7 confusion. Each is correct in all three clean calls and wrong in all three specified rotation calls. The other 50 cases include ten stable-same and 40 variable cases, rather than 50 demonstrated invariances. This concentration describes one specified corruption and service schedule; it does not identify a physical mechanism or generalize to arbitrary sensor mappings.'''+ '\n\n'+mappingtable
    methods=r'''\subsection{Result-informed evidence and decision-loss audits}
The following analyses were specified after the aggregate outcomes were known. A separate implementation verifies frozen response hashes, reconciles raw responses with parsed labels and reconstructs the class-level repeat categories. It preserves all seven TEP classes, including unstable cases. These checks address numerical traceability, not human scientific review or new external validation.

For each fixed retain-or-reject policy $p$, let $E_p$ be accepted errors, $R_p$ all rejections and $N$ planned focal decision opportunities. We assign accepted errors loss 1, accepted correct labels loss 0 and each rejected case an assumed total expected loss $c$. This total may include referral work, delay and residual downstream errors; rejection is not assumed to repair the diagnosis. With total request-related loss $K_p$ in the same units,
\begin{equation}
L_p(c,K_p)=(E_p+cR_p+K_p)/N.
\label{eq:decision_loss}
\end{equation}
We display $0\leq c\leq1$ and $K_p=0$ as an explicit sensitivity scenario, not measured costs, fitted thresholds or a new decision rule. Actual referral loss can exceed this range or depend on class. Same/View use the same request count, although token use and duration differ. Accept-all uses F1 alone; reject-all needs no diagnosis calls. Omitting request cost is therefore optimistic when comparing a check with those reference actions. Round-specific and source-deletion summaries assess dependence on the observed sample composition. Deletion is not refitting or new held-out validation. Appendix H details an additional conditional bearing-reweighting sensitivity analysis; no new population confidence interval is claimed.
'''
    h=v['studies']['hydraulic']['totals'];b=v['studies']['paderborn']['totals']
    assert [(h[x]['accepted_error'],h[x]['rejected']) for x in ['Same','View']]==[(4,17),(5,10)]
    assert [(b[x]['accepted_error'],b[x]['rejected']) for x in ['Same','View']]==[(58,6),(21,42)]
    assert h['Accept_all_F1']['accepted_error']==10 and b['Accept_all_F1']['accepted_error']==61
    rr={(x['dataset'],x['subset']):x for x in v['round_loss_contrasts']}
    assert [rr[('hydraulic','round_1')][k] for k in ['delta_correct','delta_errors','delta_rejections']]==[0,0,0]
    assert [rr[('paderborn','round_1')][k] for k in ['delta_correct','delta_errors','delta_rejections']]==[-1,-20,21]
    assert [rr[('paderborn','round_2')][k] for k in ['delta_correct','delta_errors','delta_rejections']]==[2,-17,15]
    deletion=[x for x in v['leave_one_group_out'] if x['dataset']=='paderborn' and x['excluded_dimension']=='source_unit']
    ki=next(x for x in deletion if x['excluded_group']=='KI04')
    assert ki['c_exact']=='35/37' and len(deletion)==9
    figpath=ROOT/'paper/figures/decision_loss_2026-09-30.pdf'
    shutil.copyfile(VALUE/'decision_loss.pdf',figpath)
    results=r'''\subsection{Decision-loss boundaries and their sample sensitivity}
At $K_p=0$, the hydraulic Same/View losses are $(4+17c)/54$ and $(5+10c)/54$. Hence View is lower only for $c>1/7$: six additional retained correct diagnoses accompany one additional released error (Figure~\ref{fig:decision_loss}). Relative to accepting all F1 labels, the respective empirical break-even rejection losses are $6/17$ and $1/2$. The entire net Same-to-View difference comes from round 2; round 1 has equal aggregate error, correct-acceptance and rejection counts. This does not imply identical per-case decisions or a repeated improvement across both rounds.

For bearings, losses are $(58+6c)/72$ and $(21+42c)/72$. Aggregated View is lower than Same throughout the displayed range, yet it beats both accept-all and reject-all only for $7/10<c<20/21$ in this zero-request-cost scenario. Its retained-error fraction is still $21/30$. The all-round comparison also conceals a reversal: round 1 releases 20 fewer errors but retains one fewer correct diagnosis, making View worse than Same for $c>20/21$; round 2 releases 17 fewer errors and retains two more correct diagnoses. Deleting any one observed bearing group still reduces released errors (29--40 fewer), but deleting KI04 reverses the cost comparison above $c=35/37$. These deletions reweight existing evidence and do not test unseen bearings. The exact intersections are sample descriptions, not recommended deployment thresholds.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/decision_loss_2026-09-30.pdf}
\caption{Descriptive decision-loss sensitivity for existing frozen decisions, using Equation~\ref{eq:decision_loss} with $K_p=0$. These are cost-scenario lines, not interpolated risk--coverage curves or new policy operating points. Repeated decisions involve 27 hydraulic cycles or nine bearing specimens, respectively. Costs and downstream referral outcomes were not measured; equal request count does not ensure equal cost.}
\label{fig:decision_loss}
\end{figure}
'''
    discussion=r'''The new accounting makes an engineering decision conditional on explicit requirements. A practitioner must specify tolerated released errors, referral capacity, expected referral outcomes and request-related costs before selecting a check. With nonzero costs, the hydraulic View-versus-Same boundary becomes $c>(1+K_{\mathrm{View}}-K_{\mathrm{Same}})/7$. The study supplies observed outcome counts and service resources, but not those deployment valuations. In particular, the bearing aggregate advantage and the hydraulic round-2 tradeoff cannot certify a dependable diagnosis workflow. The actionable result is a reproducible way to reject an unsupported assurance claim and identify which evidence is missing, rather than a validated field benefit. The class-transition and cost analyses were added after observing outcomes and require fresh sources and service periods before any selected rule can be claimed to generalize.'''
    conclusion=r'''The additional class audit localizes most recurring mapping errors to one reciprocal class confusion, while descriptive cost accounting reveals round- and specimen-sensitive tradeoffs hidden by pooled screening scores. These empirical intersections must not be used as validated operating thresholds.'''
    costrows=[]
    for ds,label in [('hydraulic','Hydraulic'),('paderborn','Bearing')]:
        for p in ['Accept_all_F1','Same','View','Reject_all']:
            z=v['studies'][ds]['totals'][p]
            costrows.append([label,{'Accept_all_F1':'Accept all F1','Reject_all':'Reject all'}.get(p,p),z['N'],z['accepted_correct'],z['accepted_error'],z['rejected'],f"$({z['accepted_error']}+{z['rejected']}c)/{z['N']}$"])
    costtable=table('Complete descriptive action counts and empirical losses with omitted request-related cost ($K_p=0$). Rejection is a separate outcome and supplies no corrected label.','tab:loss_counts','llrrrrl',r'Task & Action & $N$ & Correct & Error & Rejected & $L_p(c,0)$',costrows)
    roundrows=[]
    for z in v['round_loss_contrasts']:
        roundrows.append(['Hydraulic' if z['dataset']=='hydraulic' else 'Bearing',esc(z['subset']),z['N'],z['delta_correct'],z['delta_errors'],z['delta_rejections'],z['c_exact'] or 'all $c$'])
    roundtable=table('View-minus-Same count differences across both rounds and separately by round. The last column is the exact zero-request-cost tie, including ties outside the displayed cost range; it is not a fitted policy threshold.','tab:loss_rounds','llrrrrl',r'Task & Subset & $N$ & $\Delta$correct & $\Delta$error & $\Delta$reject & Tie $c$',roundrows)
    delrows=[[esc(z['excluded_group']),z['retained_N'],z['delta_correct_View_minus_Same'],z['delta_error_View_minus_Same'],z['delta_rejected_View_minus_Same'],z['c_exact'] or z['kind']] for z in deletion]
    deltable=table('Complete deletion sensitivity for nine observed test bearings. Each deletion removes four regimes and both rounds from the existing summary; no model is refitted and no new test specimen is introduced. All differences are View minus Same.','tab:bearing_deletion','lrrrrl',r'Deleted & $N$ & $\Delta$correct & $\Delta$error & $\Delta$reject & Tie $c$',delrows)
    supplement=[r'\clearpage\section*{Appendix H: Result-informed evidence and decision-loss audits}',r'\addcontentsline{toc}{section}{Appendix H: Evidence and decision-loss audits}',
       'These analyses were completed on 30 September 2026 after the earlier outcomes were known. No inference, classifier fitting, policy selection or new source acquisition is performed. The separate audit scripts verify frozen-source hashes and independently reconstruct the reported labels, repeat categories and gate actions. Their outputs retain all classes, rounds and source groups; computational checks are distinct from accountable human scientific verification.',
       methods.replace(r'\subsection{Result-informed evidence and decision-loss audits}',r'\subsection*{H.1 Outcome and loss definitions}'),
       mappingtable, 'The 20 stable-different anonymous IDs and all six labels for every case are recorded in the machine-readable evidence audit. The remaining categories comprise ten stable-same and 40 unstable cases; they must not be pooled as invariant controls. Other Sol mapping and context contrasts have zero cases meeting this finite-repeat criterion. Astra has one stable-different A-back/B-back case; no all-model absence claim is made.',
       r'\subsection*{H.2 Complete action counts and sensitivity}',costtable,roundtable,deltable,
       r'For bearings, 10,000 paired resamples with seed 20260929 keep all four regimes and both rounds of each physical bearing together, sampling three bearings with replacement within each of the three classes. The percentile bands in the accompanying CSV describe sensitivity to empirical specimen weights conditional on the recorded responses, fixed backgrounds and balanced classes. Shared requests still couple different bearings. These are not population confidence intervals, resampled service calls, independent device replications or calibrated uncertainty about costs. No hydraulic rig-level interval is estimated from a single rig. All hydraulic block/panel deletions and both datasets\textquotesingle{} source and round partitions remain in the CSV exports.',
       r'\subsection*{H.3 Interpretation and reconstruction}',discussion,
       r'\begingroup\raggedright The complete artifacts are \path{evidence_boundary_audit_2026-09-30.json} and the \path{decision_value_audit_2026-09-29} directory (analysis schema started 29 September; completed 30 September). They bind source hashes, full case IDs, all pairwise loss intersections, source/round summaries and sensitivity settings. No subgroup or cost value was chosen to claim a favorable deployment result.\par\endgroup',r'\FloatBarrier']
    return {'EVIDENCE_BOUNDARY_METHODS':methods,'MAPPING_TRANSITIONS':mapping,'DECISION_VALUE_RESULTS':results,'EVIDENCE_BOUNDARY_DISCUSSION':discussion,'DECISION_VALUE_CONCLUSION':conclusion},'\n\n'.join(supplement)
