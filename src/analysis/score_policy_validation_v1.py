"""Offline pre-specified scoring; audit and response freeze precede answer access."""
import argparse
from collections import Counter
from datetime import datetime
import csv
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.policy_validation_v1 import audit, read, digest, need, now, token_count, summary


def exact_paired_p(differences):
    """All 2^n within-pair slot reassignments, retaining zero differences."""
    dist=Counter({0:1})
    for d in differences:
        new=Counter()
        for total,n in dist.items(): new[total+d]+=n; new[total-d]+=n
        dist=new
    target=abs(sum(differences))
    return sum(n for x,n in dist.items() if abs(x)>=target)/2**len(differences)


def decision(values, rule):
    if any(v is None for v in values): return None,'required_call_missing'
    if rule=='single': return values[0],'accepted'
    if rule=='agree': return (values[0],'accepted') if len(set(values))==1 else (None,'disagreement')
    if rule=='majority':
        label,count=Counter(values).most_common(1)[0]
        return (label,'accepted') if count>=2 else (None,'tie')
    raise ValueError(rule)


def summarize_policy(rows):
    counts=Counter(r['action'] for r in rows)
    planned=len(rows); ac=counts['accepted_correct']; ae=counts['accepted_error']; accepted=ac+ae
    base_correct=sum(r['base_correct'] is True for r in rows); base_errors=sum(r['base_correct'] is False for r in rows)
    rc=sum(r['output'] is None and r['base_correct'] is True for r in rows)
    re=sum(r['output'] is None and r['base_correct'] is False for r in rows)
    return {'planned':planned,'accepted':accepted,'accepted_correct':ac,'accepted_error':ae,
      'required_call_missing':sum(r['reason']=='required_call_missing' for r in rows),
      'disagreement_or_tie':sum(r['reason'] in {'disagreement','tie'} for r in rows),
      'base_correct':base_correct,'base_errors':base_errors,'base_missing':planned-base_correct-base_errors,
      'rejected_base_correct':rc,'rejected_base_error':re,
      'disagreement_rejected_base_correct':sum(r['reason'] in {'disagreement','tie'} and r['base_correct'] is True for r in rows),
      'disagreement_rejected_base_error':sum(r['reason'] in {'disagreement','tie'} and r['base_correct'] is False for r in rows),
      'corrected_base_error':sum(r['base_correct'] is False and r['action']=='accepted_correct' for r in rows),
      'harmed_base_correct':sum(r['base_correct'] is True and r['action']=='accepted_error' for r in rows),
      'coverage':accepted/planned if planned else None,'selective_error':ae/accepted if accepted else None,
      'usable_correct_fraction':ac/planned if planned else None,
      'error_capture':re/base_errors if base_errors else None,'false_rejection':rc/base_correct if base_correct else None,
      'rejection_break_even_cost_ignoring_extra_calls':re/(rc+re) if rc+re else None}


def score(base):
    p,groups,pairs,audited,files=audit(base,True)
    need(digest(base/'answers/truth.json')==p['answers_sha256'],'Answer key changed')
    truth=read(base/'answers/truth.json')['labels']
    d=read(base/'inference/dataset.json'); panels={x['panel_id']:x for x in d['panels']}
    records={r['group']:r for r in audited}
    predictions={r['group']:None if r['predictions'] is None else {x['id']:x['predicted'] for x in r['predictions']} for r in audited}
    request_rows=[]; pair_rows=[]; pred_rows=[]
    for pair in pairs:
        focal=panels[pair['panel_id']]['focal_ids']; bycond={}
        for slot,(n,c) in enumerate(zip(pair['names'],pair['conditions'])):
            r=records[n]; pred=predictions[n]; ids=[s['id'] for s in groups[n]['samples']]
            correct=sum(pred[i]==truth[i] for i in ids) if pred else 0
            focalcorrect=sum(pred[i]==truth[i] for i in focal) if pred else 0
            row={'group':n,'panel':pair['panel_id'],'round':pair['round'],'pair':pair['index'],'repeat':pair['repeat'],'slot':slot,'condition':c,
                 'status':r['status'],'planned_slots':len(ids),'valid_slots':len(ids) if pred else 0,'usable_correct':correct,
                 'focal_planned':len(focal),'focal_valid':len(focal) if pred else 0,'focal_usable_correct':focalcorrect,
                 'tokens':token_count(r['usage']),'elapsed_seconds':r.get('elapsed_seconds'),
                 'started_at':r.get('started_at'),'completed_at':r.get('completed_at')}
            if r['status']!='not_attempted':
                dispatch=read(base/'predictions'/p['model']/f'{n}_dispatch.json')
                row['grid_lateness_seconds']=dispatch['monotonic_offset']-dispatch['target_offset']
            else: row['grid_lateness_seconds']=None
            request_rows.append(row); bycond[c]=row
            pred_rows.extend({'group':n,'panel':pair['panel_id'],'round':pair['round'],'condition':c,'repeat':pair['repeat'],'id':i,'focal':i in focal,
                              'truth':truth[i],'predicted':None if pred is None else pred[i],'correct':None if pred is None else pred[i]==truth[i]} for i in ids)
        f,r=bycond['F'],bycond['R']
        skew=abs((datetime.fromisoformat(f['started_at'])-datetime.fromisoformat(r['started_at'])).total_seconds()) if f['started_at'] and r['started_at'] else None
        pair_rows.append({'pair':pair['index'],'panel':pair['panel_id'],'round':pair['round'],'repeat':pair['repeat'],
                          'both_dispatched':all(x['status']!='not_attempted' for x in bycond.values()),
                          'both_valid':all(x['status']=='complete' for x in bycond.values()),'launch_skew_seconds':skew,
                          'f_correct':f['focal_usable_correct'],'r_correct':r['focal_usable_correct'],'focal_n':len(focal),
                          'f_minus_r_correct':f['focal_usable_correct']-r['focal_usable_correct']})
    # Only the complete, scheduled assignment supports the predeclared randomization test.
    eligible=all(r['both_dispatched'] and r['launch_skew_seconds']<=2 for r in pair_rows) and all(r['grid_lateness_seconds'] is not None and r['grid_lateness_seconds']<=3 for r in request_rows)
    context={'pairs_scheduled':len(pair_rows),'pairs_dispatched':sum(r['both_dispatched'] for r in pair_rows),
       'pairs_valid':sum(r['both_valid'] for r in pair_rows),
       'mean_f_minus_r_usable_correct_fraction':statistics.mean(r['f_minus_r_correct']/r['focal_n'] for r in pair_rows),
       'focal_slots_each_condition':sum(r['focal_n'] for r in pair_rows),
       'f_usable_correct':sum(r['f_correct'] for r in pair_rows),'r_usable_correct':sum(r['r_correct'] for r in pair_rows),
       'randomization_schedule_eligible':eligible,
       'two_sided_sharp_null_p':exact_paired_p([r['f_minus_r_correct'] for r in pair_rows]) if eligible else None,
       'assumptions':'Conditional on these request slots: no cross-request interference/carryover, sharp null for slot outcomes including failure, seeded assignments independent of outcomes; eligibility and complete-dispatch selection invariant to F/R assignment (not established by observed timing alone). Time-varying potential slot outcomes permitted, server timing/weights not observed. No population inference over output slots.'}
    policy_rows=[]; costs=[]; flips=[]; increments=[]
    for block in sorted({x['block'] for x in pairs}):
        pp=[x for x in pairs if x['block']==block]; focal=panels[pp[0]['panel_id']]['focal_ids']
        names={(q['conditions'][i],q['repeat']):q['names'][i] for q in pp for i in range(2)}
        rules={'Single_F':('single',[('F',1)]),'Single_R':('single',[('R',1)]),
               'Same':('agree',[('F',1),('F',2)]),'View':('agree',[('F',1),('R',2)]),
               'Majority_F':('majority',[('F',j) for j in (1,2,3)]),'Majority_R':('majority',[('R',j) for j in (1,2,3)]),
               'Unanimity_F':('agree',[('F',j) for j in (1,2,3)]),'Unanimity_R':('agree',[('R',j) for j in (1,2,3)])}
        br={}
        for policy,(rule,keys) in rules.items():
            ns=[names[k] for k in keys]; baseline=names[(keys[0][0],1)]
            blockrows=[]
            for i in focal:
                vals=[None if predictions[n] is None else predictions[n][i] for n in ns]
                out,reason=decision(vals,rule)
                basepred=None if predictions[baseline] is None else predictions[baseline][i]
                row={'block':block,'panel':pp[0]['panel_id'],'round':pp[0]['round'],'policy':policy,'id':i,'truth':truth[i],
                     'base_prediction':basepred,'base_correct':None if basepred is None else basepred==truth[i],
                     'output':out,'reason':reason,'action':'rejected' if out is None else ('accepted_correct' if out==truth[i] else 'accepted_error')}
                policy_rows.append(row); blockrows.append(row)
            br[policy]={r['id']:r for r in blockrows}
            costs.append({'block':block,'panel':pp[0]['panel_id'],'round':pp[0]['round'],'policy':policy,
                'planned_requests':len(ns),'attempted_requests':sum(records[n]['status']!='not_attempted' for n in ns),
                'known_tokens':sum(token_count(records[n]['usage']) or 0 for n in ns),
                'unknown_token_calls':sum(records[n]['status']!='not_attempted' and token_count(records[n]['usage']) is None for n in ns),
                'client_service_seconds_sum':sum(records[n].get('elapsed_seconds',0) for n in ns),
                'check_start_gap_seconds':abs((datetime.fromisoformat(records[ns[-1]]['started_at'])-datetime.fromisoformat(records[ns[0]]['started_at'])).total_seconds()) if all(records[n].get('started_at') for n in ns) else None})
        for i in focal:
            same,view=br['Same'][i],br['View'][i]
            increments.append({'block':block,'panel':pp[0]['panel_id'],'round':pp[0]['round'],'id':i,'base_correct':same['base_correct'],
                'same_accept':same['output'] is not None,'view_accept':view['output'] is not None,
                'view_only_reject':same['output'] is not None and view['output'] is None,
                'same_only_reject':same['output'] is None and view['output'] is not None,
                'both_checks_valid':same['reason']!='required_call_missing' and view['reason']!='required_call_missing'})
        fn=[names[('F',j)] for j in (1,2,3)]; rn=[names[('R',j)] for j in (1,2,3)]
        for kind,combos in [('within_F',itertools.combinations(fn,2)),('within_R',itertools.combinations(rn,2)),('cross_FR',itertools.product(fn,rn))]:
            for a,b in combos:
                valid=predictions[a] is not None and predictions[b] is not None
                flips.append({'block':block,'panel':pp[0]['panel_id'],'round':pp[0]['round'],'kind':kind,'left':a,'right':b,'planned':len(focal),
                              'paired':len(focal) if valid else 0,'flips':sum(predictions[a][i]!=predictions[b][i] for i in focal) if valid else None})
    policies={k:summarize_policy([r for r in policy_rows if r['policy']==k]) for k in sorted({r['policy'] for r in policy_rows})}
    for k,v in policies.items():
        if k.startswith('Majority'):
            v['rejection_break_even_cost_ignoring_extra_calls']=None
        cc=[r for r in costs if r['policy']==k]
        v.update({s:sum(r[s] for r in cc) for s in ['planned_requests','attempted_requests','known_tokens','unknown_token_calls','client_service_seconds_sum']})
    incremental={f'{side}_{label}':sum(r[side] and r['base_correct'] is value and r['both_checks_valid'] for r in increments)
                 for side in ['view_only_reject','same_only_reject'] for label,value in [('base_error',False),('base_correct',True)]}
    result={'schema':'policy-validation-v1-scores','scored_at':now(),'dataset':p['dataset_id'],
        'protocol_sha256':digest(base/'protocol.json'),'frozen_manifest_sha256':digest(base/'predictions/frozen_manifest.json'),
        'totals':summary(audited),'context':context,'policies':policies,'incremental_gate_rejections':incremental,
        'scope':p['scope'],'limitations':['Prospectively fixed exploratory follow-up, one model alias and one service period per dataset.',
        'Policies selected before new results using prior studies; no policy tuning or positive-result replacement.',
        'Repeated record decisions are not independent bearings, rigs or field deployments. Dataset/task outputs analyzed separately.',
        'Rejecting a correct answer is a cost, and rejected errors are not assumed repaired by an untested human.',
        'Same/View second checks are near-simultaneous F2/R2; residual per-request service variability remains.',
        'Randomization p-value requires stated no-interference sharp-null assumptions; no claim to observe server timing or weights.']}
    result['totals'].update(usable_correct_slots=sum(r['usable_correct'] for r in request_rows),
        focal_valid_slots=sum(r['focal_valid'] for r in request_rows),focal_planned_slots=sum(r['focal_planned'] for r in request_rows))
    out=base/'analysis';out.mkdir(exist_ok=True);(out/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('requests',request_rows),('randomized_pairs',pair_rows),('predictions',pred_rows),('policy_cases',policy_rows),('policy_costs',costs),('flips',flips),('incremental_gate_cases',increments)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);score(a.parse_args().root.resolve())
