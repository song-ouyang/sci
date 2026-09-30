"""Score ONLY frozen measured follow-up outputs; no network or inference."""
from pathlib import Path
import json
import csv
import sys
from datetime import datetime, timezone
from collections import Counter
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.run_hydraulic_v1 import audit_outputs, summary, validate, digest, group_audit, token_count

BASE = ROOT/'results/hydraulic_validation_v1'


def metric(truth, predictions, ids):
    if predictions is None:
        return {'n':len(ids),'valid':0,'correct':None,'accuracy':None,'macro_f1':None,'usable_correct':0}
    yy=[truth[i]['true_label'] for i in ids]; pp=[predictions[i] for i in ids]
    correct=sum(a==b for a,b in zip(yy,pp))
    return {'n':len(ids),'valid':len(ids),'correct':correct,'accuracy':correct/len(ids),
            'macro_f1':float(f1_score(yy,pp,labels=['No_leakage','Weak_leakage','Severe_leakage'],average='macro',zero_division=0)),
            'usable_correct':correct}


def pair(name, left, right, ids, truth, pred):
    row={'comparison':name,'left':left,'right':right,'n_scheduled':len(ids)}
    if pred.get(left) is None or pred.get(right) is None:
        return dict(row,n=0,flips=None,left_correct=None,right_correct=None,left_minus_right_pp=None)
    a=pred[left];b=pred[right]
    ca=sum(a[i]==truth[i]['true_label'] for i in ids);cb=sum(b[i]==truth[i]['true_label'] for i in ids)
    return dict(row,n=len(ids),flips=sum(a[i]!=b[i] for i in ids),left_correct=ca,right_correct=cb,
                left_minus_right_pp=100*(ca-cb)/len(ids))


def score(base=BASE):
    # Audit hashes and all terminal responses BEFORE opening the answer file.
    _, _, audited_records, _ = audit_outputs(base,require_frozen=True)
    frozen_audit=summary(audited_records)
    p,groups,blocks=validate(base)
    if digest(base/'answers/records.json')!=p['answers_sha256']:
        raise ValueError('Answer key changed')
    for rel,h in p['data_sha256'].items():
        if digest(ROOT/rel)!=h: raise ValueError('Derived data changed: '+rel)
    for rel,h in p['audit_sha256'].items():
        if digest(base/rel)!=h: raise ValueError('Study audit changed: '+rel)
    answers=json.loads((base/'answers/records.json').read_text())['panels']
    truth={r['id']:dict(r,panel=panel) for panel,rows in answers.items() for r in rows}
    focal=[r['id'] for r in answers['focal']]
    metrics=[];focal_metrics=[];cases=[];pred={};usage=[]
    for name,g in groups.items():
        a=group_audit(base,p,name,g)
        predictions=None if a['predictions'] is None else {r['id']:r['predicted'] for r in a['predictions']}
        pred[name]=predictions
        ids=[s['id'] for s in g['samples']]
        info={'group':name,'condition':g['condition'],'repeat':g['repeat'],'status':a['status']}
        request_metric = metric(truth,predictions,ids)
        if g['condition']=='singleton':
            request_metric['macro_f1']=None  # Report macro F1 for the full singleton policy below.
        metrics.append(dict(info,**request_metric))
        fi=[i for i in ids if i in focal]
        focal_metrics.append(dict(info,**metric(truth,predictions,fi)))
        usage.append(dict(info,usage=a.get('usage'),elapsed_seconds=a.get('elapsed_seconds')))
        for i in ids:
            t=truth[i]
            cases.append(dict(info,id=i,panel=t['panel'],cycle_1based=t['cycle_1based'],block_id=t['block_id'],
                    cooler=t['profile'][0],valve=t['profile'][1],pump=t['profile'][2],accumulator=t['profile'][3],
                    true_label=t['true_label'],predicted=None if predictions is None else predictions[i],
                    correct=None if predictions is None else int(predictions[i]==t['true_label'])))
    paired=[]
    comparisons=[('duplicate_focal','clean','clean_repeat'),('duplicate_panel_a','a_front','a_front_repeat'),
                 ('correct_pair_reorder','clean','pair_reverse'),('descriptor_error','clean','descriptor_rotation'),
                 ('fixed_focal_vs_a','clean','a_front'),('position_a','a_front','a_back'),
                 ('position_b','b_front','b_back'),('background_front','a_front','b_front'),('background_back','a_back','b_back')]
    for r in range(1,6):
        for name,left,right in comparisons:
            paired.append(dict(pair(name,f'{left}_r{r}',f'{right}_r{r}',focal,truth,pred),repeat=r))
    repeat_pairs=[]
    for condition in ['clean','a_front','pair_reverse','descriptor_rotation','a_back','b_front','b_back']:
        for left in range(1,6):
            for right in range(left+1,6):
                repeat_pairs.append(pair('between_execution_blocks',f'{condition}_r{left}',f'{condition}_r{right}',focal,truth,pred))
    singleton={}
    for name,g in groups.items():
        if g['condition']=='singleton' and pred[name] is not None:
            singleton.update(pred[name])
    singleton_summary={'scheduled':len(focal),'valid':len(singleton),'correct':sum(v==truth[i]['true_label'] for i,v in singleton.items()),
                       'note':'One call per case distributed across blocks; does not estimate singleton repeatability or isolate batch size.'}
    singleton_summary['accuracy']=singleton_summary['correct']/len(singleton) if singleton else None
    singleton_summary['macro_f1']=metric(truth,singleton,list(singleton))['macro_f1'] if singleton else None
    singleton_summary['primary_clean_paired_n']=len(singleton) if pred['clean_r1'] else 0
    singleton_summary['primary_clean_flips']=sum(pred['clean_r1'][i]!=v for i,v in singleton.items()) if pred['clean_r1'] else None
    total_tokens=sum(token_count(u['usage']) or 0 for u in usage)
    execution_wall=None
    statepath=base/'predictions/execution_state.json'
    if statepath.exists():
        state=json.loads(statepath.read_text())
        execution_wall=(datetime.fromisoformat(state['finished_at'])-datetime.fromisoformat(state['started_at'])).total_seconds()
    totals={'requests_scheduled':len(groups),'requests_complete':sum(m['status']=='complete' for m in metrics),
            'status_counts':dict(Counter(m['status'] for m in metrics)),
            'slots_scheduled':sum(m['n'] for m in metrics),'slots_valid':sum(m['valid'] for m in metrics),
            'usable_correct_slots':sum(m['usable_correct'] for m in metrics),'tokens_observed':total_tokens,
            'usage_unknown_requests':sum(u['status']!='not_attempted' and token_count(u['usage']) is None for u in usage),
            'request_elapsed_seconds_sum':sum(u['elapsed_seconds'] or 0 for u in usage),'execution_wall_seconds':execution_wall}
    subgroup=[]
    for condition in ['clean','descriptor_rotation','a_front','a_back','b_front','b_back']:
        for r in range(1,6):
            name=f'{condition}_r{r}'
            for cooler in [3,20,100]:
                ids=[i for i in focal if truth[i]['profile'][0]==cooler]
                subgroup.append({'condition':condition,'repeat':r,'cooler':cooler,**metric(truth,pred[name],ids)})
    result={'schema':'hydraulic-v1-scores','scored_at':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':digest(base/'protocol.json'),'frozen_manifest_sha256':digest(base/'predictions/frozen_manifest.json'),
            'answers_sha256':p['answers_sha256'],'totals':totals,'metrics':metrics,'focal_metrics':focal_metrics,
            'paired':paired,'between_block_pairs':repeat_pairs,'singleton':singleton_summary,
            'focal_by_cooler':subgroup,'resource_by_request':usage,
            'limits':['54 cycles from one rig; no independent-device replication','Five contiguous execution blocks; no future service guarantee',
                      'Descriptive paired counts; no independent-slot statistical tests','No model fitting or labelled examples; balanced evaluation is not field prevalence',
                      'Canonical byte identity is deterministic software behavior, not efficacy proof','All planned failures and stopped jobs retained'],
            'output_audit':frozen_audit}
    out=base/'analysis';out.mkdir(exist_ok=True)
    (out/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('cases',cases),('metrics',metrics),('focal_metrics',focal_metrics),('paired',paired),('between_block_pairs',repeat_pairs),('focal_by_cooler',subgroup)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Measured single-rig follow-up','',json.dumps(totals,indent=2),'',
           '| Condition | Focal correct R1 | R2 | R3 | R4 | R5 |','|---|---:|---:|---:|---:|---:|']
    for condition in ['clean','clean_repeat','pair_reverse','descriptor_rotation','a_front','a_front_repeat','a_back','b_front','b_back']:
        rows=[next(m for m in focal_metrics if m['condition']==condition and m['repeat']==r) for r in range(1,6)]
        lines.append('| '+condition+' | '+' | '.join('N/A' if m['correct'] is None else str(m['correct'])+'/'+str(m['n']) for m in rows)+' |')
    lines+=['','Descriptive only. Stable-cycle, balanced sampling from one rig; no device-level independence. All conditions and repeats are retained.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(totals))
    return result


if __name__=='__main__':
    score()
