"""Inference/audit/scoring for the separately frozen independent-source follow-up."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from threading import Event
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.run_revision_v3 import INSTRUCTIONS,parse_raw,run_group,sha,write_json
from src.experiments.prepare_independent_validation import transform,MAPS
BASE=ROOT/'results/independent_validation'
TERMINAL={'complete','invalid_response','invalid_tool_use','invalid_event_log','failed'}
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def digest(p):return sha(p.read_bytes())
def need(ok,message):
    if not ok:raise ValueError(message)


def validate(base):
    p=read(base/'protocol.json')
    need(digest(base/'protocol.json')==(base/'protocol.sha256').read_text().strip(),'Protocol hash changed')
    for n,h in p['code_sha256'].items():need(digest(ROOT/n)==h,'Frozen code changed: '+n)
    need(digest(base/'inference/batches.json')==p['input_sha256'],'Input hash mismatch')
    need(digest(base/'session_schedule.json')==p['schedule_sha256'],'Schedule changed')
    groups=read(base/'inference/batches.json')['groups'];schedule=read(base/'session_schedule.json')
    need(len(schedule)==p['planned_sessions']==84,'Unexpected planned requests')
    need(len({(j['model'],j['group']) for j in schedule})==len(schedule),'Duplicate jobs')
    need({(j['model'],j['group']) for j in schedule}=={(m,n) for m in p['models'] for n in groups},'Incomplete schedule')
    for n,g in groups.items():
        raw=(base/'inference/prompts'/f'{n}.txt').read_text()
        need(sha(raw.encode())==p['prompt_sha256'][n],'Prompt changed')
        need(json.loads(raw.split('\n\n',1)[1])=={'valid_labels':g['valid_labels'],'samples':g['samples']},'Prompt/JSON mismatch')
        need(all(set(s)=={'id','prompt'} for s in g['samples']),'Inference payload contains private fields')
        need(len({s['id'] for s in g['samples']})==g['n_samples'],'Duplicate sample ID')
    for r in [1,2,3]:
        clean=groups[f'clean_r{r}']['samples']
        need(len(clean)==70,'Main panel length')
        for c in MAPS:
            need(groups[f'{c}_r{r}']['samples']==[dict(id=s['id'],prompt=transform(s['prompt'],c)) for s in clean],'Mapping invariant failed')
        focal=groups[f'focal_only_r{r}']['samples'];need(len(focal)==21,'Focal size')
        need({s['id'] for s in focal}==set(p['focal_ids']),'Focal panel changed')
        need(all(s in clean for s in focal),'Focal numerical values changed')
        background=[]
        for k in ['a','b']:
            front=groups[f'{k}_front_r{r}']['samples'];back=groups[f'{k}_back_r{r}']['samples']
            need(len(front)==len(back)==42 and front[:21]==focal and back[-21:]==focal,'Position invariant')
            need(front[21:]==back[:21],'Background not retained')
            need(set(s['id'] for s in front[21:]).isdisjoint(s['id'] for s in clean),'Background overlaps main')
            background.append(set(s['id'] for s in front[21:]))
        need(background[0].isdisjoint(background[1]),'Background sets overlap')
    return p,groups,schedule


def run(base,workers):
    p,groups,schedule=validate(base)
    need(not (base/'predictions/frozen_manifest.json').exists(),'Already frozen')
    cli=shutil.which('codex');need(subprocess.check_output([cli,'--version'],text=True).strip()==p['cli_version'],'CLI version changed')
    lock=base/'runner.lock';lock.open('x').write(now())
    args=SimpleNamespace(output=base/'predictions',prompts_dir=base/'inference/prompts',runner_sha256=digest(ROOT/'src/experiments/run_revision_v3.py'),protocol_sha256=digest(base/'protocol.json'),input_sha256=p['input_sha256'],effort=p['reasoning_effort'],timeout=p['timeout_seconds'],retries=1,cli_version=p['cli_version'],prompt_payloads={n:(base/'inference/prompts'/f'{n}.txt').read_text() for n in groups})
    stop=Event()
    def execute(j):
        if stop.is_set():return
        name=j['group'];folder=args.output/j['model'];meta=folder/f'{name}_metadata.json'
        if meta.exists():
            need(read(meta)['status'] in TERMINAL,'Pending interrupted job')
            print('KEEP',j,read(meta)['status'],flush=True);return
        need(not list(folder.glob(f'{name}_*')),'Orphaned attempt; manual disposition required')
        try:run_group(args,cli,j['model'],name,groups[name])
        except Exception as exc:
            if meta.exists() and read(meta)['status'] in TERMINAL:
                print('TERMINAL',j,read(meta)['status'],flush=True)
                if read(meta)['status']=='failed':stop.set()
            else:stop.set();raise
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(execute,schedule))
    finally:lock.unlink(missing_ok=True)
    if stop.is_set():raise RuntimeError('Transport failure stopped pending jobs; retained all terminal evidence')


def audit_outputs(base, require_frozen=True):
    from src.analysis.score_revision_v3 import verify_events
    p, groups, schedule = validate(base)
    folder = base/'predictions'
    files = {str(f.relative_to(folder)):digest(f) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='frozen_manifest.json'}
    if require_frozen:
        frozen = read(folder/'frozen_manifest.json')
        need(frozen['files']==files, 'Frozen result inventory/hash mismatch')
        need(frozen['protocol_sha256']==digest(base/'protocol.json'), 'Frozen protocol mismatch')
    verified=[]; sessions=set()
    for j in schedule:
        name=j['group']; g=groups[name]; d=folder/j['model']
        m=read(d/f'{name}_metadata.json')
        need(m['status'] in TERMINAL and len(m['attempts'])==1, f'Nonterminal/multiple attempts: {j}')
        need(m['batch']==name and m['n_samples']==g['n_samples'] and m['repeat']==g['repeat'] and m['condition']==g['condition'], 'Group metadata mismatch')
        need(m['tools_observed']==0 and m['answer_keys_read'] is False, 'Tool/answer isolation violation')
        need(m['protocol_sha256']==digest(base/'protocol.json') and m['input_sha256']==p['input_sha256'], 'Metadata provenance')
        need(m['model_requested']==j['model'] and m['cli_version']==p['cli_version'], 'Model/CLI mismatch')
        need(m['reasoning_effort']==p['reasoning_effort'] and m['runner_sha256']==p['code_sha256']['src/experiments/run_revision_v3.py'], 'Runner/effort mismatch')
        need(m['prompt_txt_sha256']==p['prompt_sha256'][name] and m['base_instructions_sha256']==p['instructions_sha256'], 'Prompt/instruction provenance')
        req=read(d/f'{name}_request.json')
        need(req=={'model':j['model'],'reasoning_effort':p['reasoning_effort'],'developer_instructions':INSTRUCTIONS,'user_message':(base/'inference/prompts'/f'{name}.txt').read_text()}, 'Request mismatch')
        a=m['attempts'][0]
        need(a['attempt']==1, 'More than one attempt')
        events=verify_events(d/a['events_file'])
        for sid in events['sessions']:
            need(sid not in sessions, 'Session reused');sessions.add(sid)
        preds=None
        raw_path=d/a['response_file']
        if m['status']=='complete':
            need(a['returncode']==0 and not a['timed_out'] and len(events['completed'])==1 and len(events['sessions'])==1, 'Incomplete execution')
            need(events['last_event_type']=='turn.completed' and len(events['final_messages'])==1, 'Incomplete final event')
            need(events['final_messages'][0].strip()==raw_path.read_text().strip(), 'Event/raw response mismatch')
            preds=parse_raw(raw_path.read_text(),g)
            result=d/f'{name}_results.json'
            need(read(result)==preds and digest(result)==m['prediction_sha256'], 'Prediction mismatch')
            need(events['completed'][0].get('usage')==a['usage'], 'Usage mismatch')
        else:
            need(not (d/f'{name}_results.json').exists(), 'Failed response has scored labels')
            if m['status']=='invalid_response':
                try:
                    parse_raw(raw_path.read_text(),g)
                except (ValueError,TypeError):
                    pass
                else:
                    raise ValueError('Claimed invalid response is valid')
        verified.append({'model':j['model'],'group':name,'condition':g['condition'],'repeat':g['repeat'],
                         'n':g['n_samples'],'status':m['status'],'predictions':preds,
                         'diagnostics':len(events['diagnostics']),'elapsed_seconds':a['elapsed_seconds'], 'usage':a['usage']})
    return p, groups, verified, files


def freeze(base):
    need(not (base/'runner.lock').exists(), 'Runner active')
    path=base/'predictions/frozen_manifest.json'
    need(not path.exists(), 'Already frozen; cannot replace')
    p,g,records,files=audit_outputs(base,False)
    write_json(path, {'schema':'mechanism-extension-freeze-v1','frozen_at':now(),
                     'protocol_sha256':digest(base/'protocol.json'), 'files':files,
                     'terminal_sessions':len(records),'valid_sessions':sum(r['status']=='complete' for r in records)})
    print('Frozen',len(records),'terminal jobs before scoring')


def score(base):
    # Inference audit and output freeze are checked before any answer-file access.
    p,groups,records,_=audit_outputs(base)
    need(digest(base/'answers/records.json')==p['answers_sha256'],'Answers changed')
    source=ROOT/'results/revision_v3/data/datasets.json'
    need(digest(source)==p['original_dataset_sha256'],'Original training features changed')
    answers=read(base/'answers/records.json');by_id={x['id']:x for x in answers}
    import numpy as np
    from sklearn.metrics import f1_score,confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from src.analysis.score_revision_v3 import wilson
    def metrics(ids,preds,labels):
        y=[by_id[i]['label'] for i in ids];correct=sum(a==b for a,b in zip(y,preds));n=len(y)
        return dict(n=n,correct=correct,accuracy=correct/n,macro_f1=float(f1_score(y,preds,labels=labels,average='macro',zero_division=0)),
                    accuracy_wilson95=wilson(correct,n),confusion_matrix=confusion_matrix(y,preds,labels=labels).tolist(),class_order=labels)
    scored=[];lookup={};cases=[]
    for r in records:
        pred=r['predictions'];lookup[(r['model'],r['condition'],r['repeat'])]={x['id']:x['predicted'] for x in (pred or [])}
        row={k:v for k,v in r.items() if k not in ['predictions','usage']}
        row.update(metrics([x['id'] for x in pred],[x['predicted'] for x in pred],groups[r['group']]['valid_labels']) if pred is not None else dict(correct=None,accuracy=None,macro_f1=None,accuracy_wilson95=None))
        scored.append(row)
        for s in groups[r['group']]['samples']:
            label=lookup[(r['model'],r['condition'],r['repeat'])].get(s['id'])
            cases.append(dict(model=r['model'],condition=r['condition'],repeat=r['repeat'],id=s['id'],source_group=by_id[s['id']]['group'],true_label=by_id[s['id']]['label'],predicted=label,correct=None if label is None else int(label==by_id[s['id']]['label'])))
    paired=[]
    def pair(m,r,left,right,id_pairs,kind):
        a=lookup[(m,left,r)];b=lookup[(m,right,r)]
        available=[(i,j) for i,j in id_pairs if i in a and j in b]
        row=dict(model=m,repeat=r,left=left,right=right,kind=kind,requested_n=len(id_pairs),paired_n=len(available))
        if available:
            ac=np.array([a[i]==by_id[i]['label'] for i,j in available],dtype=float)
            bc=np.array([b[j]==by_id[j]['label'] for i,j in available],dtype=float)
            dif=ac-bc;idx=np.random.default_rng(p['seed']).integers(0,len(dif),(10000,len(dif)))
            row.update(left_accuracy=float(ac.mean()),right_accuracy=float(bc.mean()),difference=float(dif.mean()),conditional_ci=np.quantile(dif[idx].mean(axis=1),[.025,.975]).tolist(),flips=sum(a[i]!=b[j] for i,j in available))
        else:row.update(left_accuracy=None,right_accuracy=None,difference=None,conditional_ci=None,flips=None)
        paired.append(row)
    for model in p['models']:
        for r in [1,2,3]:
            for c in MAPS[1:]:pair(model,r,'clean',c,[(i,i) for i in p['main_ids']],'mapping')
            pair(model,r,'clean','early',[(x['late'],x['early']) for x in p['early_late_pairs']],'same_run_phase')
            for left,right,kind in [('a_front','b_front','background'),('a_back','b_back','background'),('a_front','a_back','position'),('b_front','b_back','position'),('clean','focal_only','context_size'),('focal_only','a_front','context_size'),('focal_only','b_front','context_size')]:
                pair(model,r,left,right,[(i,i) for i in p['focal_ids']],kind)
    old=read(source)['datasets'];baselines=[]
    for dataset,conditions in [('tep',['clean','early']),('cmapss',['fd002'])]:
        train=old[dataset]['train'];X=[x['features'] for x in train];y=[x['label'] for x in train]
        for name,model in [('RF',RandomForestClassifier(n_estimators=300,random_state=20260926)),('SVM',make_pipeline(StandardScaler(),SVC(C=1,gamma='scale')))]:
            model.fit(X,y)
            for c in conditions:
                ids=[x['id'] for x in groups[f'{c}_r1']['samples']];pred=model.predict([by_id[i]['features'] for i in ids]).tolist()
                baselines.append(dict(model=name,condition=c,training=dataset+'_original_only',**metrics(ids,pred,old[dataset]['valid_labels'])))
    focal=[]
    for model in p['models']:
        for r in [1,2,3]:
            for c in ['clean','focal_only','a_front','a_back','b_front','b_back']:
                d=lookup[(model,c,r)];ids=[i for i in p['focal_ids'] if i in d]
                focal.append(dict(model=model,condition=c,repeat=r,requested_n=21,**(metrics(ids,[d[i] for i in ids],old['tep']['valid_labels']) if ids else dict(n=0,correct=None,accuracy=None))))
    result=dict(schema='independent-validation-scores-v1',scored_at=now(),protocol_sha256=digest(base/'protocol.json'),frozen_manifest_sha256=digest(base/'predictions/frozen_manifest.json'),answers_sha256=p['answers_sha256'],
        coverage=dict(planned=len(records),valid=sum(x['status']=='complete' for x in records),requested_labels=sum(x['n'] for x in records),valid_labels=sum(x['n'] for x in records if x['status']=='complete')),
        new_unique_source_units=172,new_observations=len(answers),metrics=scored,paired=paired,focal=focal,baselines=baselines,limitations=p['limitations'])
    write_json(base/'analysis/scores.json',result)
    import csv
    path=base/'analysis/cases.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    print(json.dumps(result['coverage'],indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['validate','run','status','freeze','score']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.command=='validate':validate(BASE);print('Label-free inputs and frozen code verified')
    elif a.command=='run':run(BASE,a.workers)
    elif a.command=='freeze':freeze(BASE)
    elif a.command=='score':score(BASE)
    else:
        p,g,s=validate(BASE);counts={}
        for j in s:
            path=BASE/'predictions'/j['model']/f"{j['group']}_metadata.json"
            key=read(path)['status'] if path.exists() else 'not_attempted';counts[key]=counts.get(key,0)+1
        print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
