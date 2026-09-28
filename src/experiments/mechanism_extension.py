"""Separate, post-v3 exploratory extension. Preparation/inference never read labels.

Commands: prepare, validate, run, status, freeze, score. Never edits v3 inputs/results.
The transport is the unchanged v3 runner; this protocol and orchestrator are hashed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run_revision_v3 import INSTRUCTIONS, parse_raw, run_group, sha, write_json

BASE = ROOT / 'results/mechanism_extension'
SOURCE = ROOT / 'results/revision_v3'
MODELS = ['gpt-5.6-sol', 'gpt-6-astra']
SEED = 20260927
CONDITIONS = ['clean', 'rotation', 'pair_shuffle', 'anonymous', 'random_mapping', 'sample_shuffle']
SOURCE_HASH = '48b648d949fac2119222007b8bba7e95c1a2019281df5a17c95fa4f2f59bd8d8'
CODE = ['src/experiments/mechanism_extension.py', 'src/experiments/run_revision_v3.py',
        'src/analysis/score_revision_v3.py']
TERMINAL = {'complete', 'invalid_response', 'invalid_tool_use', 'invalid_event_log', 'failed'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return sha(path.read_bytes())


def split_prompt(prompt):
    intro, body = prompt.split('\n\n', 1)
    rows = [line.rsplit(': ', 1) for line in body.splitlines()]
    need(len(rows) == 52 and all(len(r) == 2 for r in rows), 'Expected 52 exact sensor rows')
    need(all(re.fullmatch(r'-?\d+\.\d{4}', r[1]) for r in rows), 'Unexpected numeric precision')
    return intro, rows


def transform(prompt, condition):
    intro, rows = split_prompt(prompt)
    order = list(range(52))
    rng = random.Random(SEED)
    rng.shuffle(order)
    # One reproducible global derangement, identical across observations and repeats.
    derangement = list(range(52))
    rng = random.Random(SEED + 1)
    while True:
        rng.shuffle(derangement)
        if all(i != j for i, j in enumerate(derangement)):
            break
    if condition in {'clean', 'sample_shuffle'}:
        result = rows
    elif condition == 'rotation':
        result = [[rows[(i + 1) % 52][0], r[1]] for i, r in enumerate(rows)]
    elif condition == 'pair_shuffle':
        result = [rows[i] for i in order]
    elif condition == 'anonymous':
        # Remove identity, name AND unit. Change only the now-false row description.
        intro = intro.replace('Each following line names one process variable and its unit.',
                              'Each following line gives one anonymous variable in the original order; names and units are withheld.')
        result = [[f'field_{i+1:02d}', r[1]] for i, r in enumerate(rows)]
    elif condition == 'random_mapping':
        result = [[rows[derangement[i]][0], r[1]] for i, r in enumerate(rows)]
    else:
        raise ValueError(condition)
    return intro + '\n\n' + '\n'.join(': '.join(r) for r in result)


def make_groups(source_group):
    # Label-free selection for context checks: seven lowest salted ID hashes.
    selected = sorted(source_group['samples'], key=lambda s: sha(('context-v1:' + s['id']).encode()))[:7]
    selected_ids = {s['id'] for s in selected}
    selected = [s for s in source_group['samples'] if s['id'] in selected_ids]
    groups = {}
    for repeat in (1, 2, 3):
        for condition in CONDITIONS + ['subset_batch'] + [f'single_{i+1:02d}' for i in range(7)]:
            samples = deepcopy(source_group['samples'])
            if condition == 'subset_batch':
                samples = deepcopy(selected)
            elif condition.startswith('single_'):
                samples = deepcopy([selected[int(condition[-2:])-1]])
            else:
                for s in samples:
                    s['prompt'] = transform(s['prompt'], condition)
                if condition == 'sample_shuffle':
                    random.Random(SEED + 2).shuffle(samples)
            groups[f'tep_{condition}_r{repeat}'] = {
                'dataset': 'tep', 'condition': condition, 'repeat': repeat,
                'n_samples': len(samples), 'valid_labels': source_group['valid_labels'], 'samples': samples}
    return groups, [s['id'] for s in selected]


def prepare(base):
    need(not base.exists(), 'Extension directory already exists; never overwrite a prepared protocol')
    source = SOURCE/'inference/batches.json'
    need(digest(source) == SOURCE_HASH, 'Original frozen input hash mismatch')
    original = read(source)['groups']['tep_clean_r1']
    groups, selected = make_groups(original)
    header = (SOURCE/'inference/prompts/tep_clean_r1.txt').read_text().split('\n\n', 1)[0]
    cli = shutil.which('codex')
    need(cli is not None, 'Codex CLI required to freeze version before inference')
    version = subprocess.check_output([cli, '--version'], text=True).strip()
    base.mkdir(parents=True)
    write_json(base/'inference/batches.json', {'schema': 'mechanism-extension-label-free-v1', 'groups': groups})
    prompts = {}
    for name, g in groups.items():
        path = base/'inference/prompts'/f'{name}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + '\n\n' + json.dumps({'valid_labels':g['valid_labels'], 'samples':g['samples']}, ensure_ascii=False, indent=2) + '\n')
        prompts[name] = digest(path)
    schedule = []
    for index, model in enumerate(MODELS):
        jobs = list(groups)
        random.Random(SEED + 10 + index).shuffle(jobs)
        schedule += [{'model':model, 'group':name} for name in jobs]
    # Interleave two model-specific schedules, without adapting to outputs.
    n = len(groups)
    schedule = [schedule[j*n+i] for i in range(n) for j in range(2)]
    write_json(base/'session_schedule.json', schedule)
    protocol = {
        'version':'mechanism-extension-v1', 'frozen_at':now(),
        'study_type':'Exploratory follow-up designed after inspection of v3 results; not an independent confirmatory replication',
        'model_selection':'Sol and Astra selected for contrasting observed v3 mapping responses; no model-population inference',
        'models':MODELS, 'reasoning_effort':'medium', 'repeats':3, 'primary_repeat':1,
        'source_input_sha256':SOURCE_HASH, 'input_sha256':digest(base/'inference/batches.json'),
        'prompt_sha256':prompts, 'schedule_sha256':digest(base/'session_schedule.json'),
        'code_sha256':{f:digest(ROOT/f) for f in CODE}, 'cli_version':version,
        'instructions_sha256':sha(INSTRUCTIONS.encode()),
        'planned_sessions':len(schedule),
        'requested_classifications':sum(groups[j['group']]['n_samples'] for j in schedule),
        'unique_source_units':70, 'new_source_units':0, 'context_subset_ids':selected,
        'selection_rule':'Seven smallest SHA256(context-v1: + anonymous ID), retained in original relative order; no labels/predictions accessed',
        'conditions':{
            'clean':'Original 70 clean observations, fresh contemporaneous response',
            'rotation':'Original +1 descriptor/name/unit rotation, numeric order retained',
            'pair_shuffle':'Seeded joint permutation of descriptor-value pairs, semantics preserved',
            'anonymous':'Replace descriptor identity/name/unit by positional field labels; numeric order retained; neutral row-description sentence',
            'random_mapping':'One seeded noncyclic global descriptor derangement, all 52 numeric strings/positions retained',
            'sample_shuffle':'Original clean records, seeded different sample order; scores aligned by ID',
            'subset_batch':'Seven label-independently selected original clean records together',
            'single_01..07':'Same seven clean records, one fresh independent request each'},
        'permutation_seed':SEED, 'max_attempts_per_job':1, 'timeout_seconds':900,
        'failure_policy':'One CLI invocation per scheduled job, no semantic/transport rerun by wrapper. Internal reconnects remain visible. Any invalid response is N/A for the entire request. Resume only never-attempted jobs; interrupted jobs require documented disposition.',
        'scoring_policy':'Freeze all terminal records before reading answer labels. Pair by anonymous ID, never by row after sample shuffle. Report available-pair denominator and invalid requests. No repeat pooling or equivalence claims.',
        'contrasts':['clean-minus-each-full-condition on 70 IDs', 'full-clean-vs-subset-batch on seven selected IDs', 'subset-batch-vs-singleton on the same seven IDs', 'full-clean-vs-singleton on the same seven IDs'],
        'uncertainty':'Accuracy Wilson and paired run bootstrap conditional on observed outputs; 10000 resamples seed 20260927. Singleton outcomes from a repeat form a descriptive seven-case panel, not one session.',
        'limitations':['Same previously inspected 70 TEP runs; no new generalization evidence', 'Two selected services from one provider', 'One random permutation per transformation; no population of permutations', 'Only seven label-free-selected context probe cases, possibly missing classes', 'Three fresh sessions do not estimate a stable service variance', 'Descriptor invariance cannot prove memorization, causal reasoning or independence'],
    }
    write_json(base/'protocol.json', protocol)
    print(json.dumps({k:protocol[k] for k in ['planned_sessions','requested_classifications','unique_source_units','new_source_units','cli_version']}, indent=2))


def validate(base):
    p = read(base/'protocol.json')
    need(p['version'] == 'mechanism-extension-v1', 'Wrong protocol')
    need(digest(SOURCE/'inference/batches.json') == p['source_input_sha256'] == SOURCE_HASH, 'Original inputs changed')
    for f,h in p['code_sha256'].items():
        need(digest(ROOT/f) == h, f'Frozen code changed: {f}')
    need(digest(base/'inference/batches.json') == p['input_sha256'], 'Extension input changed')
    need(digest(base/'session_schedule.json') == p['schedule_sha256'], 'Schedule changed')
    groups = read(base/'inference/batches.json')['groups']
    expected, selected = make_groups(read(SOURCE/'inference/batches.json')['groups']['tep_clean_r1'])
    need(groups == expected and selected == p['context_subset_ids'], 'Transformation/selection invariants failed')
    schedule = read(base/'session_schedule.json')
    need(len(schedule) == p['planned_sessions'] == len(groups)*len(MODELS), 'Bad schedule length')
    need({(j['model'],j['group']) for j in schedule} == {(m,n) for m in MODELS for n in groups}, 'Schedule missing/duplicate job')
    for name,g in groups.items():
        raw = (base/'inference/prompts'/f'{name}.txt').read_bytes()
        need(sha(raw) == p['prompt_sha256'][name], f'Prompt hash: {name}')
        need(json.loads(raw.decode().split('\n\n',1)[1]) == {'valid_labels':g['valid_labels'],'samples':g['samples']}, 'TXT/JSON mismatch')
    return p, groups, schedule


def status(base):
    p, groups, schedule = validate(base)
    counts = {}
    for j in schedule:
        path = base/'predictions'/j['model']/f"{j['group']}_metadata.json"
        s = read(path)['status'] if path.exists() else 'not_attempted'
        counts[s] = counts.get(s,0)+1
    return {'planned':p['planned_sessions'], 'statuses':counts}


def run(base, workers):
    p, groups, schedule = validate(base)
    need(not (base/'predictions/frozen_manifest.json').exists(), 'Already frozen; no new inference allowed')
    cli = shutil.which('codex')
    need(cli and subprocess.check_output([cli,'--version'],text=True).strip() == p['cli_version'], 'CLI changed/unavailable')
    # Atomic process lock prevents two launchers from attempting the same job.
    lock = base/'runner.lock'
    with lock.open('x') as stream:
        stream.write(now()+'\n')
    try:
        args = SimpleNamespace(output=base/'predictions', prompts_dir=base/'inference/prompts',
            runner_sha256=digest(ROOT/'src/experiments/run_revision_v3.py'),
            protocol_sha256=digest(base/'protocol.json'), input_sha256=p['input_sha256'],
            effort=p['reasoning_effort'], timeout=p['timeout_seconds'], retries=1,
            cli_version=p['cli_version'], prompt_payloads={n:(base/'inference/prompts'/f'{n}.txt').read_text() for n in groups})
        def execute(j):
            folder = args.output/j['model']; name=j['group']
            meta = folder/f'{name}_metadata.json'
            if meta.exists():
                need(read(meta)['status'] in TERMINAL, f'Interrupted pending job: {j}')
                print('KEEP', j, read(meta)['status'], flush=True)
                return
            need(not list(folder.glob(f'{name}_*')), f'Orphaned artifacts: {j}')
            try:
                run_group(args, cli, j['model'], name, groups[name])
            except Exception as exc:
                # Continue semantic failures, but do not conceal interrupted/unknown jobs.
                if meta.exists() and read(meta)['status'] in TERMINAL:
                    print('TERMINAL FAILURE', j, str(exc), flush=True)
                    if read(meta)['status'] == 'failed':
                        raise RuntimeError('Transport failure: stop this model schedule; retain all records') from exc
                else:
                    raise
        # Each model consumes its prespecified order serially; two models may run concurrently.
        def run_model(model):
            for j in schedule:
                if j['model']==model:
                    execute(j)
        with ThreadPoolExecutor(max_workers=min(workers,len(MODELS))) as pool:
            list(pool.map(run_model, MODELS))
    finally:
        lock.unlink(missing_ok=True)
    print(json.dumps(status(base), indent=2))


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
    # Full audit completes before loading any answer key.
    p,groups,records,_=audit_outputs(base)
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from src.analysis.score_revision_v3 import wilson
    datasets=read(SOURCE/'data/datasets.json')['datasets']
    need(digest(SOURCE/'data/datasets.json')==read(SOURCE/'protocol.json')['dataset_sha256'], 'Answer provenance')
    # Data structure is explicitly checked, not inferred from prediction order.
    test=datasets['tep']['test']
    answers={r['id']:r['label'] for r in test}
    labels=next(iter(groups.values()))['valid_labels']
    by={}; metrics=[]
    for r in records:
        by[(r['model'],r['condition'],r['repeat'])]=r['predictions']
        pred=r['predictions']; ids=[] if pred is None else [x['id'] for x in pred]
        y=[answers[i] for i in ids]; yp=[] if pred is None else [x['predicted'] for x in pred]
        metrics.append({**{k:v for k,v in r.items() if k!='predictions'},
                        'accuracy':None if pred is None else accuracy_score(y,yp),
                        'macro_f1':None if pred is None else f1_score(y,yp,labels=labels,average='macro',zero_division=0),
                        'correct':None if pred is None else sum(a==b for a,b in zip(y,yp)),
                        'conditional_accuracy_wilson':None if pred is None else wilson(sum(a==b for a,b in zip(y,yp)),len(y))})
    contrasts=[]
    def paired(model,repeat,left,right,ids):
        def lookup(condition):
            if condition=='single_panel':
                return {x['id']:x['predicted'] for i in range(1,8) for x in (by[(model,f'single_{i:02d}',repeat)] or [])}
            return {x['id']:x['predicted'] for x in (by[(model,condition,repeat)] or [])}
        a,b=lookup(left),lookup(right)
        valid=[i for i in ids if i in a and i in b]
        row={'model':model,'repeat':repeat,'left':left,'right':right,'requested_n':len(ids),'paired_n':len(valid)}
        if valid:
            aa=np.array([a[i]==answers[i] for i in valid],dtype=float);bb=np.array([b[i]==answers[i] for i in valid],dtype=float)
            dif=aa-bb
            idx=np.random.default_rng(SEED).integers(0,len(valid),(10000,len(valid)))
            row.update(left_accuracy=float(aa.mean()),right_accuracy=float(bb.mean()),accuracy_difference=float(dif.mean()),
                       conditional_paired_ci=np.quantile(dif[idx].mean(axis=1),[.025,.975]).tolist(),
                       flips=sum(a[i]!=b[i] for i in valid),flip_rate=sum(a[i]!=b[i] for i in valid)/len(valid))
        else:
            row.update(left_accuracy=None,right_accuracy=None,accuracy_difference=None,conditional_paired_ci=None,flips=None,flip_rate=None)
        contrasts.append(row)
    all_ids=[s['id'] for s in groups['tep_clean_r1']['samples']]
    for m in MODELS:
        for r in (1,2,3):
            for c in CONDITIONS[1:]:paired(m,r,'clean',c,all_ids)
            for a,b in [('clean','subset_batch'),('subset_batch','single_panel'),('clean','single_panel')]:
                paired(m,r,a,b,p['context_subset_ids'])
    output={'schema':'mechanism-extension-scores-v1','scored_at':now(),'protocol_sha256':digest(base/'protocol.json'),
            'frozen_manifest_sha256':digest(base/'predictions/frozen_manifest.json'),'answers_sha256':digest(SOURCE/'data/datasets.json'),
            'coverage':{'planned':len(records),'valid':sum(r['status']=='complete' for r in records),
                        'requested_labels':sum(r['n'] for r in records),'valid_labels':sum(r['n'] for r in records if r['status']=='complete')},
            'context_subset_class_counts':{l:sum(answers[i]==l for i in p['context_subset_ids']) for l in labels},
            'metrics':metrics,'paired_contrasts':contrasts,'limitations':p['limitations']}
    write_json(base/'analysis/scores.json',output)
    lines=['# Exploratory mechanism extension','', 'Post-v3, same-source follow-up; no independent confirmation or deployment claim.','',str(output['coverage']),'',
           '| Model | Condition | R1 accuracy | R2 | R3 |','|---|---|---:|---:|---:|']
    for m in MODELS:
        for c in CONDITIONS+['subset_batch']:
            rr=[next(x for x in metrics if x['model']==m and x['condition']==c and x['repeat']==r) for r in (1,2,3)]
            fmt=lambda x:'N/A' if x['accuracy'] is None else f"{100*x['accuracy']:.1f}%"
            lines.append('| '+' | '.join([m,c,*[fmt(x) for x in rr]])+' |')
    lines+=['','Paired contrasts and seven-case context probes: see scores.json. Missing calls remain unavailable.']
    (base/'analysis/summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','validate','run','status','freeze','score'])
    ap.add_argument('--root',type=Path,default=BASE)
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();base=args.root.resolve()
    if args.command=='prepare':prepare(base)
    elif args.command=='validate':validate(base);print('Verified frozen inputs, transformations, code and schedule; no labels opened')
    elif args.command=='run':need(args.workers>0,'workers must be positive');run(base,args.workers)
    elif args.command=='status':print(json.dumps(status(base),indent=2))
    elif args.command=='freeze':freeze(base)
    else:score(base)

if __name__=='__main__':main()
