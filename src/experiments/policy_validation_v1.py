"""Frozen new-data randomized paired-position and fixed-policy study.

Only run invokes a model; each pair has two isolated requests on a fixed grid.
No answer access in run/audit/freeze, no resume, retry, replacement or tuning.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.context_blocked_v1 import (need, read, digest, now, exclusive_json,
    common_metadata, group_audit, inventory, summary, token_count)
from src.experiments.run_revision_v3 import INSTRUCTIONS, run_group, write_json

RUNNER = 'src/experiments/run_revision_v3.py'
WRAPPER = 'src/experiments/policy_validation_v1.py'
SCORER = 'src/analysis/score_policy_validation_v1.py'
SEED = 9282026


def validate_dataset(d):
    need(set(d) in ({'task_text','allowed_labels','dictionary','panels'},
                    {'dataset_id','task_text','allowed_labels','dictionary','panels'}), 'Unexpected dataset fields')
    need(isinstance(d['task_text'], str) and d['task_text'], 'No task')
    labels = d['allowed_labels']
    need(len(labels) == len(set(labels)) == 3 and all(isinstance(x,str) for x in labels), 'Three labels required')
    need(len(d['dictionary']) == 28, '28 features required')
    for f in d['dictionary']:
        need(set(f) == {'name','unit'} and all(isinstance(v,str) and v for v in f.values()), 'Invalid dictionary')
    need(len({f['name'] for f in d['dictionary']}) == 28, 'Duplicate features')
    all_ids, panels = set(), set()
    for p in d['panels']:
        need(set(p) == {'panel_id','focal_ids','background_ids','records'}, 'Unexpected panel fields')
        need(p['panel_id'] not in panels and re.fullmatch(r'[a-z0-9_]+',p['panel_id']), 'Invalid panel ID')
        panels.add(p['panel_id'])
        ids = p['focal_ids'] + p['background_ids']
        need(ids and len(ids) == len(set(ids)) and not set(ids)&all_ids, 'Source IDs reused')
        all_ids.update(ids)
        need(set(ids) == {r['id'] for r in p['records']} and len(ids) == len(p['records']), 'Records/IDs differ')
        for r in p['records']:
            need(set(r) == {'id','values'} and re.fullmatch(r'[0-9a-f]{8}',r['id']), 'Private record fields/ID')
            need(len(r['values']) == 28 and all(isinstance(v,str) and re.fullmatch(r'-?\d+\.\d{6}',v) for v in r['values']), 'Invalid decimal data')
    need(panels, 'Empty panels')


def payload(d, panel, condition):
    lookup = {r['id']:r for r in panel['records']}
    ids = panel['focal_ids']+panel['background_ids'] if condition == 'F' else panel['background_ids']+panel['focal_ids']
    prompt = (d['task_text']+'\nValues follow the shared dictionary exactly. Treat each record independently; '
        'no demonstrations, class counts or reference labels are provided. Return only a JSON array '
        'in the input record order, with exactly the keys id and predicted for each record.\n\n'+
        json.dumps({'valid_labels':d['allowed_labels'],'dictionary':d['dictionary'],
                    'records':[lookup[i] for i in ids]},ensure_ascii=False,separators=(',',':')))
    group = {'dataset':'measured','condition':condition,'repeat':1,
             'samples':[{'id':i,'prompt':','.join(lookup[i]['values'])} for i in ids],
             'valid_labels':d['allowed_labels']}
    return prompt, group


def make_schedule(d):
    rng = random.Random(SEED)
    blocks = []
    for r in (1,2):
        current = [(p['panel_id'],r) for p in d['panels']]
        rng.shuffle(current)
        blocks.extend(current)
    pairs=[]
    for bi,(panel,rnd) in enumerate(blocks):
        for rep in (1,2,3):
            conditions=['F','R']
            rng.shuffle(conditions)
            names=[f'b{bi+1:02d}_p{rep}_{c.lower()}' for c in conditions]
            pairs.append({'index':len(pairs),'block':bi,'panel_id':panel,'round':rnd,
                          'repeat':rep,'conditions':conditions,'names':names})
    return pairs


def prepare(base, source, dataset_id):
    need(not base.exists(), 'Study exists; use a new version, never overwrite')
    d=read(source/'dataset.json'); validate_dataset(d)
    truth=read(source/'truth.json')
    labels=truth['labels']
    ids={r['id'] for p in d['panels'] for r in p['records']}
    need(set(labels)==ids and set(labels.values()) <= set(d['allowed_labels']), 'Invalid separate answer key')
    need(dataset_id in {'hydraulic','paderborn'}, 'Unknown dataset')
    expected=(3,9,9) if dataset_id=='hydraulic' else (1,36,12)
    need(len(d['panels'])==expected[0] and all((len(p['focal_ids']),len(p['background_ids']))==expected[1:] for p in d['panels']), 'Unexpected fixed panel design')
    pairs=make_schedule(d); panels={p['panel_id']:p for p in d['panels']}
    groups={}; hashes={}
    for pair in pairs:
        for name,c in zip(pair['names'],pair['conditions']):
            text,g=payload(d,panels[pair['panel_id']],c)
            g.update(dataset=dataset_id,repeat=pair['repeat'])
            groups[name]=g
            f=base/'inference/prompts'/f'{name}.txt'; f.parent.mkdir(parents=True,exist_ok=True); f.write_text(text)
            hashes[name]=digest(f)
    write_json(base/'inference/dataset.json',d)
    write_json(base/'inference/batches.json',{'groups':groups})
    write_json(base/'schedule.json',{'pairs':pairs})
    write_json(base/'answers/truth.json',truth)
    write_json(base/'source_audit.json',read(source/'audit.json'))
    code=[RUNNER,WRAPPER,SCORER,'src/experiments/context_blocked_v1.py']
    code += ['src/experiments/prepare_policy_hydraulic_v1.py'] if dataset_id=='hydraulic' else ['src/analysis/prepare_paderborn_v1.py']
    p={'schema':'policy-validation-v1','created_at':now(),'dataset_id':dataset_id,
       'model':'gpt-5.6-sol','reasoning_effort':'medium',
       'cli_version':subprocess.check_output(['codex','--version'],text=True).strip(),
       'seed':SEED,'rounds':2,'repeats_per_condition':3,'workers_max':2,
       'pair_period_seconds':180,'timeout_seconds':170,'max_elapsed_seconds':len(pairs)*180+10,
       'max_total_tokens':900000 if dataset_id=='hydraulic' else 1000000,
       'calls_scheduled':len(groups),'slots_scheduled':sum(len(g['samples']) for g in groups.values()),
       'source':str(source.relative_to(ROOT)),
       'study_type':'Exploratory prospectively frozen follow-up on new records, designed after earlier outcomes; not original preregistration.',
       'serialization':'New compact shared dictionary + six-decimal value arrays. No exact-payload replication of older studies.',
       'randomization':'Seeded panel permutation within each of two sequential rounds; each of three pairs independently assigns F/R to two near-simultaneous launch slots. Grid anchored to one monotonic start; next pair at fixed 180-second offsets.',
       'failure_policy':'One invocation/job, no retries/resume/substitution. Stop future pairs on any failed/invalid request, unknown usage, token dispatch threshold, elapsed limit, or a grid deadline missed by >3s. Running pair is drained. Full planned denominators and all not-attempted jobs retained. No score read until audited freeze.',
       'primary_context_endpoint':'Mean pairwise F-minus-R correct focal predictions/planned focal records, request failures assigned zero usable correct. Exact two-sided sign-randomization over all assigned pairs only if every pair dispatched, <=3s grid lateness, <=2s launch skew; otherwise descriptive only. Sharp null of no effect on either slot and no between-request interference, including load/carryover; eligibility and full-dispatch selection must be invariant to F/R assignment for exactness. These assumptions are not established by the timing checks. One request, not each output, is assignment unit; no causal mechanism or future-service claim.',
       'engineering_endpoint':'Fixed first-pair F prediction. Same accepts iff F1=F2; View iff F1=R2, same two-call budget, second checks share randomized near-simultaneous pair2. Report common-base error capture, false rejection, accepted correct/planned, selective risk, incremental discordant error/correct rejections. Three-call majority and unanimity within F and within R are secondary. No threshold, panel or repeat chosen using predictions.',
       'policy_failure':'A policy requiring any invalid/missing call abstains on all scheduled focal records; source failure separate from valid-output disagreement. Majority needs >=2 of 3 votes; 3-way tie abstains. First means frozen pair-index 1, never first completion.',
       'costs':'Known input+output tokens, cached input already included; call counts and summed client execution seconds by fixed subsets, not cache-zero costs. No API price, field maintenance cost or human correction rate inferred.',
       'scope':'Hydraulic:27 new focal cycles/27 background cycles from54 previously unused joint-condition blocks, one rig. Paderborn:36 focal records from9 bearings x4 regimes,12 background records from6 other bearings x2 regimes, one different rig/task; naturally damaged accelerated-life bearings. Do not pool case accuracies or treat regimes/repeats as independent devices.',
       'source_files_sha256':{str((source/f).relative_to(ROOT)):digest(source/f) for f in ['dataset.json','truth.json','audit.json']},
       'input_sha256':digest(base/'inference/batches.json'),'dataset_sha256':digest(base/'inference/dataset.json'),
       'schedule_sha256':digest(base/'schedule.json'),'answers_sha256':digest(base/'answers/truth.json'),
       'source_audit_sha256':digest(base/'source_audit.json'),'prompts_sha256':hashes,
       'code_sha256':{f:digest(ROOT/f) for f in code}}
    write_json(base/'protocol.json',p); (base/'protocol.sha256').write_text(digest(base/'protocol.json')+'\n')
    validate(base)
    return {'status':'prepared_not_run','calls':len(groups),'protocol_sha256':digest(base/'protocol.json')}


def validate(base):
    p=read(base/'protocol.json')
    need(digest(base/'protocol.json')==(base/'protocol.sha256').read_text().strip(),'Protocol changed')
    for f,h in p['code_sha256'].items(): need(digest(ROOT/f)==h,f'Frozen code changed: {f}')
    for f,key in [('inference/batches.json','input_sha256'),('inference/dataset.json','dataset_sha256'),('schedule.json','schedule_sha256'),('source_audit.json','source_audit_sha256')]:
        need(digest(base/f)==p[key],f'Frozen input changed: {f}')
    d=read(base/'inference/dataset.json'); validate_dataset(d)
    groups=read(base/'inference/batches.json')['groups']; pairs=read(base/'schedule.json')['pairs']
    need(pairs==make_schedule(d),'Assignment schedule changed')
    need(set(groups)=={n for pair in pairs for n in pair['names']},'Group inventory changed')
    panels={q['panel_id']:q for q in d['panels']}
    for pair in pairs:
        for n,c in zip(pair['names'],pair['conditions']):
            text,g=payload(d,panels[pair['panel_id']],c); g.update(dataset=p['dataset_id'],repeat=pair['repeat'])
            need(g==groups[n],'Group changed')
            f=base/'inference/prompts'/f'{n}.txt'
            need(f.read_text()==text and digest(f)==p['prompts_sha256'][n],'Prompt changed')
    return p,groups,pairs


def audit(base, frozen=False):
    p,groups,pairs=validate(base)
    need(not (base/'runner.lock').exists(),'Runner active/interrupted')
    rows=[]; sessions=set(); previous_end=None; stopped=False; required_stop=False
    for pair in pairs:
        rr=[group_audit(base,p,n,groups[n]) for n in pair['names']]
        attempted=[r for r in rr if r['status']!='not_attempted']
        need(len(attempted) in {0,2},'Partially dispatched pair; protocol deviation')
        if not attempted: stopped=True
        if attempted:
            need(not stopped and not required_stop,'Dispatch after required stop')
            for slot,n in enumerate(pair['names']):
                dispatch=read(base/'predictions'/p['model']/f'{n}_dispatch.json')
                need(dispatch['slot']==slot and dispatch['target_offset']==pair['index']*p['pair_period_seconds'],'Slot/grid assignment changed')
                need(dispatch['monotonic_offset']>=dispatch['target_offset'],'Dispatch before grid time')
            need(all(r['block']==pair['index'] for r in rr),'Wrong pair dispatch index')
            starts=[datetime.fromisoformat(r['started_at']) for r in rr]
            ends=[datetime.fromisoformat(r['completed_at']) for r in rr]
            need(all(s<=e for s,e in zip(starts,ends)),'Negative execution envelope')
            need(previous_end is None or min(starts)>=previous_end,'Pairs overlap')
            previous_end=max(ends)
        for r in rr:
            for sid in r['sessions']:
                need(sid not in sessions,'Session reuse'); sessions.add(sid)
            rows.append(r)
        if attempted:
            late_pair=any(read(base/'predictions'/p['model']/f'{n}_dispatch.json')['monotonic_offset']-pair['index']*p['pair_period_seconds']>3 for n in pair['names'])
            required_stop=late_pair or any(r['status']!='complete' or token_count(r['usage']) is None for r in rr) or sum(token_count(r['usage']) or 0 for r in rows)>=p['max_total_tokens']
    files=inventory(base)
    allowed={'execution_state.json'}|{f"{p['model']}/{n}_{s}" for n in groups for s in ['metadata.json','dispatch.json','request.json','results.json','attempt1_events.jsonl','attempt1_response.txt','attempt1_stderr.txt']}
    need(set(files)<=allowed,'Unexpected prediction artifact')
    if frozen:
        f=read(base/'predictions/frozen_manifest.json')
        need(f['files']==files and f['protocol_sha256']==digest(base/'protocol.json'),'Frozen response mismatch')
    return p,groups,pairs,rows,files


def freeze(base):
    p,groups,pairs,rows,files=audit(base)
    m={'schema':'policy-validation-v1-freeze','frozen_at':now(),'protocol_sha256':digest(base/'protocol.json'),'files':files,**summary(rows)}
    exclusive_json(base/'predictions/frozen_manifest.json',m)
    return {k:v for k,v in m.items() if k!='files'}


def run(base):
    p,groups,pairs=validate(base)
    need(not (base/'predictions').exists(),'Never resume or retry an executed study')
    cli=shutil.which('codex'); need(cli,'Missing CLI')
    need(subprocess.check_output([cli,'--version'],text=True).strip()==p['cli_version'],'CLI changed')
    exclusive_json(base/'runner.lock',{'started_at':now()})
    folder=base/'predictions'/p['model']; folder.mkdir(parents=True)
    args=SimpleNamespace(output=base/'predictions',prompts_dir=base/'inference/prompts',runner_sha256=p['code_sha256'][RUNNER],
        protocol_sha256=digest(base/'protocol.json'),input_sha256=p['input_sha256'],effort=p['reasoning_effort'],
        timeout=p['timeout_seconds'],retries=1,cli_version=p['cli_version'],
        prompt_payloads={n:(base/'inference/prompts'/f'{n}.txt').read_text() for n in groups})
    state={'started_at':now(),'protocol_sha256':args.protocol_sha256,'pairs':[],'stop_reason':None}
    write_json(base/'predictions/execution_state.json',state)
    start=time.monotonic(); rows=[]; stop=None; execution_errors=[]
    def execute(n,index,slot,target):
        exclusive_json(folder/f'{n}_dispatch.json',{'group':n,'block':index,'slot':slot,
            'started_at':now(),'monotonic_offset':time.monotonic()-start,'target_offset':target,'protocol_sha256':args.protocol_sha256})
        try: run_group(args,cli,p['model'],n,groups[n])
        except Exception:
            need((folder/f'{n}_metadata.json').exists(),f'Interrupted job without metadata: {n}')
        return group_audit(base,p,n,groups[n])
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for pair in pairs:
                target=pair['index']*p['pair_period_seconds']
                while time.monotonic()-start < target:
                    time.sleep(max(0,min(1,target-(time.monotonic()-start))))
                if time.monotonic()-start > target+3: stop='grid_deadline_missed'
                if sum(token_count(r['usage']) or 0 for r in rows)>=p['max_total_tokens']: stop='token_dispatch_threshold'
                if time.monotonic()-start+p['timeout_seconds']>p['max_elapsed_seconds']: stop='elapsed_limit'
                if stop: break
                fs=[pool.submit(execute,n,pair['index'],slot,target) for slot,n in enumerate(pair['names'])]
                rr=[]
                for future in fs:
                    try: rr.append(future.result())
                    except Exception as exc: execution_errors.append(type(exc).__name__+': '+str(exc))
                rows.extend(rr)
                if execution_errors: stop='audit_or_execution_violation'
                state['audit_errors']=execution_errors
                if any(r['status']!='complete' for r in rr): stop='failed_or_invalid_request'
                if any(token_count(r['usage']) is None for r in rr): stop='unknown_token_usage'
                if not execution_errors and any(read(folder/f'{n}_dispatch.json')['monotonic_offset']-target>3 for n in pair['names']): stop='late_pair_dispatch'
                state['pairs'].append({'index':pair['index'],'finished_at':now(),'statuses':[r['status'] for r in rr]})
                state.update(stop_reason=stop,updated_at=now(),**summary(rows))
                write_json(base/'predictions/execution_state.json',state)
                print(json.dumps({'pair':pair['index']+1,'of':len(pairs),'statuses':[r['status'] for r in rr],'tokens':state['observed_total_tokens'],'stop':stop}),flush=True)
                if stop: break
        for pair in pairs:
            for n in pair['names']:
                if not (folder/f'{n}_metadata.json').exists():
                    if list(folder.glob(n+'_*')):
                        execution_errors.append('Orphaned execution retained without repair: '+n)
                        continue
                    m=common_metadata(base,p,n,groups[n]); m.update(status='not_attempted',attempts=[],reason=stop or 'unexpected_stop',updated_at=now())
                    exclusive_json(folder/f'{n}_metadata.json',m)
        state.update(stop_reason=stop,audit_errors=execution_errors,finished_at=now()); write_json(base/'predictions/execution_state.json',state)
    finally: (base/'runner.lock').unlink(missing_ok=True)
    if execution_errors: raise RuntimeError('Protocol violation retained; no retry/freeze/scoring: '+'; '.join(execution_errors))
    return summary(audit(base)[3])


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=['prepare','validate','run','audit','freeze'])
    a.add_argument('--root',required=True,type=Path); a.add_argument('--source',type=Path); a.add_argument('--dataset',choices=['hydraulic','paderborn'])
    args=a.parse_args(); base=args.root.resolve()
    if args.command=='prepare': out=prepare(base,args.source.resolve(),args.dataset)
    elif args.command=='run': out=run(base)
    elif args.command=='freeze': out=freeze(base)
    elif args.command=='audit': out=summary(audit(base,(base/'predictions/frozen_manifest.json').exists())[3])
    else:
        p,g,s=validate(base); out={'calls':len(g),'pairs':len(s),'answer_access':False}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
