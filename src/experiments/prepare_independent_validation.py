"""Prepare a separately frozen follow-up; labels never enter inference payloads."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import random
import shutil
import subprocess
import numpy as np
import pyreadr
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.build_revision_v3 import cm_features,cm_label,TEP_COLS,digest,write_json
from src.experiments.prepare_revision_v3 import render
from src.experiments.run_revision_v3 import INSTRUCTIONS,sha

BASE=ROOT/'results/independent_validation'
SEED=20260928
MODELS=['gpt-5.6-sol','gpt-6-astra']
MAPS=['clean','rotation','random_1','random_2','random_3','same_unit_swap','pair_shuffle']


def transform(prompt,condition):
    intro,body=prompt.split('\n\n',1)
    rows=[line.rsplit(': ',1) for line in body.splitlines()]
    if len(rows)!=52:raise ValueError('Expected 52 fields')
    order=list(range(52))
    if condition=='rotation':order=order[1:]+order[:1]
    elif condition.startswith('random_'):
        rng=random.Random(SEED+int(condition[-1]))
        while True:
            rng.shuffle(order)
            if all(i!=j for i,j in enumerate(order)):break
    elif condition=='same_unit_swap':
        # Reactor level and product-separator level: both percent units.
        if not (rows[7][0].endswith('(%)') and rows[11][0].endswith('(%)')):raise ValueError('Unit mismatch')
        order[7],order[11]=order[11],order[7]
    elif condition=='pair_shuffle':random.Random(SEED+4).shuffle(order)
    elif condition!='clean':raise ValueError(condition)
    changed=[rows[j] if condition=='pair_shuffle' else [rows[j][0],rows[i][1]] for i,j in enumerate(order)]
    return intro+'\n\n'+'\n'.join(': '.join(row) for row in changed)


def prepare():
    if (BASE/'protocol.json').exists():raise ValueError('Prepared protocol already exists')
    oldpath=ROOT/'results/revision_v3/data/datasets.json'
    old=json.loads(oldpath.read_text())['datasets'];tep=old['tep'];cm=old['cmapss']
    prior={r['provenance']['simulation_run'] for s in ['train','test'] for r in tep[s]}
    # Conservatively exclude all run IDs touched by historical 50/100/200-row
    # per-class evenly spaced preparations, beyond the controlled 170 run IDs.
    historical={int(i)//500+1 for n in [50,100,200] for i in np.linspace(0,250000-1,n,dtype=int)}
    excluded=prior|historical
    candidates=sorted(set(range(1,501))-excluded)
    if len(candidates)<112:raise ValueError('Insufficient independent TEP sources')
    rng=random.Random(SEED);rng.shuffle(candidates)
    faults=[0,1,2,4,6,7,13];assign={f:candidates[i*16:(i+1)*16] for i,f in enumerate(faults)}
    times={run:rng.randint(200,500) for runs in assign.values() for run in runs}
    records=[];raw_hashes={};legacy_verified=0
    import pandas as pd
    legacy=pd.read_csv(ROOT/'data/tep/tep_processed.csv')
    for faulty in [False,True]:
        path=ROOT/f'data/tep/raw/TEP_{"Faulty" if faulty else "FaultFree"}_Training.RData'
        print('Reading raw simulation file',path.name,flush=True)
        loaded=pyreadr.read_r(str(path));frame=next(iter(loaded.values()))
        if list(frame.columns)!=['faultNumber','simulationRun','sample']+TEP_COLS:raise ValueError('Raw schema changed')
        for fault in faults[1:] if faulty else [0]:
            sub=frame.loc[frame.faultNumber==fault]
            if len(sub)!=250000:raise ValueError('Unexpected trajectory count')
            # Verify the current legacy preparation against exact source rows,
            # rather than trusting reconstructed row-to-run arithmetic alone.
            prior_rows=legacy.loc[legacy.fault_id==fault]
            n=len(prior_rows)
            if n not in [50,100,200]:raise ValueError('Unrecognized historical sampling')
            source=sub.iloc[np.linspace(0,len(sub)-1,n,dtype=int)]
            if not np.allclose(source[TEP_COLS].to_numpy(),prior_rows[[f'var_{i}' for i in range(52)]].to_numpy(),rtol=1e-9,atol=1e-9):
                raise ValueError('Cannot verify historical source exclusion')
            if not set(source.simulationRun.astype(int))<=historical:raise ValueError('Historical source gap')
            legacy_verified+=n
            small=sub.loc[sub.simulationRun.isin(assign[fault])].copy();del sub
            for j,run in enumerate(assign[fault]):
                role='main' if j<10 else 'background_a' if j<13 else 'background_b'
                label='Normal' if fault==0 else f'Fault_{fault}'
                for phase,t in [('late',times[run])]+([('early',40)] if role=='main' else []):
                    row=small.loc[(small.simulationRun==run)&(small['sample']==t)]
                    if len(row)!=1:raise ValueError('Nonunique source row')
                    values=[float(f'{v:.4f}') for v in row.iloc[0][TEP_COLS]]
                    sid=sha(f'independent:{SEED}:tep:{run}:{phase}'.encode())[:8]
                    prompt=tep['prompt_header']+'\n\n'+'\n'.join(f'{n}: {v:.4f}' for n,v in zip(tep['feature_display_names'],values))
                    records.append(dict(id=sid,dataset='tep',label=label,features=values,prompt=prompt,
                        group=f'TEP/Training/run-{run:03d}',role=role,phase=phase,
                        focal=role=='main' and j<3,provenance=dict(raw_file=str(path.relative_to(ROOT)),simulation_run=run,sample=t,fault_number=fault)))
            del small
        raw_hashes[str(path.relative_to(ROOT))]=digest(path)
        del frame,loaded;gc.collect()
    path=ROOT/'data/cmapss/raw/test_FD002.txt';rulp=ROOT/'data/cmapss/raw/RUL_FD002.txt'
    raw=np.loadtxt(path);rul=np.loadtxt(rulp,dtype=int);available={l:[] for l in cm['valid_labels']};short=[]
    for unit in np.unique(raw[:,0]).astype(int):
        trajectory=raw[raw[:,0]==unit];trajectory=trajectory[np.argsort(trajectory[:,1])]
        if len(trajectory)<30:short.append(int(unit));continue
        if not np.all(np.diff(trajectory[:,1])==1):raise ValueError('Noncontiguous engine')
        available[cm_label(rul[unit-1])].append((unit,trajectory))
    for label,choices in available.items():
        for unit,trajectory in random.Random(SEED).sample(choices,20):
            values=cm_features(trajectory[-30:]);sid=sha(f'independent:{SEED}:FD002:{unit}'.encode())[:8]
            prompt=cm['prompt_header']+'\n\n'+'\n'.join(f'{n}: {v:.{p}f}' for n,v,p in zip(cm['feature_display_names'],values,cm['feature_precisions']))
            records.append(dict(id=sid,dataset='cmapss_fd002',label=label,features=values,prompt=prompt,
                group=f'FD002/test/engine-{unit:03d}',role='transfer',phase='final',focal=False,
                provenance=dict(raw_file=str(path.relative_to(ROOT)),engine_id=int(unit),rul_label_only=int(rul[unit-1]),last_cycle=int(trajectory[-1,1]))))
    for p in [path,rulp,ROOT/'data/cmapss/raw/readme.txt']:raw_hashes[str(p.relative_to(ROOT))]=digest(p)
    rng=random.Random(SEED+101);rng.shuffle(records)
    if len({r['id'] for r in records})!=len(records):raise ValueError('ID collision')
    late=[r for r in records if r['dataset']=='tep' and r['phase']=='late' and r['role']=='main']
    focal=[r for r in late if r['focal']];a=[r for r in records if r['role']=='background_a'];b=[r for r in records if r['role']=='background_b']
    panels={c:[{**r,'prompt':transform(r['prompt'],c)} for r in late] for c in MAPS}
    panels.update(early=[r for r in records if r['phase']=='early'],focal_only=focal,
                  a_front=focal+a,a_back=a+focal,b_front=focal+b,b_back=b+focal,
                  fd002=[r for r in records if r['dataset']=='cmapss_fd002'])
    groups={};prompt_hashes={}
    for repeat in [1,2,3]:
        for condition,panel in panels.items():
            name=f'{condition}_r{repeat}';dataset='cmapss_fd002' if condition=='fd002' else 'tep'
            groups[name]=dict(dataset=dataset,condition=condition,repeat=repeat,n_samples=len(panel),
                valid_labels=cm['valid_labels'] if condition=='fd002' else tep['valid_labels'],
                samples=[dict(id=r['id'],prompt=r['prompt']) for r in panel])
            path=BASE/'inference/prompts'/f'{name}.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(render(groups[name]));prompt_hashes[name]=digest(path)
    write_json(BASE/'inference/batches.json',dict(groups=groups))
    write_json(BASE/'answers/records.json',records)
    exclusion=dict(controlled_excluded_run_ids=sorted(prior),historical_excluded_run_ids=sorted(historical),
                   legacy_records_verified=legacy_verified,new_tep_run_ids=sorted({r['provenance']['simulation_run'] for r in records if r['dataset']=='tep'}),
                   excluded_fd002_short_history_ids=short,fd002_eligible_class_counts={k:len(v) for k,v in available.items()},
                   historical_scope='Current legacy processed rows verified against raw data; conservatively also excludes 50/100/200-per-class historical evenly spaced source run IDs. FD002 not used by available historical experiment preparations.')
    write_json(BASE/'data_audit.json',exclusion)
    by_model=[]
    for i,m in enumerate(MODELS):
        names=list(groups);random.Random(SEED+200+i).shuffle(names);by_model.append([dict(model=m,group=n) for n in names])
    schedule=[jobs[i] for i in range(len(groups)) for jobs in by_model];write_json(BASE/'session_schedule.json',schedule)
    code=['src/experiments/prepare_independent_validation.py','src/experiments/independent_validation.py','src/experiments/run_revision_v3.py','src/analysis/score_revision_v3.py','src/experiments/build_revision_v3.py']
    protocol=dict(version='independent-validation-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        study_type='Prospectively specified follow-up after original findings; independent held-out sources, selected services; not an original preregistration or independent team replication.',
        models=MODELS,reasoning_effort='medium',repeats=3,primary_repeat=1,seed=SEED,
        original_dataset_sha256=digest(oldpath),raw_sha256=raw_hashes,answers_sha256=digest(BASE/'answers/records.json'),
        data_audit_sha256=digest(BASE/'data_audit.json'),input_sha256=digest(BASE/'inference/batches.json'),prompt_sha256=prompt_hashes,
        code_sha256={n:digest(ROOT/n) for n in code},schedule_sha256=digest(BASE/'session_schedule.json'),
        instructions_sha256=sha(INSTRUCTIONS.encode()),cli_version=subprocess.check_output([shutil.which('codex'),'--version'],text=True).strip(),
        planned_sessions=len(schedule),requested_classifications=sum(groups[j['group']]['n_samples'] for j in schedule),
        new_unique_source_units=172,new_tep_source_units=112,new_fd002_engines=60,new_observations=len(records),
        focal_ids=[r['id'] for r in focal],main_ids=[r['id'] for r in late],
        early_late_pairs=[dict(late=r['id'],early=next(x['id'] for x in records if x['group']==r['group'] and x['phase']=='early')) for r in late],
        focal_class_counts=dict(Counter(r['label'] for r in focal)),
        design='70 new TEP main runs (10/class), 21 focal cases (3/class), two disjoint 21-case background sets (3/class). Cross background A/B with front/back focal positions at fixed total size 42. Focal-only size 21 is a separate context-size contrast. Three noncyclic derangements; one same-unit local level swap; correct-pair reorder control. Early sample 40 uses the same 70 runs as late snapshots and adds no sources. FD002 60 engines (20/class) test fixed representation across six operating conditions.',
        baseline_policy='Fit RF(300 trees, original seed) and standardized RBF SVM(C=1,gamma=scale) only on original training features. No FD002 target adaptation or test tuning. Score new late/early TEP and FD002; do not equate supervised training information with LLM priors.',
        sample_size_rationale='Bounded precision-oriented follow-up, not powered significance trial: 70 main units have worst-case independent-binomial 95% half-width about 11.4 pp, 60 about 12.3 pp; focal 21 about 20 pp. Grouped output dependence can widen uncertainty. Three repeats report all realized outcomes, not stable service variance.',
        failure_policy='One invocation per job; no response repair or semantic retry. Invalid batch remains N/A. Stop on transport/authorization failure and retain never-attempted status. Never substitute repeats.',
        scoring_policy='Complete and freeze all terminal outputs before opening answer labels. Pair by ID or prespecified same-run early/late mapping. Per-repeat reporting; no pooled sample inflation or multiplicity-based discovery claims.',
        timeout_seconds=900,limitations=['Simulator replication and FD001-to-FD002 transfer are distinct; no physical plant validation','Two services selected after original observations; not equal-compute ranking','Mapping realizations are fixed examples, not all possible corruptions','Background identity and position are operational context interventions; no internal reasoning identification','No early-alarm latency, realistic prevalence or maintenance cost validation'])
    write_json(BASE/'protocol.json',protocol);(BASE/'protocol.sha256').write_text(digest(BASE/'protocol.json')+'\n')
    print(json.dumps({k:protocol[k] for k in ['planned_sessions','requested_classifications','new_unique_source_units','new_observations','focal_class_counts']},indent=2))


if __name__=='__main__':prepare()
