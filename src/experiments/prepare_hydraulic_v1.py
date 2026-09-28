"""Prepare a result-blind, single-rig measured follow-up; never run inference.

All choices here are made after the simulation results and before measured-data
predictions. Selection uses source annotations, never classifier performance.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import random
import subprocess
import sys
import copy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.interface_contract import canonicalize, define_panel, render_requests

BASE = ROOT / 'results/hydraulic_validation_v1'
DATA = ROOT / 'data/uci_hydraulic_processed'
SEED = 20260928
LABELS = ['No_leakage', 'Weak_leakage', 'Severe_leakage']
CHANNELS = [f'PS{i}' for i in range(1, 7)] + ['EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1']
KINDS = [('pressure','bar')]*6 + [('motor power','W')] + [('volume flow','L/min')]*2 + [('temperature','deg C')]*4 + [('vibration','mm/s')]
HEADER = ('Classify internal pump leakage in each hydraulic test-rig observation. '
          'No_leakage means no internal leakage; Weak_leakage means weak internal leakage; '
          'Severe_leakage means severe internal leakage. Cooler, valve and accumulator '
          'conditions may also vary. Each feature is a mean or population standard deviation '
          'over one complete 60-second operating cycle (SD denotes population standard deviation). No labelled examples are provided. '
          'Use the supplied measurements. Do not assume a class distribution or derive '
          'labels from IDs, order or other observations. Return ONLY one JSON array of '
          '{"id":"...","predicted":"..."} objects, in the supplied sample order, '
          'with every sample exactly once and only the permitted labels.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(text):
    return hashlib.sha256(f'hydraulic-v1|{SEED}|{text}'.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def select_blocks(blocks, profile):
    """Use six shared nuisance settings in every pump x cooler stratum."""
    panels = {name: [] for name in ['focal', 'a', 'b']}
    counts = []
    used = set()
    strata = {(pump,cooler):[b for b in blocks if b['condition_values'][0]==cooler
               and b['condition_values'][2]==pump and b['n_stable_cycles_flag_0']>=5]
               for pump in [0,1,2] for cooler in [3,20,100]}
    common = set.intersection(*[{(b['condition_values'][1],b['condition_values'][3]) for b in v} for v in strata.values()])
    if len(common) < 6:
        raise ValueError('Fewer than six nuisance settings shared by all nine strata')
    chosen = sorted(common,key=lambda c:key(f'nuisance|{c[0]}|{c[1]}'))[:6]
    for pump in [0, 1, 2]:
        for cooler in [3, 20, 100]:
            candidates = strata[pump,cooler]
            counts.append({'pump': pump, 'cooler': cooler, 'eligible_blocks': len(candidates),'chosen_nuisance_settings':chosen})
            for j, nuisance in enumerate(chosen):
                eligible = [b for b in candidates if (b['condition_values'][1],b['condition_values'][3])==nuisance]
                b = min(eligible,key=lambda b:key(f"block|{b['start_cycle_1based']}"))
                a, z = b['start_cycle_1based']-1, b['end_cycle_1based_inclusive']
                stable = np.flatnonzero(profile[a:z,4] == 0)+a
                index = int(stable[len(stable)//2])
                if b['block_id'] in used:
                    raise ValueError('Selected block twice')
                used.add(b['block_id'])
                panels[['focal','a','b'][j//2]].append({
                    'id': key(f'cycle|{index+1}')[:8], 'cycle_1based': index+1,
                    'block_id': b['block_id'], 'block_start': a+1, 'block_end': z,
                    'profile': profile[index].tolist(), 'true_label': LABELS[pump],
                })
    for panel in panels.values():
        panel.sort(key=lambda x:x['id'])
        assert len(panel) == 18
    ids = [r['id'] for panel in panels.values() for r in panel]
    if len(ids) != len(set(ids)):
        raise ValueError('Anonymous ID collision')
    return panels, counts


def registry():
    return [{'channel_id':c+'.'+stat,
             'descriptor':f'{c} {kind} {name} ({unit})'}
            for c,(kind,unit) in zip(CHANNELS,KINDS)
            for stat,name in [('mean','mean'),('sd','SD')]]


def perturb(record, mode):
    r = copy.deepcopy(record)
    fields = r['readings']
    if mode == 'pair_reverse':
        fields.reverse()
    elif mode == 'descriptor_rotation':
        desc = [f['descriptor'] for f in fields]
        for i,f in enumerate(fields):
            f['descriptor'] = desc[(i+2)%len(fields)]
    elif mode == 'same_unit_swap':
        for i in [0,1]:
            fields[i]['descriptor'], fields[i+2]['descriptor'] = fields[i+2]['descriptor'], fields[i]['descriptor']
    elif mode == 'hidden_value_swap':
        for i in [0,1]:
            fields[i]['value'],fields[i+2]['value'] = fields[i+2]['value'],fields[i]['value']
    elif mode != 'clean':
        raise ValueError(mode)
    return r


def prompt_sample(record):
    # Trusted acquisition IDs belong to the gateway and are not exposed as an
    # alternative uncorrupted sensor-identity channel to the tested model.
    return {'id':record['id'], 'prompt':'; '.join(f["descriptor"]+'='+f["value"] for f in record['readings'])}


def prepare():
    if BASE.exists():
        raise ValueError('Refusing to overwrite an existing study directory')
    meta = json.loads((DATA/'metadata.json').read_text())
    blocks = json.loads((DATA/'blocks.json').read_text())
    profile = np.asarray(json.loads((DATA/'profile.json').read_text())['rows'], dtype=int)
    data = np.load(DATA/'measurement_summary.npz', allow_pickle=False)
    if list(data['channel_names']) != CHANNELS or data['X'].shape != (2205,28):
        raise ValueError('Unexpected feature schema')
    panels, eligibility = select_blocks(blocks['joint_condition_blocks'], profile)
    reg = registry()
    records = {}
    for row in [r for p in panels.values() for r in p]:
        values = data['X'][row['cycle_1based']-1]
        records[row['id']] = canonicalize({'id':row['id'], 'readings':[
            dict(field, value=f'{float(value):.6f}') for field,value in zip(reg,values)]}, reg)
    panel_ids = {k:[r['id'] for r in v] for k,v in panels.items()}
    focal = [records[i] for i in panel_ids['focal']]
    a = [records[i] for i in panel_ids['a']]
    b = [records[i] for i in panel_ids['b']]
    versions = {'clean':focal, 'clean_repeat':focal,
                'pair_reverse':[perturb(r,'pair_reverse') for r in focal],
                'descriptor_rotation':[perturb(r,'descriptor_rotation') for r in focal],
                'a_front':focal+a,'a_front_repeat':focal+a,'a_back':a+focal,
                'b_front':focal+b,'b_back':b+focal}
    groups, prompts, schedule = {}, {}, []
    for repeat in range(1,6):
        jobs = []
        for condition, rows in versions.items():
            name = f'{condition}_r{repeat}'
            groups[name] = {'dataset':'hydraulic', 'condition':condition, 'repeat':repeat,
                            'samples':[prompt_sample(r) for r in rows], 'valid_labels':LABELS}
            jobs.append(name)
        for index, record in enumerate(focal):
            if index%5 == repeat-1:
                name = f'single_{record["id"]}_r{repeat}'
                groups[name] = {'dataset':'hydraulic', 'condition':'singleton', 'repeat':repeat,
                                'samples':[prompt_sample(record)], 'valid_labels':LABELS}
                jobs.append(name)
        random.Random(SEED+repeat).shuffle(jobs)
        schedule.append(jobs)
    for name,g in groups.items():
        prompts[name] = HEADER+'\n\n'+json.dumps({'valid_labels':g['valid_labels'],'samples':g['samples']},ensure_ascii=False,separators=(',',':'))+'\n'
    for repeat in range(1,6):
        assert prompts[f'clean_r{repeat}']==prompts[f'clean_repeat_r{repeat}']==prompts['clean_r1']
        assert prompts[f'a_front_r{repeat}']==prompts[f'a_front_repeat_r{repeat}']==prompts['a_front_r1']
    # Gateway checks are measured separately, with explicit blind spots.
    contract = []
    for mode in ['clean','pair_reverse','descriptor_rotation','same_unit_swap','hidden_value_swap']:
        accepted, rejected, exact = 0,0,0
        for original in records.values():
            try:
                out = canonicalize(perturb(original,mode),reg)
                accepted += 1
                exact += out==original
            except ValueError:
                rejected += 1
        contract.append({'mode':mode,'n':len(records),'accepted':accepted,'rejected':rejected,'canonical_equals_clean':exact})
    for panel in [focal,focal+a]:
        fixed = define_panel([r['id'] for r in panel],len(panel),reg)
        assert render_requests(panel,reg,fixed) == render_requests([perturb(r,'pair_reverse') for r in reversed(panel)],reg,fixed)
    write(BASE/'answers/records.json',{'panels':panels})
    write(BASE/'data_audit.json',{'single_rig':True,'cycles':54,'joint_condition_blocks':54,
          'source_units_not_independent_devices':True,'panels':panel_ids,'eligible_strata':eligibility,
          'selection':'Six hash-selected valve/accumulator combinations shared across all nine pump x cooler strata; >=5 stable cycles per block; middle stable cycle; same two nuisance combinations per panel. No outcome selection.',
          'cooler_segments':blocks['cooler_segments'], 'data_metadata_sha256':digest(DATA/'metadata.json'),
          'features':'14 physical channels, cycle mean and population SD, fixed six decimals; virtual/efficiency channels CE/CP/SE excluded',
          'target_change':'Cooler was a candidate in the draft. Before inference, raw annotations showed one contiguous time segment per cooler state. Pump leakage chosen with balanced cooler strata; no new predictions had been obtained.',
          'limits':['One rig; blocks/cycles remain dependent','Stable-cycle selection excludes transients','Public pretraining exposure unknown','No fitting or labelled prompt examples; no claimed independent-device generalization']})
    write(BASE/'contract_audit.json',{'diagnostic_inference':False,'n_records':54,'results':contract,
           'limitation':'Detection assumes independently trusted acquisition IDs; internally consistent value/channel misbinding is not detectable. Identical canonical bytes are software properties, not efficacy results.'})
    write(BASE/'inference/registry.json',reg)
    write(BASE/'inference/batches.json',{'groups':groups})
    write(BASE/'inference/panels.json',panel_ids)
    for name,payload in prompts.items():
        path=BASE/'inference/prompts'/f'{name}.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(payload)
    write(BASE/'schedule.json',{'blocks':schedule})
    code = ['src/experiments/prepare_hydraulic_v1.py','src/experiments/run_hydraulic_v1.py',
            'src/experiments/run_revision_v3.py','src/experiments/interface_contract.py','src/analysis/prepare_hydraulic_data.py',
            'src/analysis/score_hydraulic_v1.py']
    protocol = {'schema':'hydraulic-validation-v1','created_at':datetime.now(timezone.utc).isoformat(),
          'design':'Result-informed exploratory single-rig measured follow-up; frozen before new inference; not preregistered original study',
          'model':'gpt-5.6-sol','reasoning_effort':'medium','cli_version':subprocess.check_output(['codex','--version'],text=True).strip(),
          'timeout_seconds':900,'workers_max':2,'max_elapsed_seconds':14400,'max_total_tokens':1500000,
          'budget_note':'Tokens are a dispatch-stop threshold, not a hard generation cap; up to two in-flight requests can exceed it. CLI does not expose a reliable per-request output cap. Unknown usage stops dispatch.',
          'calls_scheduled':len(groups),'slots_scheduled':sum(len(g['samples']) for g in groups.values()),
          'seed':SEED,'time_blocks':5,'primary_repeat':1,'retries_per_job':0,
          'model_selection':'Selected from existing simulation sensitivity before any new measured-data outputs',
          'task':'Internal pump leakage: no, weak, severe; 54 measured cycles from 54 joint-condition blocks on ONE rig, focal/A/B panels18 each',
          'sampling':'All 9 pump x cooler strata represented by 2 cycles per panel, each from a different >=5-stable-cycle block. Six valve/accumulator settings are shared across strata, two per panel. Profile labels used for sampling only; fixed pump cycling order can still confound time.',
          'feature_policy':'14 physical sensors; mean/population SD over complete 60-s cycle; six decimal places; no target, other profile labels, stable flags, cycle numbers or CE/CP/SE in prompts',
          'conditions':list(versions)+['singleton'],
          'interventions':'pair_reverse reverses intact descriptor/value fields; descriptor_rotation reassigns descriptors by one physical channel (two scalar features) keeping numeric strings/positions; A/B front/back cross membership and position at n=36',
          'canonical_policy':'Trusted-ID metadata validation and canonical fields; fixed focal or focal+A panel is predefined, membership changes require a new panel. Rejected corrupt inputs produce no automatic label. Gateway detection is not a new classifier.',
          'controls':'clean/clean_repeat and a_front/a_front_repeat are byte-identical within all 5 contiguous execution blocks; no promise of independent calendar-day replication',
          'singleton_policy':'One call per focal cycle distributed round-robin across the five blocks; time/composition jointly change, not an isolated batch-size effect',
          'scoring_plan':'Freeze all terminal outputs before opening answers; whole-request validity; report all repeats separately, accuracy and macroF1, focal ID-paired flips, clean duplicates, request/slot coverage, usable-correct/planned, token usage and elapsed time. No p-values or pseudo-independent CIs; no subtraction of repeat flips as a net context effect.',
          'failure_policy':'One invocation per scheduled job; no semantic repair, low-score rerun, primary replacement or partial scoring; preserve failures and not-attempted budget/auth stops',
          'inference_access':'Fresh empty-directory CLI session, tools/plugins/project discovery disabled; inference code does not load answers',
          'answers_sha256':digest(BASE/'answers/records.json'),'input_sha256':digest(BASE/'inference/batches.json'),
          'schedule_sha256':digest(BASE/'schedule.json'),
          'prompts_sha256':{n:digest(BASE/'inference/prompts'/f'{n}.txt') for n in groups},
          'code_sha256':{n:digest(ROOT/n) for n in code},
          'data_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(DATA.iterdir()) if p.is_file()},
          'audit_sha256':{n:digest(BASE/n) for n in ['data_audit.json','contract_audit.json','inference/registry.json','inference/panels.json']}}
    assert protocol['calls_scheduled']==63 and protocol['slots_scheduled']==1278
    write(BASE/'protocol.json',protocol)
    (BASE/'protocol.sha256').write_text(digest(BASE/'protocol.json')+'\n')
    print(json.dumps({k:protocol[k] for k in ['calls_scheduled','slots_scheduled','model','created_at']}))


if __name__ == '__main__':
    prepare()
