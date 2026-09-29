"""Frozen cross-specimen baseline data; no training, predictions, or model calls.

Expand only the six original background bearings. Retain the original 36 focal
six-decimal inputs as the test set. HTTP failures may resume the same members;
never substitute records. The old preparation module and artifacts are read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis import prepare_paderborn_v1 as original

OLD = ROOT / 'data/paderborn_processed_v1'
OUT = ROOT / 'data/bearing_baseline_v1'
CODE = 'src/analysis/prepare_bearing_baseline_v1.py'
TEST = 'tests/test_bearing_baseline_data_v1.py'
need, load, write, digest = original.need, original.load, original.write, original.digest


def selected_records(old_records):
    backgrounds = [r for r in old_records if r['role'] == 'background']
    focal = [r for r in old_records if r['role'] == 'focal']
    bearings = {r['bearing']: r['label'] for r in backgrounds}
    need(len(backgrounds) == 12 and len(focal) == 36 and len(bearings) == 6, 'Unexpected original split')
    need(Counter(bearings.values()) == Counter({x: 2 for x in original.LABELS}), 'Development class groups')
    need(len({r['bearing'] for r in focal}) == 9 and not set(bearings) & {r['bearing'] for r in focal}, 'Test group overlap')
    old_members = {r['member'] for r in backgrounds}
    result = []
    for bearing, label in sorted(bearings.items()):
        for regime in original.REGIMES:
            repetitions = sorted(range(1, 21), key=lambda r: original.keyed('record-choice', bearing, regime, r))[:5]
            for repetition in repetitions:
                member = f'{bearing}/{regime}_{bearing}_{repetition}.mat'
                result.append({'id': original.keyed('anonymous-record', bearing, regime, repetition)[:8],
                    'label': label, 'bearing': bearing, 'regime': regime, 'repetition': repetition,
                    'member': member, 'archive_url': original.URL + bearing + '.rar',
                    'reuse_original': member in old_members})
    result.sort(key=lambda r: original.keyed('record-order', r['id']))
    need(len(result) == len({r['id'] for r in result}) == 120, 'Development count/ID collision')
    need(sum(r['reuse_original'] for r in result) == 12, 'Original background inclusion')
    need(not {r['id'] for r in result} & {r['id'] for r in focal}, 'Development/test record overlap')
    return result


def original_inputs(rows):
    paths = [OLD / x for x in ('selection.json', 'selection.sha256', 'dataset.json', 'truth.json', 'audit.json')]
    for row in rows:
        if row['reuse_original']:
            raw = OLD / 'source_members' / row['member']
            paths += [raw, raw.with_name(raw.name + '.source.json'), raw.with_name(raw.name + '.member.rar')]
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in paths}


def plan(out):
    need(not out.exists(), 'New version required: output directory already exists')
    selection = load(OLD / 'selection.json')
    dataset = load(OLD / 'dataset.json')
    need(digest((OLD / 'selection.json').read_bytes()) == (OLD / 'selection.sha256').read_text().strip(), 'Old selection hash')
    need(dataset['dictionary'] == selection['dictionary'] == original.dictionary(), 'Representation changed')
    rows = selected_records(selection['records'])
    codes = [CODE, TEST, 'src/analysis/prepare_paderborn_v1.py']
    p = {'schema': 'bearing-baseline-data-plan-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'code_sha256': {r: digest((ROOT / r).read_bytes()) for r in codes},
        'original_input_sha256': original_inputs(rows), 'seed': original.SEED,
        'development_records': rows, 'test_ids': dataset['panels'][0]['focal_ids'],
        'dictionary': dataset['dictionary'], 'development_bearings': sorted({r['bearing'] for r in rows}),
        'test_bearings': sorted({r['bearing'] for r in selection['records'] if r['role'] == 'focal'}),
        'selection_policy': 'Original six background specimens only; four original regimes each; first five repetitions in the original SHA256 record-choice ranking of1..20; no windows, feature/score selection, or replacements.',
        'test_policy': 'Exactly the original36 focal IDs, order, and28 six-decimal strings; no additional measurements of any test bearing.',
        'representation': selection['representation'], 'unit_policy': selection['unit_policy'],
        'spectrum': selection['spectrum'], 'numeric_policy': 'Both development and test model inputs are the28 six-decimal strings. Optional numeric arrays contain only those quantized values, not unrounded summaries.',
        'new_http_body_limit_bytes': original.LIMIT, 'old_reused_bytes_charged': 0, 'workers_max': 4,
        'transport_policy': 'Reuse12 original MAT/RAR/provenance triplets. Resume only the frozen120 member list. All new HTTP body reservations, including failed/interrupted attempts and headers, count toward300000000 bytes; no full-archive fallback. Original206/Range/ETag/headerCRC/memberCRC checks retained.',
        'scope': '120 labeled development records from6 specimens and36 fixed test records from9 other specimens, one rig. This is an outcome-informed additional supervised benchmark after prior LLM results, not a blind prospective external study. Labels and source metadata are not predictor features.',
        'new_model_calls': 0}
    write(out / 'plan.json', p)
    (out / 'plan.sha256').write_text(digest((out / 'plan.json').read_bytes()) + '\n')
    print(json.dumps({'plan': str(out / 'plan.json'), 'sha256': digest((out / 'plan.json').read_bytes()), 'development': 120, 'reused': 12, 'new': 108, 'test': 36}), flush=True)


def validate_plan(out):
    p = load(out / 'plan.json')
    need(digest((out / 'plan.json').read_bytes()) == (out / 'plan.sha256').read_text().strip(), 'Plan changed')
    for rel, expected in {**p['code_sha256'], **p['original_input_sha256']}.items():
        need(digest((ROOT / rel).read_bytes()) == expected, 'Frozen code/original input changed: ' + rel)
    need(p['development_records'] == selected_records(load(OLD / 'selection.json')['records']), 'Member selection changed')
    need(p['dictionary'] == original.dictionary(), 'Feature dictionary changed')
    return p


class Downloader(original.Downloader):
    """Reuse verified transport, but account from zero new HTTP bytes."""
    def __init__(self, out):
        if not (out / 'download_ledger.json').exists():
            write(out / 'download_ledger.json', {'limit_bytes': original.LIMIT,
                'prior_probe_bytes': 0, 'reserved_body_bytes': 0, 'requests': []})
        super().__init__(out)
        need(self.ledger['prior_probe_bytes'] == 0 and self.ledger['limit_bytes'] == original.LIMIT, 'Wrong new-download budget')
        need(self.ledger['reserved_body_bytes'] == sum(r['requested_bytes'] for r in self.ledger['requests']), 'New-download ledger inconsistent')


def fetch(out, workers=4):
    p = validate_plan(out)
    need(1 <= workers <= 4, 'Workers must be1..4')
    for row in p['development_records']:
        if not row['reuse_original']:
            continue
        original_path = OLD / 'source_members' / row['member']
        for suffix in ('', '.source.json', '.member.rar'):
            src = original_path.with_name(original_path.name + suffix)
            dst = out / 'source_members' / (row['member'] + suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                need(digest(dst.read_bytes()) == digest(src.read_bytes()), 'Reused artifact changed')
            else:
                shutil.copyfile(src, dst)
    downloader = Downloader(out)
    grouped = [[r for r in p['development_records'] if r['bearing'] == b] for b in p['development_bearings']]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(lambda records: original.fetch_bearing(out, records, downloader), grouped):
            print(json.dumps(row), flush=True)
    validate_plan(out)
    print(json.dumps({'reserved_new_http_body_bytes': downloader.ledger['reserved_body_bytes']}), flush=True)


def summarize_measurement(data):
    signals = {s['Name']: s for s in data['Y']}
    need(set(signals) == set(original.CHANNELS) and len(data['Y']) == 7, 'Unexpected signal inventory')
    diagnostics = []
    for name in original.CHANNELS:
        s = signals[name]
        need(np.asarray(s['Unit']).size == 0, 'Unexpected source amplitude unit')
        samples = np.asarray(s['Data'])
        timebase = np.asarray(data['X'][int(s['XIndex']) - 1]['Data'])
        need(samples.ndim == timebase.ndim == 1 and len(samples) == len(timebase) and len(samples) >= 2, 'Signal/timebase shape mismatch')
        need(np.isfinite(samples).all() and np.isfinite(timebase).all() and np.all(np.diff(timebase) > 0), 'Invalid signal/timebase')
        diagnostics.append({'name': name, 'source_unit': '', 'samples': len(samples), 'raster': s['Raster'],
            'time_start': float(timebase[0]), 'time_end': float(timebase[-1]),
            'median_step': float(np.median(np.diff(timebase))), 'min_step': float(np.min(np.diff(timebase))),
            'max_step': float(np.max(np.diff(timebase)))})
    return [f'{v:.6f}' for v in original.summarize({n: signals[n]['Data'] for n in original.CHANNELS})], diagnostics


def metadata_row(row, values):
    return {**{k: row[k] for k in ('id', 'label', 'bearing', 'regime', 'repetition', 'member')}, 'values': values}


def fixed_test_rows(selection, dataset, truth):
    panel = dataset['panels'][0]
    sources = {r['id']: r for r in selection['records']}
    values = {r['id']: r['values'] for r in panel['records']}
    rows = []
    for ident in panel['focal_ids']:
        r = sources[ident]
        need(r['role'] == 'focal' and truth['labels'][ident] == r['label'], 'Original test metadata mismatch')
        rows.append(metadata_row(r, values[ident]))
    need(len(rows) == 36 and len({r['bearing'] for r in rows}) == 9, 'Test dimensions')
    return rows


def extract(out):
    p = validate_plan(out)
    names = ['development.json', 'test.json', 'measurement_summary.npz', 'audit.json']
    need(not any((out / name).exists() for name in names), 'Extraction output exists; refusing replacement')
    records, evidence = [], []
    old_dataset = load(OLD / 'dataset.json')
    old_values = {r['id']: r['values'] for r in old_dataset['panels'][0]['records']}
    for row in p['development_records']:
        path = out / 'source_members' / row['member']
        proof_path = path.with_name(path.name + '.source.json')
        source = load(proof_path); raw = path.read_bytes(); h = source['member']
        rarpath = path.with_name(path.name + '.member.rar')
        rar = rarpath.read_bytes()
        need(h['name'] == row['member'] and source['archive']['url'] == row['archive_url'], 'Source identity mismatch')
        need(digest(raw) == source['mat_sha256'] and len(raw) == h['unpacked_size'] and zlib.crc32(raw) & 0xffffffff == h['file_crc32'], 'Raw MAT hash/size/CRC mismatch')
        need(digest(rar) == source['derived_single_member_rar_sha256'], 'Derived RAR changed')
        need(len(rar) == 20 + h['head_size'] + h['packed_size'] + len(original.END), 'Derived RAR size')
        need(original.parse_header(rar[20:20+h['head_size']], h['offset']) == h, 'Authentic member header changed')
        need(digest(rar[20+h['head_size']:-len(original.END)]) == source['packed_payload_sha256'], 'Packed payload changed')
        mat = {k: v for k, v in original.loadmat(path, simplify_cells=True).items() if not k.startswith('__')}
        need(list(mat) == [path.stem], 'Unexpected MAT root variable')
        data = mat[path.stem]
        values, channels = summarize_measurement(data)
        if row['reuse_original']:
            need(values == old_values[row['id']], 'Recomputed reused features differ from frozen input')
        records.append(metadata_row(row, values))
        evidence.append({**row, 'mat_sha256': source['mat_sha256'], 'provenance_sha256': digest(proof_path.read_bytes()),
            'derived_single_member_rar_sha256': source['derived_single_member_rar_sha256'], 'channels': channels,
            'recorded_datetime': str(data['Description']['General'].get('DateTime', ''))})
    test = fixed_test_rows(load(OLD / 'selection.json'), old_dataset, load(OLD / 'truth.json'))
    need([r['id'] for r in test] == p['test_ids'], 'Test order changed')
    need(not {r['bearing'] for r in records} & {r['bearing'] for r in test}, 'Specimen leakage')
    need(Counter((r['bearing'], r['regime']) for r in records) == Counter({(b, g): 5 for b in p['development_bearings'] for g in original.REGIMES}), 'Development balance changed')
    for name, rows in [('development', records), ('test', test)]:
        write(out / (name + '.json'), {'schema': 'bearing-baseline-' + name + '-v1', 'dictionary': p['dictionary'], 'records': rows})
    np.savez_compressed(out / 'measurement_summary.npz', development_X=np.asarray([r['values'] for r in records], dtype=float),
        test_X=np.asarray([r['values'] for r in test], dtype=float), development_ids=np.asarray([r['id'] for r in records]),
        test_ids=np.asarray([r['id'] for r in test]), feature_names=np.asarray([r['name'] for r in p['dictionary']]))
    validate_plan(out)
    audit = {'schema': 'bearing-baseline-preprocessing-audit-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest((out / 'plan.json').read_bytes()), 'code_sha256': p['code_sha256'],
        'original_input_sha256': p['original_input_sha256'], 'original_inputs_unchanged': True,
        'development_records': 120, 'development_bearings': 6, 'test_records': 36, 'test_bearings': 9,
        'reused_raw_records': 12, 'new_raw_records': 108, 'physical_rigs': 1,
        'development_test_specimens_disjoint': True, 'test_values_and_order_exactly_preserved': True,
        'all_reused_feature_strings_exact': True, 'all_source_amplitude_units_empty': True,
        'numeric_policy': p['numeric_policy'], 'records': evidence,
        'download_ledger': load(out / 'download_ledger.json'),
        'output_sha256': {name: digest((out / name).read_bytes()) for name in names if name != 'audit.json'},
        'new_model_calls': 0, 'scope': p['scope']}
    need(audit['download_ledger']['reserved_body_bytes'] <= original.LIMIT and audit['download_ledger']['prior_probe_bytes'] == 0, 'Download budget changed')
    write(out / 'audit.json', audit)
    print(json.dumps({'status': 'complete', 'development_records': 120, 'test_records': 36,
        'new_http_reserved_bytes': audit['download_ledger']['reserved_body_bytes'], 'output_sha256': audit['output_sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'fetch', 'extract'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'plan': plan(args.output)
    elif args.command == 'fetch': fetch(args.output, args.workers)
    else: extract(args.output)
