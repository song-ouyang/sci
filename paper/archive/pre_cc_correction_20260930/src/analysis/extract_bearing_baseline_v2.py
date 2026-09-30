"""Versioned floating-Nyquist boundary recovery; fixed members/features unchanged.

The last rFFT bin is mathematically at or below fs/2, but NumPy may express an
even-length Nyquist as32000.000000000004. Include every legal upper-band bin;
never alter the original frozen preparer, selection, source records or test data.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis import prepare_bearing_baseline_v1 as v1

old = v1.original
need, load, write, digest = old.need, old.load, old.write, old.digest
CODE = 'src/analysis/extract_bearing_baseline_v2.py'
TEST = 'tests/test_bearing_baseline_extraction_v2.py'


def summarize(channels, close_nyquist=True):
    """Same operations/order as legacy; only the last upper-bound mask differs."""
    values = []
    for name in old.CHANNELS:
        x = np.asarray(channels[name], dtype=np.float64)
        need(x.ndim == 1 and len(x) >= 2 and np.isfinite(x).all(), 'Invalid signal')
        values += [float(np.mean(x)), float(np.std(x, ddof=0))]
    for name in old.FAST:
        x = np.asarray(channels[name], dtype=np.float64)
        rms = float(np.sqrt(np.mean(x * x)))
        centered = x - np.mean(x)
        variance = float(np.mean(centered ** 2))
        need(rms > 0 and variance > 0, 'Undefined shape statistic')
        values += [rms, float(np.max(np.abs(x)) / rms), float(np.mean(centered ** 4) / variance ** 2)]
    x = np.asarray(channels['vibration_1'], dtype=np.float64)
    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    power[1:-1 if len(x) % 2 == 0 else None] *= 2
    frequency = np.fft.rfftfreq(len(x), 1 / 64000)
    need(frequency[-1] <= 32000 + 1e-8, 'Frequency exceeds mathematical Nyquist beyond floating tolerance')
    total = float(np.sum(power)); need(total > 0, 'Undefined spectrum')
    for lo, hi in old.BANDS:
        mask = frequency >= lo
        if hi != 32000: mask &= frequency < hi
        elif not close_nyquist: mask &= frequency <= hi
        values.append(float(np.sum(power[mask]) / total))
    need(len(values) == 28 and np.isfinite(values).all(), 'Feature validation')
    if close_nyquist: need(abs(sum(values[-5:]) - 1) < 1e-12, 'Spectral partition')
    return values


def measurement(data):
    signals = {s['Name']: s for s in data['Y']}
    need(set(signals) == set(old.CHANNELS) and len(data['Y']) == 7, 'Signal inventory')
    channels = {}
    diagnostics = []
    for name in old.CHANNELS:
        s = signals[name]; need(np.asarray(s['Unit']).size == 0, 'Unexpected source Unit')
        samples = np.asarray(s['Data']); timebase = np.asarray(data['X'][int(s['XIndex']) - 1]['Data'])
        need(samples.ndim == timebase.ndim == 1 and len(samples) == len(timebase) and len(samples) >= 2, 'Signal/timebase shape')
        need(np.isfinite(samples).all() and np.isfinite(timebase).all() and np.all(np.diff(timebase) > 0), 'Signal/timebase values')
        channels[name] = samples
        diagnostics.append({'name': name, 'source_unit': '', 'samples': len(samples), 'raster': s['Raster'],
            'time_start': float(timebase[0]), 'time_end': float(timebase[-1]),
            'median_step': float(np.median(np.diff(timebase))), 'min_step': float(np.min(np.diff(timebase))),
            'max_step': float(np.max(np.diff(timebase)))})
    corrected, prior = summarize(channels), summarize(channels, False)
    try:
        legacy = old.summarize(channels)
        need(legacy == prior, 'Copied non-corrected operations differ from immutable legacy')
        legacy_status = 'pass'
    except ValueError as exc:
        need(str(exc) == 'Spectral partition does not sum to1', 'Unexpected legacy failure: ' + str(exc))
        legacy_status = 'partition_assertion_failure'
    strings = [f'{x:.6f}' for x in corrected]
    prior_strings = [f'{x:.6f}' for x in prior]
    affected = [i for i, (a, b) in enumerate(zip(prior, corrected)) if a != b]
    need(set(affected) <= {27}, 'Correction changed a feature outside final frequency band')
    x = channels['vibration_1']; f = np.fft.rfftfreq(len(x), 1 / 64000)
    impact = {'legacy_status': legacy_status, 'vibration_samples': len(x), 'last_rfft_frequency_hz': float(f[-1]),
        'frequency_overshoot_hz': max(0., float(f[-1] - 32000)),
        'legacy_partition_sum_error': float(1 - sum(prior[-5:])),
        'high_precision_changed_feature_indices': affected,
        'six_decimal_changed_feature_indices': [i for i, (a, b) in enumerate(zip(prior_strings, strings)) if a != b],
        'last_band_before': prior[-1], 'last_band_after': corrected[-1],
        'last_band_string_before': prior_strings[-1], 'last_band_string_after': strings[-1]}
    return strings, diagnostics, impact


def read_member(base, row):
    path = base / 'source_members' / row['member']; proof_path = path.with_name(path.name + '.source.json')
    proof = load(proof_path); h = proof['member']; raw = path.read_bytes()
    rar = path.with_name(path.name + '.member.rar').read_bytes()
    need(h['name'] == row['member'] and proof['archive']['url'] == row['archive_url'], 'Member identity')
    need(digest(raw) == proof['mat_sha256'] and len(raw) == h['unpacked_size'] and zlib.crc32(raw) & 0xffffffff == h['file_crc32'], 'MAT hash/length/CRC')
    need(digest(rar) == proof['derived_single_member_rar_sha256'], 'RAR hash')
    need(len(rar) == 20 + h['head_size'] + h['packed_size'] + len(old.END), 'RAR length')
    need(old.parse_header(rar[20:20+h['head_size']], h['offset']) == h, 'Member header CRC/data')
    need(digest(rar[20+h['head_size']:-len(old.END)]) == proof['packed_payload_sha256'], 'Packed bytes hash')
    mat = {k: v for k, v in old.loadmat(path, simplify_cells=True).items() if not k.startswith('__')}
    need(list(mat) == [path.stem], 'MAT root variable')
    return mat[path.stem], proof, digest(proof_path.read_bytes())


def plan(out):
    v1.validate_plan(out)
    need(not (out / 'extraction_amendment_v2.json').exists(), 'Amendment already frozen')
    failure = load(out / 'extraction_failure.json')
    need(failure['failure_count'] == 1 and not any(failure['outputs_created']), 'Unexpected earlier failure')
    p = {'schema': 'bearing-baseline-extraction-amendment-v2', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'prior_plan_sha256': digest((out / 'plan.json').read_bytes()),
        'prior_failure_sha256': digest((out / 'extraction_failure.json').read_bytes()),
        'code_sha256': {r: digest((ROOT / r).read_bytes()) for r in [CODE, TEST]},
        'change': 'For the final10000..32000Hz inclusive band, include all legal rFFT bins with frequency>=10000. rFFT has no mathematically above-Nyquist bins; a32000.000000000004Hz floating representation must not drop Nyquist. Other features, masks, one-sided weighting, normalization, records and source splits unchanged.',
        'observed_trigger': failure['failures'][0],
        'output_checks': 'Recompute all48 original records and require all28 six-decimal strings to equal frozen old inputs; retain original36 test strings verbatim. Compare corrected/unmodified operations on all120 development records; explicitly report changes and legacy assertion failures. No sample replacement or score access.',
        'new_model_calls': 0, 'new_downloads': 0}
    write(out / 'extraction_amendment_v2.json', p)
    (out / 'extraction_amendment_v2.sha256').write_text(digest((out / 'extraction_amendment_v2.json').read_bytes()) + '\n')
    print(json.dumps({'amendment_sha256': digest((out / 'extraction_amendment_v2.json').read_bytes())}), flush=True)


def validate(out):
    p = v1.validate_plan(out); a = load(out / 'extraction_amendment_v2.json')
    need(digest((out / 'extraction_amendment_v2.json').read_bytes()) == (out / 'extraction_amendment_v2.sha256').read_text().strip(), 'Amendment changed')
    for rel, h in a['code_sha256'].items(): need(digest((ROOT / rel).read_bytes()) == h, 'V2 code changed')
    need(digest((out / 'plan.json').read_bytes()) == a['prior_plan_sha256'] and digest((out / 'extraction_failure.json').read_bytes()) == a['prior_failure_sha256'], 'Prior frozen evidence changed')
    return p, a


def extract(out):
    p, amendment = validate(out)
    outputs = ['development.json', 'test.json', 'measurement_summary.npz', 'audit.json']
    need(not any((out / n).exists() for n in outputs), 'Outputs already exist')
    old_selection = load(v1.OLD / 'selection.json'); old_dataset = load(v1.OLD / 'dataset.json')
    original_hashes = {r['id']: r['mat_sha256'] for r in load(v1.OLD / 'audit.json')['records']}
    old_values = {r['id']: r['values'] for r in old_dataset['panels'][0]['records']}
    original_checks = []
    for row in old_selection['records']:
        data, proof, proof_hash = read_member(v1.OLD, row)
        need(proof['mat_sha256'] == original_hashes[row['id']], 'Original48 raw hash differs from frozen source audit')
        values, _, impact = measurement(data)
        need(values == old_values[row['id']], 'Original48 six-decimal features changed: ' + row['id'])
        original_checks.append({'id': row['id'], 'member': row['member'], 'all28_strings_equal': True,
            'mat_sha256': proof['mat_sha256'], 'impact': impact})
    records, evidence, impacts = [], [], []
    for row in p['development_records']:
        data, proof, proof_hash = read_member(out, row)
        values, channels, impact = measurement(data)
        if row['reuse_original']: need(values == old_values[row['id']], 'Reused feature strings changed')
        records.append(v1.metadata_row(row, values))
        evidence.append({**row, 'mat_sha256': proof['mat_sha256'], 'provenance_sha256': proof_hash,
            'derived_single_member_rar_sha256': proof['derived_single_member_rar_sha256'], 'channels': channels,
            'recorded_datetime': str(data['Description']['General'].get('DateTime', ''))})
        impacts.append({'id': row['id'], 'member': row['member'], 'reuse_original': row['reuse_original'], **impact})
    test = v1.fixed_test_rows(old_selection, old_dataset, load(v1.OLD / 'truth.json'))
    need([r['id'] for r in test] == p['test_ids'], 'Test order')
    need(not {r['bearing'] for r in records} & {r['bearing'] for r in test}, 'Specimen leakage')
    need(Counter((r['bearing'], r['regime']) for r in records) == Counter({(b, g): 5 for b in p['development_bearings'] for g in old.REGIMES}), 'Development balance')
    validate(out)
    for name, rows in [('development', records), ('test', test)]:
        write(out / (name + '.json'), {'schema': 'bearing-baseline-' + name + '-v1', 'dictionary': p['dictionary'], 'records': rows})
    np.savez_compressed(out / 'measurement_summary.npz', development_X=np.asarray([r['values'] for r in records], dtype=float),
        test_X=np.asarray([r['values'] for r in test], dtype=float), development_ids=np.asarray([r['id'] for r in records]),
        test_ids=np.asarray([r['id'] for r in test]), feature_names=np.asarray([r['name'] for r in p['dictionary']]))
    audit = {'schema': 'bearing-baseline-preprocessing-audit-v2', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest((out / 'plan.json').read_bytes()), 'extraction_amendment_sha256': digest((out / 'extraction_amendment_v2.json').read_bytes()),
        'code_sha256': {**p['code_sha256'], **amendment['code_sha256']}, 'original_input_sha256': p['original_input_sha256'],
        'original_inputs_unchanged': True, 'development_records': 120, 'development_bearings': 6,
        'test_records': 36, 'test_bearings': 9, 'reused_raw_records': 12, 'new_raw_records': 108, 'physical_rigs': 1,
        'development_test_specimens_disjoint': True, 'test_values_and_order_exactly_preserved': True,
        'all_reused_feature_strings_exact': True, 'original48_recomputed_all28_strings_equal': True,
        'original48_recomputation': original_checks, 'development_numeric_amendment_impact': impacts,
        'development_legacy_assertion_failures': sum(r['legacy_status'] != 'pass' for r in impacts),
        'development_high_precision_changed_records': sum(bool(r['high_precision_changed_feature_indices']) for r in impacts),
        'development_six_decimal_changed_records': sum(bool(r['six_decimal_changed_feature_indices']) for r in impacts),
        'all_source_amplitude_units_empty': True, 'numeric_policy': p['numeric_policy'], 'records': evidence,
        'download_ledger': load(out / 'download_ledger.json'),
        'output_sha256': {n: digest((out / n).read_bytes()) for n in outputs if n != 'audit.json'},
        'new_model_calls': 0, 'scope': p['scope']}
    need(audit['download_ledger']['reserved_body_bytes'] <= old.LIMIT and audit['download_ledger']['prior_probe_bytes'] == 0, 'New body budget')
    write(out / 'audit.json', audit)
    print(json.dumps({k: audit[k] for k in ['development_records', 'test_records', 'development_legacy_assertion_failures',
        'development_high_precision_changed_records', 'development_six_decimal_changed_records', 'output_sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=['plan', 'extract'])
    parser.add_argument('--output', type=Path, default=v1.OUT); args = parser.parse_args()
    if args.command == 'plan': plan(args.output)
    else: extract(args.output)
