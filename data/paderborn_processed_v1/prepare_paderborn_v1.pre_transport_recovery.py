"""Versioned Paderborn subset: bounded official Range downloads, no inference.

Freeze selection before fetching records; never select by observed signal/score.
Original archives are not downloaded in full. RAR4 nonsolid members are copied
with their authentic headers, decoded locally, and checked by original CRC32.
Amplitude Units are empty in the source MAT and are never guessed/calibrated.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
import urllib.request
import zlib
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/paderborn_processed_v1'
URL = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'
SEED = 20260928
LIMIT = 300_000_000
PROBE_BYTES = 2_427_525  # 4096-byte header probe + 84+2423345 member probe.
LABELS = ['healthy', 'outer_race_fault', 'inner_race_fault']
BEARINGS = {
    'healthy': ['K001', 'K002', 'K003', 'K004', 'K005'],
    'outer_race_fault': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'],
    'inner_race_fault': ['KI04', 'KI14', 'KI16', 'KI18', 'KI21'],
}
REGIMES = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']
CHANNELS = ['force', 'phase_current_1', 'phase_current_2', 'speed',
            'temp_2_bearing_module', 'torque', 'vibration_1']
FAST = ['phase_current_1', 'phase_current_2', 'vibration_1']
BANDS = [(0, 100), (100, 500), (500, 2000), (2000, 10000), (10000, 32000)]
MAGIC = bytes.fromhex('526172211a0700')
END = bytes.fromhex('c43d7b00400700')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def keyed(*parts):
    return digest('|'.join([str(SEED), *map(str, parts)]).encode())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(data)
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def selection():
    common_background = sorted(REGIMES, key=lambda r: keyed('background-regime', r))[:2]
    records = []
    for label in LABELS:
        ordered = sorted(BEARINGS[label], key=lambda b: keyed('bearing-panel', b))
        for index, bearing in enumerate(ordered):
            role = 'focal' if index < 3 else 'background'
            for regime in (REGIMES if role == 'focal' else common_background):
                repetition = min(range(1, 21), key=lambda r: keyed('record-choice', bearing, regime, r))
                name = f'{regime}_{bearing}_{repetition}.mat'
                records.append({'id': keyed('anonymous-record', bearing, regime, repetition)[:8],
                    'label': label, 'bearing': bearing, 'regime': regime, 'repetition': repetition,
                    'role': role, 'member': f'{bearing}/{name}', 'archive_url': URL + bearing + '.rar'})
    records.sort(key=lambda r: keyed('record-order', r['id']))
    need(len(records) == 48 and len({r['id'] for r in records}) == 48, 'Selection size/ID collision')
    return records


def dictionary():
    fields = []
    raw = 'original recorded scale; physical amplitude unit unspecified'
    for c in CHANNELS:
        fields += [{'name': c + '__mean', 'unit': raw}, {'name': c + '__population_sd', 'unit': raw}]
    for c in FAST:
        fields += [{'name': c + '__rms', 'unit': raw},
                   {'name': c + '__absolute_crest_factor', 'unit': 'dimensionless'},
                   {'name': c + '__pearson_kurtosis', 'unit': 'dimensionless'}]
    fields += [{'name': f'vibration_1__relative_power_{lo}_{hi}_Hz', 'unit': 'dimensionless'} for lo, hi in BANDS]
    need(len(fields) == 28, 'Feature count')
    return fields


def rules():
    return {'schema': 'paderborn-selection-and-representation-v1', 'seed': SEED,
        'classes': LABELS, 'candidate_bearings': BEARINGS, 'records': selection(),
        'selection': 'SHA256(seed|bearing-panel|bearing) per class: first3 focal, last2 background; all4 focal regimes; same2 SHA-ranked background regimes; one SHA-minimum repetition among1..20 per bearing/regime.',
        'source': 'Official Paderborn KAt DataCenter; original 2016 paper Table10 bearing IDs; healthy and accelerated-lifetime damage only.',
        'unit_policy': 'All source Y.Unit fields must be empty. Original amplitude values are retained with explicitly unspecified physical amplitude units; no inferred A or m/s^2 labels.',
        'representation': 'Whole source record, native sample counts, no windowing/resampling/normalization/fitting. Seven channel mean/popSD; three high-rate channel RMS/absolute crest/Pearson kurtosis; five fixed vibration periodogram fractions.',
        'spectrum': 'Subtract vibration mean; unwindowed rFFT, nominal fs64000Hz; double interior one-sided powers, leave DC and even-N Nyquist undoubled; fractions over [0,100),[100,500),[500,2000),[2000,10000),[10000,32000]Hz. No spectral feature/threshold selection by observations.',
        'time_axis': 'Validate finite increasing source X arrays and matched lengths; retain timing diagnostics. Nominal64kHz comes from publisher. HostService X is not treated as an independently calibrated regular hardware clock.',
        'dictionary': dictionary(), 'download_body_limit_bytes': LIMIT,
        'prior_probe_bytes': PROBE_BYTES, 'new_model_calls': 0,
        'interpretation': '48 records from15 physical bearing specimens on one measurement rig; focal36 from9 specimens, background12 from6 disjoint specimens. Four observations per focal bearing are dependent. A separate bearing task cannot establish pump-leakage cross-device generalization.'}


def plan(out):
    path = out / 'selection.json'
    need(not path.exists(), 'Selection already exists; refusing replacement')
    p = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'code_sha256': digest(Path(__file__).read_bytes()), **rules()}
    write(path, p)
    (out / 'selection.sha256').write_text(digest(path.read_bytes()) + '\n')
    print(json.dumps({'selection_sha256': digest(path.read_bytes()), 'records': 48, 'features': 28}))


def validate_plan(out):
    p = load(out / 'selection.json')
    need(digest((out / 'selection.json').read_bytes()) == (out / 'selection.sha256').read_text().strip(), 'Selection hash changed')
    need(p['code_sha256'] == digest(Path(__file__).read_bytes()), 'Preprocessing code changed after selection freeze')
    need({k: p[k] for k in rules()} == rules(), 'Frozen rules changed')
    return p


class Downloader:
    def __init__(self, out):
        self.out, self.lock = out, threading.Lock()
        path = out / 'download_ledger.json'
        self.ledger = load(path) if path.exists() else {'limit_bytes': LIMIT,
            'prior_probe_bytes': PROBE_BYTES, 'reserved_body_bytes': PROBE_BYTES, 'requests': []}

    def get(self, url, start, end, etag):
        size = end - start + 1
        with self.lock:
            need(self.ledger['reserved_body_bytes'] + size <= LIMIT, 'Download body budget exhausted')
            self.ledger['reserved_body_bytes'] += size
            row = {'url': url, 'start': start, 'end': end, 'requested_bytes': size, 'status': 'reserved'}
            self.ledger['requests'].append(row)
            write(self.out / 'download_ledger.json', self.ledger)
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'If-Range': etag,
                                                    'Accept-Encoding': 'identity'})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                need(response.status == 206, 'Range unavailable or archive changed; full-download fallback forbidden')
                cr = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                need(cr is not None and tuple(map(int, cr.groups()[:2])) == (start, end), 'Range offsets changed')
                need(response.headers.get('ETag') == etag, 'Archive ETag changed')
                data = response.read(size + 1)
                need(len(data) == size, 'Truncated/oversized Range body')
            with self.lock:
                row.update(status='complete', body_bytes=len(data), sha256=digest(data))
                write(self.out / 'download_ledger.json', self.ledger)
            return data
        except Exception as exc:
            with self.lock:
                row.update(status='failed', error=type(exc).__name__)
                write(self.out / 'download_ledger.json', self.ledger)
            raise


def parse_header(data, offset=0):
    need(len(data) >= 7, 'Short RAR header')
    crc, kind, flags, length = struct.unpack_from('<HBHH', data)
    need(7 <= length <= len(data), 'Invalid/unsupported RAR header length')
    header = data[:length]
    need(zlib.crc32(header[2:]) & 0xffff == crc, 'RAR header CRC mismatch')
    result = {'offset': offset, 'kind': kind, 'flags': flags, 'head_size': length,
              'packed_size': 0, 'header_hex': header.hex()}
    if kind == 0x74:
        need(not flags & (0x04 | 0x10 | 0x100 | 0x200), 'Encrypted/solid/large/unicode RAR member unsupported')
        packed, unpacked, host, filecrc, stamp, version, method, namesize, attr = struct.unpack_from('<IIBIIBBHI', header, 7)
        name = header[32:32 + namesize].decode('ascii').replace('\\', '/')
        need(not name.startswith('/') and '..' not in Path(name).parts, 'Unsafe RAR member')
        result.update(packed_size=packed, unpacked_size=unpacked, file_crc32=filecrc,
                      name=name, method=method, attributes=attr)
    elif flags & 0x8000:
        result['packed_size'] = struct.unpack_from('<I', header, 7)[0]
    return result


def fetch_bearing(out, records, downloader):
    bearing, url = records[0]['bearing'], records[0]['archive_url']
    dest = out / 'source_members' / bearing
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=40) as r:
        meta = {'url': url, 'size_bytes': int(r.headers['Content-Length']),
                'etag': r.headers['ETag'], 'last_modified': r.headers.get('Last-Modified')}
    need(meta['size_bytes'] > 10_000_000 and meta['etag'], 'Unexpected official archive')
    initial = downloader.get(url, 0, 19, meta['etag'])
    need(initial[:7] == MAGIC and parse_header(initial[7:])['kind'] == 0x73, 'Not expected RAR4 archive')
    need(not parse_header(initial[7:])['flags'] & (0x08 | 0x80), 'Solid/encrypted RAR archive')
    wanted = {r['member']: r for r in records}
    found, offset, scanned = {}, 20, 0
    while offset < meta['size_bytes'] and len(found) < len(wanted):
        b = downloader.get(url, offset, min(offset + 511, meta['size_bytes'] - 1), meta['etag'])
        h = parse_header(b, offset)
        scanned += 1
        need(scanned <= 200, 'Unexpected archive member count')
        if h['kind'] == 0x7b:
            break
        if h.get('name') in wanted:
            name = Path(h['name']).name
            evidence = dest / (name + '.source.json')
            rawpath = dest / name
            if evidence.exists() and rawpath.exists():
                previous = load(evidence)
                need(previous['archive'] == meta and previous['member'] == h, 'Cached source provenance changed')
                need(digest(rawpath.read_bytes()) == previous['mat_sha256'], 'Cached MAT changed')
                found[h['name']] = previous
            else:
                need(0 < h['packed_size'] < 15_000_000 and 0 < h['unpacked_size'] < 30_000_000, 'Unexpected member size')
                start = offset + h['head_size']
                packed = downloader.get(url, start, start + h['packed_size'] - 1, meta['etag'])
                rarpath = dest / (name + '.member.rar')
                rarpath.write_bytes(initial + bytes.fromhex(h['header_hex']) + packed + END)
                with rawpath.open('wb') as f:
                    p = subprocess.run(['/usr/bin/bsdtar', '-xOf', str(rarpath), h['name']], stdout=f, stderr=subprocess.PIPE)
                need(p.returncode == 0, 'RAR member decompression failed: ' + p.stderr.decode(errors='replace')[:200])
                raw = rawpath.read_bytes()
                need(len(raw) == h['unpacked_size'] and zlib.crc32(raw) & 0xffffffff == h['file_crc32'], 'Original member size/CRC mismatch')
                previous = {'archive': meta, 'member': h, 'packed_payload_sha256': digest(packed),
                    'derived_single_member_rar_sha256': digest(rarpath.read_bytes()), 'mat_sha256': digest(raw),
                    'note': 'Derived standalone RAR, not complete archive. Authentic main/file headers and selected packed bytes; original uncompressed CRC32 verified.'}
                write(evidence, previous)
                found[h['name']] = previous
        offset += h['head_size'] + h['packed_size']
    need(set(found) == set(wanted), 'Selected member absent; no substitution allowed: ' + bearing)
    write(dest / 'archive_metadata.json', dict(meta, headers_scanned=scanned, selected_members=sorted(found)))
    return {'bearing': bearing, 'members': len(found)}


def fetch(out, workers=4):
    p = validate_plan(out)
    need(1 <= workers <= 4, 'Download workers must be1..4')
    need(shutil.which('bsdtar') is not None, 'bsdtar is required; no install attempted')
    downloader = Downloader(out)
    grouped = [[r for r in p['records'] if r['bearing'] == b] for b in sorted({r['bearing'] for r in p['records']})]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(lambda rs: fetch_bearing(out, rs, downloader), grouped):
            print(json.dumps(row), flush=True)
    print(json.dumps({'download_reserved_body_bytes': downloader.ledger['reserved_body_bytes']}))


def summarize(channels):
    values = []
    for name in CHANNELS:
        x = np.asarray(channels[name], dtype=np.float64)
        need(x.ndim == 1 and len(x) >= 2 and np.isfinite(x).all(), 'Invalid signal: ' + name)
        values += [float(np.mean(x)), float(np.std(x, ddof=0))]
    for name in FAST:
        x = np.asarray(channels[name], dtype=np.float64)
        rms = float(np.sqrt(np.mean(x * x)))
        centered = x - np.mean(x)
        variance = float(np.mean(centered ** 2))
        need(rms > 0 and variance > 0, 'Undefined high-rate shape statistic: ' + name)
        values += [rms, float(np.max(np.abs(x)) / rms), float(np.mean(centered ** 4) / variance ** 2)]
    vibration = np.asarray(channels['vibration_1'], dtype=np.float64)
    power = np.abs(np.fft.rfft(vibration - np.mean(vibration))) ** 2
    power[1:-1 if len(vibration) % 2 == 0 else None] *= 2
    frequency = np.fft.rfftfreq(len(vibration), 1 / 64000)
    total = float(np.sum(power))
    need(total > 0, 'Undefined spectrum')
    for lo, hi in BANDS:
        mask = (frequency >= lo) & (frequency <= hi if hi == 32000 else frequency < hi)
        values.append(float(np.sum(power[mask]) / total))
    need(len(values) == 28 and np.isfinite(values).all(), 'Feature count/finite validation')
    need(abs(sum(values[-5:]) - 1) < 1e-12, 'Spectral partition does not sum to1')
    return values


def extract(out):
    p = validate_plan(out)
    need(not (out / 'dataset.json').exists(), 'Dataset already exists; refusing replacement')
    X, records, truth, audits = [], [], {}, []
    for row in p['records']:
        path = out / 'source_members' / row['member']
        source = load(path.with_name(path.name + '.source.json'))
        need(digest(path.read_bytes()) == source['mat_sha256'], 'Source MAT changed')
        mat = {k: v for k, v in loadmat(path, simplify_cells=True).items() if not k.startswith('__')}
        need(list(mat) == [path.stem], 'MAT variable does not match selected member')
        data = mat[path.stem]
        signals = {s['Name']: s for s in data['Y']}
        need(set(signals) == set(CHANNELS) and len(data['Y']) == 7, 'Unexpected signal inventory')
        channel_audits = []
        for name in CHANNELS:
            s = signals[name]
            need(np.asarray(s['Unit']).size == 0, 'Unexpected nonempty Unit; review required')
            samples = np.asarray(s['Data'])
            timebase = np.asarray(data['X'][int(s['XIndex']) - 1]['Data'])
            need(samples.ndim == timebase.ndim == 1 and len(samples) == len(timebase), 'Signal/timebase shape mismatch')
            need(np.isfinite(timebase).all() and np.all(np.diff(timebase) > 0), 'Timebase not finite/increasing')
            channel_audits.append({'name': name, 'source_unit': '', 'samples': len(samples), 'raster': s['Raster'],
                'time_start': float(timebase[0]), 'time_end': float(timebase[-1]),
                'median_step': float(np.median(np.diff(timebase))), 'min_step': float(np.min(np.diff(timebase))),
                'max_step': float(np.max(np.diff(timebase)))})
        values = summarize({name: signals[name]['Data'] for name in CHANNELS})
        X.append(values)
        records.append({'id': row['id'], 'values': [f'{v:.6f}' for v in values]})
        truth[row['id']] = row['label']
        audits.append({**row, 'mat_sha256': source['mat_sha256'], 'channels': channel_audits,
            'recorded_datetime': str(data['Description']['General'].get('DateTime', ''))})
    dataset = {'dataset_id': 'paderborn_v1', 'task_text':
        'Classify the bearing state as healthy, outer_race_fault, or inner_race_fault from whole-record summaries. '
        'Each record describes a ball bearing in a controlled drive-system test rig. force is radial bearing force; '
        'phase_current_1 and phase_current_2 are two motor phase current signals; speed is shaft speed; '
        'temp_2_bearing_module is bearing-module temperature; torque is shaft torque; vibration_1 is bearing-housing acceleration. '
        'The source MAT physical-unit fields are empty: amplitude summaries retain the original recorded scale and are not claimed to be calibrated SI values. '
        'Population SD uses ddof0. RMS is the root mean square; absolute crest factor is maximum absolute amplitude divided by RMS; '
        'Pearson kurtosis is the centered fourth moment divided by the squared centered second moment (not excess kurtosis). '
        'Vibration relative powers partition the demeaned unwindowed one-sided periodogram into the listed fixed frequency bands, '
        'using the publisher nominal sampling frequency64000Hz. No per-record bearing identifiers, acquisition times or operating-condition labels are supplied.',
        'allowed_labels': LABELS, 'dictionary': dictionary(), 'panels': [{'panel_id': 'p01',
            'focal_ids': [r['id'] for r in p['records'] if r['role'] == 'focal'],
            'background_ids': [r['id'] for r in p['records'] if r['role'] == 'background'], 'records': records}]}
    need(len(dataset['panels'][0]['focal_ids']) == 36 and len(dataset['panels'][0]['background_ids']) == 12, 'Panel counts')
    np.savez_compressed(out / 'measurement_summary.npz', X=np.asarray(X), record_ids=np.asarray([r['id'] for r in p['records']]),
                        feature_names=np.asarray([d['name'] for d in dictionary()]))
    write(out / 'dataset.json', dataset)
    write(out / 'truth.json', {'labels': truth})
    audit = {'schema': 'paderborn-preprocessing-audit-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_sha256': digest((out / 'selection.json').read_bytes()), 'code_sha256': p['code_sha256'],
        'records': audits, 'n_records': 48, 'n_physical_bearings': 15, 'n_measurement_rigs': 1,
        'focal_records': 36, 'focal_physical_bearings': 9, 'background_records': 12, 'background_physical_bearings': 6,
        'focal_background_bearing_disjoint': True, 'all_source_amplitude_units_empty': True,
        'signal_sampling_rates_nominal_hz': {'Mech_4kHz': 4000, 'HostService': 64000, 'Temp_1Hz': 1},
        'source_body_budget': load(out / 'download_ledger.json'), 'truth_format': 'JSON object with labels mapping: anonymous8hexID -> allowed label',
        'output_sha256': {name: digest((out / name).read_bytes()) for name in ['dataset.json', 'truth.json', 'measurement_summary.npz']},
        'new_model_calls': 0, 'limitations': [rules()['interpretation'], rules()['unit_policy'],
            'Broad summary features do not retain the full diagnostic waveform or validate bearing fault isolation.',
            'Accelerated-lifetime specimen damage is not factory-deployed field failure; specimens share one measurement rig.',
            'Multiple operating regimes from one bearing are dependent; report physical specimens separately from records.']}
    write(out / 'audit.json', audit)
    print(json.dumps({k: audit[k] for k in ['n_records', 'n_physical_bearings', 'n_measurement_rigs', 'focal_records', 'background_records']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'fetch', 'extract'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'plan':
        plan(args.output)
    elif args.command == 'fetch':
        fetch(args.output, args.workers)
    else:
        extract(args.output)


if __name__ == '__main__':
    main()
