"""Offline sensor serialization prototype; no inference or diagnostic claims.

Requires acquisition-side channel IDs that are independent of display metadata.
Does not infer a correct mapping, inspect labels or detect internally consistent
upstream value/channel misbinding. Fixed panels reject membership changes.
"""
from decimal import Decimal, InvalidOperation
import hashlib
import json


def _keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError('Unexpected schema (answer labels and extra fields are forbidden)')


def _text(value):
    if not isinstance(value, str) or not value or any(c in value for c in '\n\r\t'):
        raise ValueError('Expected nonempty single-line string')
    return value


def registry_digest(registry):
    if not isinstance(registry, list) or not registry:
        raise ValueError('Empty registry')
    ids = []
    for field in registry:
        _keys(field, ['channel_id', 'descriptor'])
        ids.append(_text(field['channel_id']))
        _text(field['descriptor'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate registry channel')
    return hashlib.sha256(json.dumps(registry, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def canonicalize(sample, registry):
    """Reject inconsistent metadata; reorder intact pairs without changing values."""
    registry_digest(registry)
    _keys(sample, ['id', 'readings'])
    _text(sample['id'])
    if not isinstance(sample['readings'], list):
        raise ValueError('Readings must be a list')
    expected = {f['channel_id']: f['descriptor'] for f in registry}
    observed = {}
    for field in sample['readings']:
        _keys(field, ['channel_id', 'descriptor', 'value'])
        cid = _text(field['channel_id'])
        if cid in observed or cid not in expected:
            raise ValueError('Duplicate or unknown acquisition channel')
        if field['descriptor'] != expected[cid]:
            raise ValueError('Descriptor differs from trusted channel dictionary')
        value = _text(field['value'])
        try:
            valid = Decimal(value).is_finite()
        except InvalidOperation:
            valid = False
        if not valid or value.strip() != value:
            raise ValueError('Invalid numerical string')
        observed[cid] = dict(field)
    if set(observed) != set(expected):
        raise ValueError('Missing acquisition channel')
    return {'id': sample['id'], 'readings': [observed[f['channel_id']] for f in registry]}


def define_panel(ids, batch_size, registry):
    """Explicit new panel version; call before outcomes, never silently update it."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Invalid batch size')
    ids = list(ids)
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing/duplicate IDs')
    ids = sorted(_text(i) for i in ids)
    return {'schema': 'fixed-panel-v1', 'registry_sha256': registry_digest(registry),
            'batches': [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]}


def render_requests(samples, registry, panel):
    """Return deterministic label-free JSON; membership must match the fixed panel."""
    _keys(panel, ['schema', 'registry_sha256', 'batches'])
    if panel['schema'] != 'fixed-panel-v1' or panel['registry_sha256'] != registry_digest(registry):
        raise ValueError('Panel registry/version mismatch')
    if not isinstance(panel['batches'], list) or not panel['batches'] or any(not isinstance(b, list) or not b for b in panel['batches']):
        raise ValueError('Invalid batches')
    ids = [_text(i) for batch in panel['batches'] for i in batch]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate panel ID')
    records = {}
    for sample in samples:
        clean = canonicalize(sample, registry)
        if clean['id'] in records:
            raise ValueError('Duplicate sample ID')
        records[clean['id']] = clean
    if set(records) != set(ids):
        raise ValueError('Panel membership changed; define a separate panel version')
    return [json.dumps([records[i] for i in batch], ensure_ascii=False, separators=(',', ':'))
            for batch in panel['batches']]
