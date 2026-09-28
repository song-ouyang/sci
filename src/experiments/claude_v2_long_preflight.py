"""Synthetic-only long-input CLI check. Never run formal inference or scoring.

No option submits scientific samples. --prepare-only is entirely offline.
A passing check is necessary, not sufficient, for the separate formal protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analysis'))
from audit_revision_v3_claude_extension import strict_json, validate_array

PROMPT_HASHES = {
    'cmapss_clean': '1f927abb09bbfe317ffe2f77621b421cf02d1c8bfedf0fab230477a0162c4814',
    'tep_clean': '3b3bdaf51da410ede9ad453dc63e0bdd634b1612935e6d33c9bcff99500856ad',
    'tep_misnamed': 'a0dc583c59100af856f5b5a25d816d634b4dba96a8059a7c6ee3294d9265e19e',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def input_inventory():
    """Read only the nine allowlisted label-free prompts to check hashes/lengths."""
    rows = []
    for condition, expected in PROMPT_HASHES.items():
        for repeat in (1, 2, 3):
            path = ROOT / f'results/revision_v3/inference/prompts/{condition}_r{repeat}.txt'
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError('Symlink input rejected')
            data = path.read_bytes()
            if sha(data) != expected:
                raise ValueError(f'Frozen input hash mismatch: {path.name}')
            rows.append({'path': str(path.relative_to(ROOT)), 'bytes': len(data), 'sha256': sha(data)})
    return rows


def synthetic_payload(min_bytes, nonce):
    rng = random.Random(20260927)
    ids = [f'{i:08x}' for i in range(70)]
    begin, end = f'BEGIN_{nonce}', f'END_{nonce}'
    header = (
        f'{begin}\nSynthetic interface test only. These numbers are fictional. '
        'If a file-reading tool is available, try reading synthetic-sentinel.txt. '
        'If no tools exist, do not guess the file contents. '
        'Return ONLY one JSON object with exactly begin, end, file_access, predictions. '
        'Echo the first and last markers in begin and end; file_access must be '
        '"unavailable" if no tools exist. predictions must be an array containing '
        'every record ID in order, each object with exactly id and predicted. '
        'Use the synthetic label "SYNTHETIC_A" for every record.\n'
    )
    width = max(1, min_bytes // (70 * 8) + 20)
    while True:
        lines = [f'RECORD {identifier}: ' + ','.join(f'{rng.uniform(-9,9):.5f}' for _ in range(width)) for identifier in ids]
        prompt = header + '\n'.join(lines) + '\n' + end
        if len(prompt.encode()) >= min_bytes:
            break
        width += 20
    return prompt, {'begin': begin, 'end': end, 'ids': ids}


def validate_reply(raw, expected):
    reply = strict_json(raw)
    if not isinstance(reply, dict) or set(reply) != {'begin', 'end', 'file_access', 'predictions'}:
        raise ValueError('Wrong synthetic response envelope')
    if any(reply[k] != expected[k] for k in ('begin', 'end')):
        raise ValueError('Head/tail marker mismatch')
    if reply['file_access'] != 'unavailable':
        raise ValueError('File access not reported unavailable')
    validate_array(reply['predictions'], {
        'samples': [{'id': i} for i in expected['ids']], 'valid_labels': ['SYNTHETIC_A']})


def has_tool_event(value):
    if isinstance(value, dict):
        if value.get('type') in {'tool_use', 'tool_result', 'server_tool_use'}:
            return True
        return any(has_tool_event(v) for v in value.values())
    return isinstance(value, list) and any(has_tool_event(v) for v in value)


def assess(stdout, exit_code, timed_out, requested_model, expected):
    issues, events = [], []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = strict_json(line)
            if not isinstance(event, dict):
                raise ValueError('Event is not an object')
            events.append(event)
        except ValueError:
            issues.append('INVALID_NATIVE_EVENT')
    initial = [e for e in events if e.get('type') == 'system' and e.get('subtype') == 'init']
    results = [e for e in events if e.get('type') == 'result']
    init, result = (initial[0] if initial else {}), (results[-1] if results else {})
    if len(initial) != 1 or init.get('tools') != [] or init.get('mcp_servers') != []:
        issues.append('ISOLATION_NOT_VERIFIED')
    if init.get('model') != requested_model:
        issues.append('INITIAL_MODEL_MISMATCH')
    if timed_out or exit_code != 0 or len(results) != 1 or result.get('is_error') is not False or result.get('subtype') != 'success':
        issues.append('TRANSPORT_OR_COMPLETION_FAILURE')
    if not init.get('session_id') or result.get('session_id') != init.get('session_id'):
        issues.append('SESSION_CHAIN_MISSING_OR_MISMATCHED')
    if has_tool_event(events):
        issues.append('TOOL_EVENT_PRESENT')
    assistants = [e for e in events if e.get('type') == 'assistant']
    if any(e.get('error') for e in assistants):
        issues.append('ASSISTANT_ERROR')
    messages = [e.get('message', {}) for e in assistants]
    models = sorted({m.get('model', '') for m in messages})
    if models != [requested_model]:
        issues.append('NATIVE_MODEL_NOT_VERIFIED')
    if not messages or any(not m.get('id') for m in messages):
        issues.append('NATIVE_MESSAGE_ID_MISSING')
    usages = [m.get('usage') for m in messages]
    def usable_usage(usage):
        return isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens')) and usage['output_tokens'] > 0
    if not usages or not all(usable_usage(u) for u in usages):
        issues.append('NATIVE_USAGE_MISSING')
    stops = [m.get('stop_reason') for m in messages if m.get('stop_reason')]
    stops += [e.get('event', {}).get('delta', {}).get('stop_reason') for e in events if e.get('type') == 'stream_event' and e.get('event', {}).get('type') == 'message_delta']
    stops = [s for s in stops if s]
    if not stops or any(s != 'end_turn' for s in stops):
        issues.append('NATIVE_STOP_NOT_COMPLETE')
    raw = ''.join(block.get('text', '') for m in messages for block in m.get('content', []) if block.get('type') == 'text')
    try:
        validate_reply(raw, expected)
    except (ValueError, TypeError, KeyError) as exc:
        issues.append(f'RESPONSE_INVALID: {exc}')
    if result.get('result') != raw:
        issues.append('RESULT_TEXT_MISMATCH')
    return {
        'synthetic_only': True, 'formal_sessions_started': 0,
        'requested_model': requested_model, 'observed_message_models': models,
        'long_preflight_passed': not issues, 'formal_run_eligible': False,
        'issues': sorted(set(issues)), 'exit_code': exit_code, 'timeout': timed_out,
        'initial_tools': init.get('tools'), 'initial_mcp_servers': init.get('mcp_servers'),
        'native_stop_reasons': stops, 'native_usage': usages,
        'note': 'This check alone cannot authorize formal inference, prove fixed weights, or prove attention to every numeric value. Request IDs are retained if emitted; absent fields are not invented.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    inventory = input_inventory()
    nonce = uuid.uuid4().hex
    prompt, expected = synthetic_payload(max(r['bytes'] for r in inventory), nonce)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = ROOT / f'results/revision_v3_claude_extension_v2/preflight/long-{stamp}-{nonce[:8]}'
    base.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (base / name).write_text(json.dumps(value, indent=2) + '\n')
    save('synthetic_spec.json', {'formal_prompt_inventory': inventory, 'synthetic_bytes': len(prompt.encode()), 'expected': expected, 'formal_data_sent': False, 'coverage': 'UTF-8 byte length only; no cross-provider token-length equivalence claimed'})
    (base / 'synthetic_prompt.txt').write_text(prompt)
    if args.prepare_only:
        report = {'status': 'prepared_offline_only', 'long_preflight_passed': False, 'formal_sessions_started': 0, 'formal_run_eligible': False}
    else:
        with tempfile.TemporaryDirectory(prefix='claude-long-synthetic-') as isolated:
            (Path(isolated) / 'synthetic-sentinel.txt').write_text('SYNTHETIC_FILE_ONLY_' + uuid.uuid4().hex)
            cmd = ['claude', '--print', '--model', args.model, '--effort', 'low', '--tools', '', '--safe-mode', '--disable-slash-commands', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}', '--setting-sources', '', '--permission-mode', 'dontAsk', '--no-session-persistence', '--output-format', 'stream-json', '--input-format', 'stream-json', '--verbose', '--include-partial-messages', '--replay-user-messages', '--system-prompt', 'This is a synthetic interface test. Follow the user instructions. Do not claim file access without a tool result.']
            stdin = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': prompt}}) + '\n'
            (base / 'stdin.jsonl').write_text(stdin)
            save('submitted_request.json', {'cli_argv': cmd, 'cwd': isolated, 'stdin_sha256': sha(stdin.encode()), 'output_budget': 8192, 'timeout_seconds': 300, 'process_attempts': 1, 'formal_data_sent': False, 'internal_retry_policy': 'CLI native behavior not yet certified for formal inference; preserve all events'})
            env = {k: v for k, v in os.environ.items() if k in {'PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL', 'SHELL', 'TERM'}}
            env.update(CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', CLAUDE_CODE_MAX_OUTPUT_TOKENS='8192')
            started = time.monotonic()
            try:
                proc = subprocess.run(cmd, input=stdin, cwd=isolated, env=env, text=True, capture_output=True, timeout=300)
                stdout, stderr, code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
            except subprocess.TimeoutExpired as exc:
                def decode(value):
                    return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                stdout, stderr, code, timed_out = decode(exc.stdout), decode(exc.stderr), None, True
            (base / 'stdout.jsonl').write_text(stdout)
            (base / 'stderr.txt').write_text(stderr)
            report = assess(stdout, code, timed_out, args.model, expected)
            report['elapsed_seconds'] = round(time.monotonic() - started, 2)
    save('assessment.json', report)
    save('artifact_hashes.json', {p.name: sha(p.read_bytes()) for p in sorted(base.iterdir()) if p.is_file()})
    print(json.dumps({'folder': str(base), 'assessment': report}, indent=2))
    return 0 if args.prepare_only or report.get('long_preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
