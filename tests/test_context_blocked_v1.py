"""Mocked ABBA/BAAB protocol tests: zero actual model calls or network access."""
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import context_blocked_v1 as runner
from src.analysis import score_context_blocked_v1 as scoring


class ContextBlockedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='context-blocked-test-', dir=runner.ROOT / 'tmp')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name) / 'study'
        with patch.object(runner.subprocess, 'check_output', return_value='codex-test'):
            runner.prepare(self.base)
        self.protocol, self.groups, self.blocks = runner.validate(self.base)
        self.calls = []

    def fake_call(self, args, cli, model, name, group, *, mode='complete', tokens=20):
        self.assertEqual(args.retries, 1)
        self.assertEqual(args.timeout, 300)
        self.assertEqual(cli, '/fake/codex')
        self.assertEqual(name, [n for b in self.blocks for n in b][len(self.calls)])
        self.calls.append(name)
        folder = args.output / model
        request = {'model': model, 'reasoning_effort': args.effort,
                   'developer_instructions': runner.INSTRUCTIONS, 'user_message': args.prompt_payloads[name]}
        runner.write_json(folder / f'{name}_request.json', request)
        result = [{'id': s['id'], 'predicted': group['valid_labels'][0]} for s in group['samples']]
        raw = json.dumps([] if mode == 'invalid_response' else result)
        usage = None if mode in ('failed', 'unknown_usage') else {'input_tokens': tokens, 'output_tokens': 10}
        events = [{'type': 'thread.started', 'thread_id': f'mock-{name}'}, {'type': 'turn.started'}]
        if mode == 'failed':
            events.append({'type': 'turn.failed'})
        else:
            events += [{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': raw}},
                       {'type': 'turn.completed', 'usage': usage}]
            (folder / f'{name}_attempt1_response.txt').write_text(raw)
        if mode not in ('failed', 'invalid_response'):
            runner.write_json(folder / f'{name}_results.json', result)
        (folder / f'{name}_attempt1_events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
        (folder / f'{name}_attempt1_stderr.txt').write_text('')
        metadata = runner.common_metadata(self.base, self.protocol, name, group)
        metadata.update(status=mode if mode in ('failed', 'invalid_response') else 'complete',
                        previous_execution_metadata_file=None, updated_at=runner.now(),
                        fingerprint=runner.sha(json.dumps({'model': model, 'effort': args.effort,
                        'payload': args.prompt_payloads[name], 'instructions': runner.INSTRUCTIONS,
                        'protocol': 'codex-grouped-revision-v3'}, sort_keys=True).encode()),
                        attempts=[{'attempt': 1, 'returncode': -1 if mode == 'failed' else 0,
                                   'timed_out': mode == 'failed', 'elapsed_seconds': 0.001,
                                   'events_file': f'{name}_attempt1_events.jsonl',
                                   'response_file': f'{name}_attempt1_response.txt',
                                   'stderr_file': f'{name}_attempt1_stderr.txt',
                                   'usage': usage, 'malformed_event_lines': []}])
        if mode not in ('failed', 'invalid_response'):
            metadata['prediction_sha256'] = runner.digest(folder / f'{name}_results.json')
        runner.write_json(folder / f'{name}_metadata.json', metadata)
        if mode in ('failed', 'invalid_response'):
            raise RuntimeError('synthetic failure')

    def run_mock(self, mode='complete', tokens=20):
        with patch.object(runner.shutil, 'which', return_value='/fake/codex'), \
             patch.object(runner.subprocess, 'check_output', return_value='codex-test'), \
             patch.object(runner, 'run_group', side_effect=lambda *args: self.fake_call(*args, mode=mode, tokens=tokens)):
            return runner.run(self.base)

    def test_prepare_exact_payloads_balanced_schedule_and_no_answer_fields(self):
        self.assertEqual(self.protocol['block_sequences'], ['ABBA', 'ABBA', 'BAAB', 'BAAB'])
        self.assertEqual(len(self.groups), 16)
        self.assertEqual(sum(len(g['samples']) for g in self.groups.values()), 672)
        for name, group in self.groups.items():
            source = self.protocol['source_payloads'][group['condition']]['source_prompt']
            self.assertEqual((self.base / 'inference/prompts' / f'{name}.txt').read_bytes(),
                             (runner.ROOT / source).read_bytes())
            self.assertTrue(all(set(s) == {'id', 'prompt'} for s in group['samples']))
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            runner.prepare(self.base)

    def test_full_serial_run_freeze_score_and_no_reruns(self):
        result = self.run_mock()
        self.assertEqual(result['complete_sessions'], 16)
        self.assertEqual(result['planned_prediction_slots'], 672)
        self.run_mock()
        self.assertEqual(len(self.calls), 16)
        runner.freeze(self.base)
        with patch('sys.stdout', new=io.StringIO()):
            result = scoring.score(self.base)
        self.assertEqual(result['totals']['complete_blocks'], 4)
        self.assertEqual(len(result['paired']), 24)
        self.assertEqual(len(result['case_stability']), 84)
        self.assertTrue(all(r['observed_two_repeat_stable_same'] == 21 for r in result['block_summaries']))
        self.assertTrue(all(r['paired_cases'] == 21 for r in result['paired']))

    def test_semantic_failure_stops_future_dispatch_and_preserves_all_slots(self):
        result = self.run_mock('invalid_response')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['not_attempted_sessions'], 15)
        runner.freeze(self.base)
        with patch('sys.stdout', new=io.StringIO()):
            scored = scoring.score(self.base)
        self.assertEqual(scored['totals']['planned_prediction_slots'], 672)
        self.assertEqual(scored['totals']['valid_prediction_slots'], 0)
        self.assertTrue(all(r['incomplete_block'] == 21 for r in scored['block_summaries']))

    def test_unknown_usage_stops_dispatch(self):
        result = self.run_mock('unknown_usage')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['unknown_usage_sessions'], 1)
        self.assertEqual(result['not_attempted_sessions'], 15)
        runner.freeze(self.base)

    def test_transport_failure_stops_without_rerun(self):
        result = self.run_mock('failed')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['not_attempted_sessions'], 15)
        self.run_mock()
        self.assertEqual(len(self.calls), 1)
        runner.freeze(self.base)

    def test_partial_execution_artifact_prevents_retry(self):
        name = self.blocks[0][0]
        folder = self.base / 'predictions' / self.protocol['model']
        folder.mkdir(parents=True)
        (folder / f'{name}_attempt1_events.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Partial/orphaned'):
            self.run_mock()
        self.assertEqual(self.calls, [])

    def test_token_dispatch_threshold_does_not_drop_attempted_outputs(self):
        result = self.run_mock(tokens=600000)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result['observed_total_tokens'], 1200020)
        self.assertEqual(result['not_attempted_sessions'], 14)
        runner.freeze(self.base)

    def test_insufficient_remaining_wall_time_makes_zero_calls(self):
        runner.exclusive_json(self.base / 'predictions/execution_state.json', {
            'protocol_sha256': runner.digest(self.base / 'protocol.json'), 'stop_reason': None,
            'started_at': (datetime.now(timezone.utc) - timedelta(seconds=1600)).isoformat()})
        result = self.run_mock()
        self.assertEqual(self.calls, [])
        self.assertEqual(result['not_attempted_sessions'], 16)
        runner.freeze(self.base)

    def test_second_worker_or_changed_schedule_rejected(self):
        with self.assertRaisesRegex(ValueError, 'worker count'):
            runner.run(self.base, workers=2)
        schedule = {'blocks': [list(reversed(b)) for b in self.blocks]}
        # ABBA and BAAB are palindromes; swap middle with outer to break the
        # actual condition schedule while keeping all groups exactly once.
        schedule['blocks'][0][0], schedule['blocks'][0][1] = schedule['blocks'][0][1], schedule['blocks'][0][0]
        runner.write_json(self.base / 'schedule.json', schedule)
        self.protocol['schedule_sha256'] = runner.digest(self.base / 'schedule.json')
        runner.write_json(self.base / 'protocol.json', self.protocol)
        (self.base / 'protocol.sha256').write_text(runner.digest(self.base / 'protocol.json'))
        with self.assertRaisesRegex(ValueError, 'schedule changed'):
            runner.validate(self.base)

    def test_scorer_never_reads_answers_before_frozen_output_audit(self):
        touched = []
        original = Path.read_text
        def tracked(path, *args, **kwargs):
            touched.append(str(path))
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', tracked), self.assertRaises(ValueError):
            scoring.score(self.base)
        self.assertFalse(any('/answers/' in p for p in touched))

    def test_missing_block_prediction_is_not_patched_from_another_call(self):
        predictions = {'a1': {'x': 'A'}, 'a2': {'x': 'A'}, 'b1': {'x': 'B'}, 'b2': None}
        result = scoring.classify_id('x', ['a1', 'a2'], ['b1', 'b2'], predictions)
        self.assertEqual(result['category'], 'incomplete_block')
        self.assertEqual(result['b_predictions'], ['B', None])


if __name__ == '__main__':
    unittest.main()
