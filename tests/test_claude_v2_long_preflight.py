"""Offline, fictional CLI events only; these fixtures are not experiment evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'src/experiments/claude_v2_long_preflight.py'
spec = importlib.util.spec_from_file_location('claude_long_preflight', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LongPreflightTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'begin': 'BEGIN_TEST', 'end': 'END_TEST', 'ids': ['00000000', '00000001']}
        self.reply = {'begin': 'BEGIN_TEST', 'end': 'END_TEST', 'file_access': 'unavailable', 'predictions': [{'id': i, 'predicted': 'SYNTHETIC_A'} for i in self.expected['ids']]}
        self.raw = json.dumps(self.reply)
        self.events = [
            {'type': 'system', 'subtype': 'init', 'tools': [], 'mcp_servers': [], 'model': 'TEST_MODEL', 'session_id': 'TEST_SESSION'},
            {'type': 'assistant', 'message': {'id': 'TEST_MSG', 'model': 'TEST_MODEL', 'content': [{'type': 'text', 'text': self.raw}], 'usage': {'input_tokens': 9, 'output_tokens': 8}, 'stop_reason': 'end_turn'}},
            {'type': 'result', 'subtype': 'success', 'is_error': False, 'session_id': 'TEST_SESSION', 'result': self.raw},
        ]

    def assess(self, events=None, **kwargs):
        return module.assess('\n'.join(json.dumps(e) for e in (self.events if events is None else events)), kwargs.get('code', 0), kwargs.get('timeout', False), 'TEST_MODEL', self.expected)

    def test_synthetic_size_and_order(self):
        prompt, expected = module.synthetic_payload(223740, 'TEST')
        self.assertGreaterEqual(len(prompt.encode()), 223740)
        self.assertTrue(prompt.startswith(expected['begin']))
        self.assertTrue(prompt.endswith(expected['end']))
        self.assertEqual(len(expected['ids']), 70)
        self.assertEqual(prompt.count('RECORD '), 70)

    def test_valid_fixture_does_not_authorize_formal_run(self):
        report = self.assess()
        self.assertTrue(report['long_preflight_passed'])
        self.assertFalse(report['formal_run_eligible'])
        self.assertEqual(report['formal_sessions_started'], 0)

    def test_subtype_success_does_not_override_error(self):
        self.events[-1]['is_error'] = True
        self.assertFalse(self.assess()['long_preflight_passed'])

    def test_tools_native_model_stop_and_usage_required(self):
        changes = [(0, 'tools', ['Read']), (0, 'mcp_servers', [{'name': 'test'}]), (1, 'model', '<synthetic>'), (1, 'id', None), (1, 'usage', None), (1, 'stop_reason', 'max_tokens')]
        for index, key, value in changes:
            with self.subTest(key=key):
                events = copy.deepcopy(self.events)
                target = events[index]['message'] if index == 1 else events[index]
                target[key] = value
                self.assertFalse(self.assess(events)['long_preflight_passed'])

    def test_tool_use_rejected_even_when_initial_tools_empty(self):
        self.events[1]['message']['content'].append({'type': 'tool_use', 'name': 'Read'})
        self.assertIn('TOOL_EVENT_PRESENT', self.assess()['issues'])

    def test_session_mismatch_timeout_and_invalid_json_rejected(self):
        self.assertFalse(self.assess(code=1)['long_preflight_passed'])
        self.assertFalse(self.assess(timeout=True)['long_preflight_passed'])
        self.events[-1]['session_id'] = 'OTHER_SESSION'
        self.assertFalse(self.assess()['long_preflight_passed'])
        self.assertFalse(module.assess('not-json', 0, False, 'TEST_MODEL', self.expected)['long_preflight_passed'])

    def test_truncated_duplicate_wrong_order_label_and_fields(self):
        variants = []
        for change in ('missing', 'duplicate', 'reorder', 'label', 'extra', 'marker'):
            reply = copy.deepcopy(self.reply)
            if change == 'missing': reply['predictions'].pop()
            if change == 'duplicate': reply['predictions'][1]['id'] = '00000000'
            if change == 'reorder': reply['predictions'].reverse()
            if change == 'label': reply['predictions'][0]['predicted'] = 'synthetic_a'
            if change == 'extra': reply['predictions'][0]['reason'] = 'test'
            if change == 'marker': reply['end'] = 'wrong'
            variants.append(json.dumps(reply))
        variants += [self.raw[:-1], self.raw.replace('"begin":', '"begin": "duplicate", "begin":', 1), 'Here: ' + self.raw]
        for raw in variants:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    module.validate_reply(raw, self.expected)

    def test_stream_stop_accepted_but_result_text_must_match(self):
        self.events[1]['message']['stop_reason'] = None
        self.events.insert(2, {'type': 'stream_event', 'event': {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}}})
        self.assertTrue(self.assess()['long_preflight_passed'])
        self.events[-1]['result'] += ' trailing'
        self.assertFalse(self.assess()['long_preflight_passed'])


if __name__ == '__main__':
    unittest.main()
