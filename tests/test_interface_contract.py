import copy
import unittest
from src.experiments.interface_contract import canonicalize, define_panel, render_requests


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = [{'channel_id': 'p1', 'descriptor': 'Pressure inlet (bar)'},
                         {'channel_id': 'p2', 'descriptor': 'Pressure outlet (bar)'}]
        self.sample = {'id': 'a', 'readings': [
            {'channel_id': 'p1', 'descriptor': 'Pressure inlet (bar)', 'value': '1.2300'},
            {'channel_id': 'p2', 'descriptor': 'Pressure outlet (bar)', 'value': '0.9900'}]}

    def test_correct_pair_permutation_retains_values_exactly(self):
        changed = copy.deepcopy(self.sample)
        changed['readings'].reverse()
        self.assertEqual(canonicalize(changed, self.registry), self.sample)

    def test_same_unit_descriptor_swap_is_rejected(self):
        changed = copy.deepcopy(self.sample)
        a, b = changed['readings']
        a['descriptor'], b['descriptor'] = b['descriptor'], a['descriptor']
        with self.assertRaises(ValueError): canonicalize(changed, self.registry)

    def test_hidden_value_misbinding_is_explicitly_not_detected(self):
        changed = copy.deepcopy(self.sample)
        a, b = changed['readings']
        a['value'], b['value'] = b['value'], a['value']
        self.assertEqual(canonicalize(changed, self.registry), changed)
        self.assertNotEqual(changed, self.sample)

    def test_missing_duplicate_unknown_channels_are_rejected(self):
        for mode in ['missing', 'duplicate', 'unknown', 'identity_absent']:
            with self.subTest(mode=mode):
                changed = copy.deepcopy(self.sample)
                if mode == 'missing': changed['readings'].pop()
                if mode == 'duplicate': changed['readings'][1] = changed['readings'][0]
                if mode == 'unknown': changed['readings'][1]['channel_id'] = 'p3'
                if mode == 'identity_absent': del changed['readings'][1]['channel_id']
                with self.assertRaises(ValueError): canonicalize(changed, self.registry)

    def test_label_leakage_fields_are_rejected(self):
        for level in ['sample', 'reading']:
            changed = copy.deepcopy(self.sample)
            target = changed if level == 'sample' else changed['readings'][0]
            target['label'] = 'Normal'
            with self.assertRaises(ValueError): canonicalize(changed, self.registry)

    def test_invalid_numbers_and_prompt_text_are_rejected(self):
        for value in ['NaN', 'Infinity', '1\nlabel: Normal', 'fault_1', 1.23]:
            with self.subTest(value=value):
                changed = copy.deepcopy(self.sample)
                changed['readings'][0]['value'] = value
                with self.assertRaises(ValueError): canonicalize(changed, self.registry)

    def test_panel_order_stable_but_membership_change_rejected(self):
        b = copy.deepcopy(self.sample); b['id'] = 'b'
        panel = define_panel(['b', 'a'], 1, self.registry)
        self.assertEqual(render_requests([b, self.sample], self.registry, panel),
                         render_requests([self.sample, b], self.registry, panel))
        with self.assertRaises(ValueError): render_requests([b], self.registry, panel)
        c = copy.deepcopy(b); c['id'] = 'c'
        with self.assertRaises(ValueError): render_requests([b, self.sample, c], self.registry, panel)

    def test_registry_mutation_is_rejected(self):
        panel = define_panel(['a'], 1, self.registry)
        changed = copy.deepcopy(self.registry)
        changed.reverse()
        with self.assertRaises(ValueError): render_requests([self.sample], changed, panel)

    def test_duplicate_samples_and_panel_ids_rejected(self):
        with self.assertRaises(ValueError): define_panel(['a', 'a'], 1, self.registry)
        panel = define_panel(['a'], 1, self.registry)
        with self.assertRaises(ValueError): render_requests([self.sample, self.sample], self.registry, panel)


if __name__ == '__main__': unittest.main()
