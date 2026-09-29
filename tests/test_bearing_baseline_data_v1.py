"""Synthetic, offline invariants; no raw-data, network, or model access."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.analysis import prepare_bearing_baseline_v1 as prep


class BaselineDataTests(unittest.TestCase):
    def test_group_partition_and_nested_five_record_selection(self):
        old = prep.original.selection()
        rows = prep.selected_records(old)
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(r['reuse_original'] for r in rows), 12)
        self.assertEqual(len({r['bearing'] for r in rows}), 6)
        self.assertFalse({r['bearing'] for r in rows} & {r['bearing'] for r in old if r['role'] == 'focal'})
        for bearing in {r['bearing'] for r in rows}:
            for regime in prep.original.REGIMES:
                selected = {r['repetition'] for r in rows if r['bearing'] == bearing and r['regime'] == regime}
                expected = sorted(range(1, 21), key=lambda r: prep.original.keyed('record-choice', bearing, regime, r))[:5]
                self.assertEqual(selected, set(expected))

    def test_original_overlap_rejected(self):
        rows = prep.original.selection()
        background = next(r for r in rows if r['role'] == 'background')
        next(r for r in rows if r['role'] == 'focal')['bearing'] = background['bearing']
        with self.assertRaises(ValueError): prep.selected_records(rows)

    def test_zero_new_budget_and_resume_no_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.assertEqual(prep.Downloader(out).ledger['reserved_body_bytes'], 0)
            self.assertEqual(prep.Downloader(out).ledger['prior_probe_bytes'], 0)
            self.assertEqual(prep.original.PROBE_BYTES, 2427525)
            ledger = prep.load(out / 'download_ledger.json'); ledger['reserved_body_bytes'] = 1
            prep.write(out / 'download_ledger.json', ledger)
            with self.assertRaises(ValueError): prep.Downloader(out)

    @staticmethod
    def measurement():
        x = np.array([-1., 1.] * 32)
        return {'X': [{'Data': np.arange(len(x)) / 64000}], 'Y': [
            {'Name': n, 'Unit': np.array([]), 'Data': x.copy(), 'XIndex': 1, 'Raster': 'synthetic'}
            for n in prep.original.CHANNELS]}

    def test_synthetic_same28_six_decimal_features(self):
        values, channels = prep.summarize_measurement(self.measurement())
        self.assertEqual(len(values), 28)
        self.assertEqual(values[:14], ['0.000000', '1.000000'] * 7)
        self.assertEqual(values[14:23], ['1.000000'] * 9)
        self.assertEqual(len(channels), 7)
        self.assertEqual(values[-1], '1.000000')

    def test_unit_shape_and_time_rejections(self):
        for change in ('unit', 'shape', 'time', 'nan'):
            data = self.measurement()
            if change == 'unit': data['Y'][0]['Unit'] = 'N'
            if change == 'shape': data['Y'][0]['Data'] = np.arange(3)
            if change == 'time': data['X'][0]['Data'][2] = 0
            if change == 'nan': data['Y'][0]['Data'][0] = np.nan
            with self.assertRaises(ValueError): prep.summarize_measurement(data)

    def test_test_input_string_and_order_preservation(self):
        old = prep.original.selection(); ids = [r['id'] for r in old if r['role'] == 'focal']
        rows = [{'id': r['id'], 'values': [f'{i}.000001'] * 28} for i, r in enumerate(old)]
        dataset = {'panels': [{'focal_ids': ids, 'records': rows}]}
        truth = {'labels': {r['id']: r['label'] for r in old}}
        test = prep.fixed_test_rows({'records': old}, dataset, truth)
        lookup = {r['id']: r['values'] for r in rows}
        self.assertEqual([r['id'] for r in test], ids)
        self.assertTrue(all(r['values'] == lookup[r['id']] for r in test))
        changed = copy.deepcopy(truth); changed['labels'][ids[0]] = 'invalid'
        with self.assertRaises(ValueError): prep.fixed_test_rows({'records': old}, dataset, changed)


if __name__ == '__main__': unittest.main()
