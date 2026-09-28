"""Arithmetic checks for retrospective reporting; never certify model evidence."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'src/analysis/recheck_claude_descriptive.py'
spec = importlib.util.spec_from_file_location('claude_descriptive_recheck', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DescriptiveTests(unittest.TestCase):
    def test_sd_conventions_distinguish_three_repeat_summary(self):
        stats = m.statistics_pct([0, 50, 100])
        self.assertEqual(stats['mean_pct'], 50)
        self.assertEqual(stats['sample_sd_pp_ddof1'], 50)
        self.assertAlmostEqual(stats['population_sd_pp_ddof0'], (5000 / 3) ** .5)
        with self.assertRaises(ValueError):
            m.statistics_pct([0, 100])

    def test_macro_f1_includes_absent_class_as_zero(self):
        acc, f1 = m.independent_metrics(['A', 'A'], ['A', 'A'], ['A', 'B'])
        self.assertEqual(acc, 1)
        self.assertEqual(f1, .5)

    def test_recount_confusion_and_denominators(self):
        acc, f1 = m.independent_metrics(['A', 'A', 'B', 'B'], ['A', 'B', 'B', 'B'], ['A', 'B'])
        self.assertEqual(acc, .75)
        self.assertAlmostEqual(f1, ((2 / 3) + (4 / 5)) / 2)
        with self.assertRaises(ValueError):
            m.independent_metrics(['A'], [], ['A'])


if __name__ == '__main__':
    unittest.main()
