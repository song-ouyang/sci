"""Synthetic numerical-edge tests; no data, network, or prediction access."""
import unittest
import numpy as np
from src.analysis import extract_bearing_baseline_v2 as v2


class NyquistRecoveryTests(unittest.TestCase):
    def test_even_nyquist_rounding_is_in_last_band(self):
        x = np.tile([-1., 1.], 143784)
        channels = {name: x for name in v2.old.CHANNELS}
        self.assertGreater(np.fft.rfftfreq(len(x), 1/64000)[-1], 32000)
        before = v2.summarize(channels, False); after = v2.summarize(channels)
        self.assertLess(sum(before[-5:]), 1e-12)
        self.assertAlmostEqual(after[-1], 1)
        self.assertEqual(before[:27], after[:27])

    def test_ordinary_even_matches_immutable_legacy(self):
        x = np.tile([-1., 1.], 32)
        channels = {name: x for name in v2.old.CHANNELS}
        self.assertEqual(v2.summarize(channels), v2.old.summarize(channels))

    def test_odd_rfft_tail_remains_doubled(self):
        n = 101; x = np.cos(2*np.pi*50*np.arange(n)/n)
        channels = {name: x for name in v2.old.CHANNELS}
        self.assertEqual(v2.summarize(channels), v2.old.summarize(channels))
        self.assertAlmostEqual(v2.summarize(channels)[-1], 1)


if __name__ == '__main__': unittest.main()
