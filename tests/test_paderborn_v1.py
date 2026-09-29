"""Offline synthetic tests only; no public records, model, or network calls."""
import unittest
import struct
import zlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from src.analysis.prepare_paderborn_v1 import selection, dictionary, CHANNELS, FAST, summarize, parse_header, Downloader, LIMIT


class PaderbornPreparationTests(unittest.TestCase):
    def test_specimen_panel_and_regime_balance(self):
        rows = selection()
        self.assertEqual(rows, selection())
        self.assertEqual(len({r['id'] for r in rows}), 48)
        f = [r for r in rows if r['role'] == 'focal']
        b = [r for r in rows if r['role'] == 'background']
        self.assertEqual((len(f), len(b)), (36, 12))
        self.assertFalse({r['bearing'] for r in f} & {r['bearing'] for r in b})
        for label in {r['label'] for r in rows}:
            self.assertEqual(len({r['bearing'] for r in f if r['label'] == label}), 3)
            self.assertEqual(len({r['bearing'] for r in b if r['label'] == label}), 2)
        self.assertTrue(all(len(r['id']) == 8 and 1 <= r['repetition'] <= 20 for r in rows))
        self.assertEqual(len(dictionary()), 28)

    def test_population_and_shape_statistics(self):
        x = np.array([-1., 1.] * 32)
        values = summarize({name: x for name in CHANNELS})
        np.testing.assert_allclose(values[:14], [0., 1.] * 7)
        np.testing.assert_allclose(values[14:23], [1., 1., 1.] * 3)
        np.testing.assert_allclose(values[-5:], [0., 0., 0., 0., 1.], atol=1e-14)

    def test_fixed_band_power_and_scale_invariance(self):
        t = np.arange(64000) / 64000
        x = np.sin(2 * np.pi * 200 * t)
        values = summarize({name: x for name in CHANNELS})
        scaled = summarize({name: 7 * x for name in CHANNELS})
        np.testing.assert_allclose(values[-5:], [0., 1., 0., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(values[-5:], scaled[-5:], atol=1e-12)
        np.testing.assert_allclose([values[15], values[16]], [np.sqrt(2), 1.5], atol=1e-12)

    def test_bad_highrate_or_nonfinite_rejected(self):
        arrays = {name: np.array([-1., 1.]) for name in CHANNELS}
        arrays['vibration_1'] = np.zeros(2)
        with self.assertRaises(ValueError):
            summarize(arrays)
        arrays['vibration_1'] = np.array([np.nan, 1.])
        with self.assertRaises(ValueError):
            summarize(arrays)

    def test_rar_crc_and_solid_guard(self):
        raw = bytes([0x73]) + struct.pack('<HH', 0, 13) + bytes(6)
        header = struct.pack('<H', zlib.crc32(raw) & 0xffff) + raw
        self.assertEqual(parse_header(header)['kind'], 0x73)
        with self.assertRaises(ValueError):
            parse_header(header[:-1] + b'x')
        name = b'K001\\sample.mat'
        body = bytes([0x74]) + struct.pack('<HH', 0x8000, 32 + len(name))
        body += struct.pack('<IIBIIBBHI', 100, 200, 2, 1234, 0, 29, 0x33, len(name), 0x20) + name
        member = struct.pack('<H', zlib.crc32(body) & 0xffff) + body
        self.assertEqual(parse_header(member)['name'], 'K001/sample.mat')
        self.assertEqual(parse_header(member)['packed_size'], 100)
        solid_body = body[:1] + struct.pack('<H', 0x8010) + body[3:]
        solid = struct.pack('<H', zlib.crc32(solid_body) & 0xffff) + solid_body
        with self.assertRaises(ValueError):
            parse_header(solid)

    def test_range_full_response_rejected_without_read_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            downloader = Downloader(Path(folder))
            session, response = MagicMock(), MagicMock()
            response.status_code = 200
            session.get.return_value.__enter__.return_value = response
            downloader.local.session = session
            with self.assertRaises(ValueError):
                downloader.get('https://example.test/K001.rar', 0, 2, 'tag')
            response.raw.read.assert_not_called()
            self.assertEqual(session.get.call_count, 1)

    def test_verified_header_cache_and_budget_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            downloader = Downloader(Path(folder))
            session, response = MagicMock(), MagicMock()
            response.status_code = 206
            response.headers = {'ETag': 'tag', 'Content-Range': 'bytes 0-2/1000'}
            response.raw.read.return_value = b'abc'
            session.get.return_value.__enter__.return_value = response
            downloader.local.session = session
            self.assertEqual(downloader.get('https://example.test/K001.rar', 0, 2, 'tag'), b'abc')
            budget = downloader.ledger['reserved_body_bytes']
            self.assertEqual(downloader.get('https://example.test/K001.rar', 0, 2, 'tag'), b'abc')
            self.assertEqual(downloader.ledger['reserved_body_bytes'], budget)
            self.assertEqual(session.get.call_count, 1)
            downloader.ledger['reserved_body_bytes'] = LIMIT
            with self.assertRaises(ValueError):
                downloader.get('https://example.test/K001.rar', 3, 5, 'tag')
            self.assertEqual(session.get.call_count, 1)


if __name__ == '__main__':
    unittest.main()
