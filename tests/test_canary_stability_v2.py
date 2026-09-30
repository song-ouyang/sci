"""Check result-blind source selection, independence counts and isolated runner."""
import json
import unittest
import numpy as np
from src.experiments.prepare_canary_stability_v2 import DATA, select, old_records

class StabilitySelectionTests(unittest.TestCase):
    def test_new_cycles_and_blocks_and_spaced_pairs(self):
        blocks = json.loads((DATA/"blocks.json").read_text())["joint_condition_blocks"]
        profile = np.asarray(json.loads((DATA/"profile.json").read_text())["rows"],dtype=int)
        past = old_records()
        records,audit = select(blocks,profile,past)
        self.assertEqual(len(records),27)
        self.assertFalse({r["cycle_1based"] for r in records}&{r["cycle_1based"] for r in past})
        self.assertFalse({r["block_id"] for r in records}&{r["block_id"] for r in past})
        focal = [r for r in records if r["role"] == "focal"]
        canary = [r for r in records if r["role"] == "canary"]
        self.assertEqual((len(focal),len(canary)),(18,9))
        self.assertEqual(len({r["block_id"] for r in focal}),9)
        self.assertEqual(len({r["block_id"] for r in canary}),9)
        for bid in {r["block_id"] for r in focal}:
            cycles = sorted(r["cycle_1based"] for r in focal if r["block_id"] == bid)
            self.assertEqual(len(cycles),2)
            self.assertGreaterEqual(cycles[1]-cycles[0],2)
        self.assertTrue(all(r["profile"][4] == 0 for r in records))
        self.assertEqual(records,select(blocks,profile,list(reversed(past)))[0])

if __name__ == "__main__":
    unittest.main()
