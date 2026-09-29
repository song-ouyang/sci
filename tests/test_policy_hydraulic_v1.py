"""Offline invariants for unseen hydraulic policy panels; no model calls."""
import copy
import json
import unittest

import numpy as np

from src.experiments import prepare_policy_hydraulic_v1 as prep


def fixture():
    # Six nuisance settings represent the old 54 blocks; ten remain per stratum.
    rows, blocks, excluded = [], [], set()
    nuisance = [(v, a) for v in [73, 80, 90, 100] for a in [90, 100, 115, 130]]
    for pump in [0, 1, 2]:
        for cooler in [3, 20, 100]:
            for j, (valve, accumulator) in enumerate(nuisance):
                block_id = len(blocks) + 1
                start = len(rows) + 1
                condition = [cooler, valve, pump, accumulator]
                # Two transient cycles, then six stable: later middle is cycle start+5.
                rows.extend([condition + [1]] * 2 + [condition + [0]] * 6)
                blocks.append({"block_id": block_id, "start_cycle_1based": start,
                               "end_cycle_1based_inclusive": len(rows),
                               "condition_values": condition,
                               "n_stable_cycles_flag_0": 6})
                if j < 6:
                    excluded.add(block_id)
    profile = np.asarray(rows)
    values = np.arange(len(rows) * 28, dtype=float).reshape(len(rows), 28) / 101
    return blocks, profile, values, excluded


class PolicyHydraulicPreparationTests(unittest.TestCase):
    def setUp(self):
        self.blocks, self.profile, self.values, self.excluded = fixture()

    def build(self):
        return prep.build_dataset(self.blocks, self.profile, self.values, self.excluded)

    def test_balanced_new_blocks_and_middle_stable_cycle(self):
        dataset, truth, audit = self.build()
        selected = audit["selected_records"]
        self.assertEqual(len(selected), 54)
        self.assertEqual(len({r["block_id"] for r in selected}), 54)
        self.assertFalse({r["block_id"] for r in selected} & self.excluded)
        self.assertEqual(audit["n_unused_common_nuisance_settings"], 10)
        self.assertEqual({r["n"] for r in audit["unused_eligible_blocks_per_stratum"]}, {10})
        for panel in dataset["panels"]:
            for role in ["focal", "background"]:
                chosen = [r for r in selected if r["panel_id"] == panel["panel_id"] and r["role"] == role]
                self.assertEqual({(r["profile"][2], r["profile"][0]) for r in chosen},
                                 {(p, c) for p in [0, 1, 2] for c in [3, 20, 100]})
                self.assertEqual(len({(r["profile"][1], r["profile"][3]) for r in chosen}), 1)
                for row in chosen:
                    self.assertEqual(row["cycle_1based"], row["block_start"] + 5)
                    self.assertEqual(row["profile"][4], 0)
                    self.assertEqual(truth["labels"][row["id"]], prep.LABELS[row["profile"][2]])

    def test_selection_independent_of_measurement_values_and_block_order(self):
        dataset, _, audit = self.build()
        altered = self.values + 999999.1234567
        changed, _, audit2 = prep.build_dataset(list(reversed(self.blocks)), self.profile,
                                               altered, self.excluded)
        self.assertEqual(audit["selected_records"], audit2["selected_records"])
        self.assertEqual([p["focal_ids"] for p in dataset["panels"]],
                         [p["focal_ids"] for p in changed["panels"]])
        self.assertNotEqual(dataset["panels"][0]["records"][0]["values"],
                            changed["panels"][0]["records"][0]["values"])

    def test_physical_units_order_and_six_decimal_source_values(self):
        dataset, _, audit = self.build()
        d = dataset["dictionary"]
        self.assertEqual(len(d), 28)
        expected_units = ["bar"] * 12 + ["W"] * 2 + ["L/min"] * 4 + ["deg C"] * 8 + ["mm/s"] * 2
        self.assertEqual([x["unit"] for x in d], expected_units)
        self.assertEqual(d[0]["name"], "PS1 pressure mean")
        self.assertEqual(d[-1]["name"], "VS1 vibration SD")
        self.assertFalse(any(name in json.dumps(d) for name in ["CE ", "CP ", "SE "]))
        provenance = {r["id"]: r for r in audit["selected_records"]}
        for panel in dataset["panels"]:
            for row in panel["records"]:
                i = provenance[row["id"]]["cycle_1based"] - 1
                self.assertEqual(row["values"], [f"{float(x):.6f}" for x in self.values[i]])

    def test_payload_allowlist_anonymity_and_truth_separation(self):
        dataset, truth, _ = self.build()
        prep.validate_dataset(dataset)
        self.assertEqual(set(truth), {"labels"})
        self.assertEqual(set(truth["labels"]), {r["id"] for p in dataset["panels"] for r in p["records"]})
        for panel in dataset["panels"]:
            self.assertEqual(set(panel), {"panel_id", "focal_ids", "background_ids", "records"})
            for row in panel["records"]:
                self.assertEqual(set(row), {"id", "values"})
                self.assertRegex(row["id"], r"^[0-9a-f]{8}$")
        task = dataset["task_text"].lower()
        for leak in ["balanced", "equal numbers", "three per", "nine", "27", "54"]:
            self.assertNotIn(leak, task)

    def test_record_private_field_rejected(self):
        dataset, _, _ = self.build()
        dataset["panels"][0]["records"][0]["true_label"] = "No_leakage"
        with self.assertRaisesRegex(ValueError, "Private/unknown"):
            prep.validate_dataset(dataset)

    def test_dimension_unit_and_nonfinite_corruption_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            prep.build_dataset(self.blocks, self.profile, self.values[:, :-1], self.excluded)
        values = self.values.copy()
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            prep.build_dataset(self.blocks, self.profile, values, self.excluded)
        dataset, _, _ = self.build()
        dataset["dictionary"][0]["unit"] = "deg C"
        with self.assertRaisesRegex(ValueError, "dictionary"):
            prep.validate_dataset(dataset)

    def test_additional_exclusions_prevent_reuse(self):
        _, _, audit = self.build()
        blocked = audit["selected_records"][0]["block_id"]
        _, _, audit2 = prep.build_dataset(self.blocks, self.profile, self.values,
                                          self.excluded | {blocked})
        self.assertNotIn(blocked, {r["block_id"] for r in audit2["selected_records"]})
        self.assertEqual(audit2["n_unused_common_nuisance_settings"], 9)

    def test_insufficient_unused_settings_fails(self):
        remaining = {b["block_id"] for b in self.blocks if b["condition_values"][1] == 100}
        excluded = {b["block_id"] for b in self.blocks} - remaining
        with self.assertRaisesRegex(ValueError, "Fewer than six"):
            prep.build_dataset(self.blocks, self.profile, self.values, excluded)

    def test_duplicate_blocks_and_profile_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate block"):
            prep.build_dataset(self.blocks + [copy.deepcopy(self.blocks[0])],
                               self.profile, self.values, self.excluded)
        _, _, audit = self.build()
        profile = self.profile.copy()
        profile[audit["selected_records"][0]["block_start"] - 1, 0] = 999
        with self.assertRaisesRegex(ValueError, "Block definition"):
            prep.build_dataset(self.blocks, profile, self.values, self.excluded)

    def test_front_back_preserve_exact_records(self):
        dataset, _, _ = self.build()
        for p in dataset["panels"]:
            records = {r["id"]: r for r in p["records"]}
            front = [records[i] for i in p["focal_ids"] + p["background_ids"]]
            rear = [records[i] for i in p["background_ids"] + p["focal_ids"]]
            self.assertEqual(front[:9], rear[9:])
            self.assertEqual(front[9:], rear[:9])
        # Data preparation has no completion-time selection or predictions.
        self.assertFalse(any("predicted" in r for p in dataset["panels"] for r in p["records"]))


if __name__ == "__main__":
    unittest.main()
