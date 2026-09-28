"""Scientific invariants for the post-v3 experiment; no model calls."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiments.mechanism_extension import transform,split_prompt,make_groups,CONDITIONS

class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups=json.loads((ROOT/'results/revision_v3/inference/batches.json').read_text())['groups']
        cls.clean=cls.groups['tep_clean_r1']

    def test_rotation_exactly_matches_original_frozen_intervention(self):
        for a,b in zip(self.clean['samples'],self.groups['tep_misnamed_r1']['samples']):
            self.assertEqual(transform(a['prompt'],'rotation'),b['prompt'])

    def test_joint_reordering_preserves_every_semantic_pair(self):
        for s in self.clean['samples']:
            _,a=split_prompt(s['prompt']);_,b=split_prompt(transform(s['prompt'],'pair_shuffle'))
            self.assertCountEqual(a,b);self.assertNotEqual(a,b)

    def test_wrong_mapping_preserves_numeric_strings_and_descriptor_inventory(self):
        for s in self.clean['samples']:
            intro,a=split_prompt(s['prompt'])
            for c in ('rotation','random_mapping'):
                intro_b,b=split_prompt(transform(s['prompt'],c))
                self.assertEqual(intro,intro_b)
                self.assertEqual([x[1] for x in a],[x[1] for x in b])
                self.assertCountEqual([x[0] for x in a],[x[0] for x in b])
                self.assertTrue(all(x[0]!=y[0] for x,y in zip(a,b)))
            self.assertNotEqual(transform(s['prompt'],'random_mapping'),transform(s['prompt'],'rotation'))

    def test_anonymous_removes_all_descriptors_and_retains_values(self):
        for s in self.clean['samples']:
            _,a=split_prompt(s['prompt']);_,b=split_prompt(transform(s['prompt'],'anonymous'))
            self.assertEqual([x[1] for x in a],[x[1] for x in b])
            self.assertEqual([x[0] for x in b],[f'field_{i+1:02d}' for i in range(52)])

    def test_context_comparisons_share_exact_records(self):
        gs,ids=make_groups(self.clean)
        self.assertEqual(len(gs),42);self.assertEqual(len(set(ids)),7)
        byid={s['id']:s for s in self.clean['samples']}
        for r in (1,2,3):
            subset=gs[f'tep_subset_batch_r{r}']['samples']
            self.assertEqual(subset,[byid[i] for i in ids])
            singles=[gs[f'tep_single_{i+1:02d}_r{r}']['samples'][0] for i in range(7)]
            self.assertEqual(singles,subset)
            sh=gs[f'tep_sample_shuffle_r{r}']['samples']
            self.assertCountEqual(sh,self.clean['samples']);self.assertNotEqual(sh,self.clean['samples'])
        for name,g in gs.items():
            if g['repeat']>1:
                self.assertEqual(g['samples'],gs[name[:-1]+'1']['samples'])

    def test_generation_is_deterministic_and_does_not_mutate_source(self):
        before=copy.deepcopy(self.clean)
        self.assertEqual(make_groups(self.clean),make_groups(self.clean))
        self.assertEqual(before,self.clean)

    def test_no_answers_in_extension(self):
        gs,_=make_groups(self.clean)
        for g in gs.values():
            self.assertEqual(set(g),{'dataset','condition','repeat','n_samples','valid_labels','samples'})
            for s in g['samples']:self.assertEqual(set(s),{'id','prompt'})

    def test_reject_malformed_sensor_block(self):
        with self.assertRaises(ValueError):transform('task\n\nsensor: 1.0','clean')

if __name__=='__main__':unittest.main()
