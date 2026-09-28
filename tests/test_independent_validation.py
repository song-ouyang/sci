import unittest
from src.experiments.prepare_independent_validation import transform,MAPS


class IndependentMappingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[[f'sensor_{i} (%)',f'{i+0.1:.4f}'] for i in range(52)]
        self.prompt='Task\n\n'+'\n'.join(': '.join(x) for x in self.rows)

    def parsed(self,c):return [s.rsplit(': ',1) for s in transform(self.prompt,c).split('\n\n')[1].splitlines()]

    def test_mapping_faults_retain_exact_numeric_positions(self):
        for c in MAPS:
            if c!='pair_shuffle':self.assertEqual([x[1] for x in self.parsed(c)],[x[1] for x in self.rows])

    def test_three_distinct_derangements(self):
        maps=[]
        for c in ['random_1','random_2','random_3']:
            names=[x[0] for x in self.parsed(c)];maps.append(tuple(names))
            self.assertTrue(all(x!=y[0] for x,y in zip(names,self.rows)))
            self.assertEqual(set(names),{r[0] for r in self.rows})
        self.assertEqual(len(set(maps)),3)

    def test_correct_pair_reorder_preserves_physical_associations(self):
        self.assertEqual(dict(self.parsed('pair_shuffle')),dict(self.rows))
        self.assertNotEqual(self.parsed('pair_shuffle'),self.rows)

    def test_local_swap_changes_exactly_two_same_unit_descriptors(self):
        changed=self.parsed('same_unit_swap')
        self.assertEqual([i for i,(a,b) in enumerate(zip(changed,self.rows)) if a!=b],[7,11])
        bad=self.prompt.replace('sensor_7 (%)','sensor_7 (kPa)')
        with self.assertRaises(ValueError):transform(bad,'same_unit_swap')


if __name__=='__main__':unittest.main()
