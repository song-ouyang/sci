import copy
import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
from datetime import datetime,timedelta,timezone
from contextlib import redirect_stdout
import io
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiments.policy_validation_v1 import validate_dataset,payload,make_schedule
from src.analysis.score_policy_validation_v1 import exact_paired_p,decision,summarize_policy

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.d=json.loads((ROOT/'data/policy_hydraulic_v1/dataset.json').read_text())
    def test_physical_records_remain_identical_after_position_change(self):
        d=self.d;p=d['panels'][0]
        f,fg=payload(d,p,'F');r,rg=payload(d,p,'R')
        ff=json.loads(f.split('\n\n')[-1]);rr=json.loads(r.split('\n\n')[-1])
        self.assertEqual(ff['dictionary'],rr['dictionary'])
        self.assertEqual(ff['records'][9:]+ff['records'][:9],rr['records'])
        self.assertEqual({x['id']:x['values'] for x in ff['records']},{x['id']:x['values'] for x in rr['records']})
        self.assertEqual(set(ff),{'records','valid_labels','dictionary'})
    def test_private_fields_and_extra_records_rejected(self):
        for mutate in [lambda d:d['panels'][0]['records'][0].update(label='No_leakage'),lambda d:d['panels'][0]['records'].append(d['panels'][0]['records'][0]),lambda d:d.update(answers={})]:
            d=copy.deepcopy(self.d);mutate(d)
            with self.assertRaises(ValueError):validate_dataset(d)
    def test_nonfinite_and_nonspecified_precision_rejected(self):
        for val in ['nan','inf','1.0','1e3',1.000000]:
            d=copy.deepcopy(self.d);d['panels'][0]['records'][0]['values'][0]=val
            with self.assertRaises(ValueError):validate_dataset(d)
    def test_randomization_has_complete_pair_and_round_blocks(self):
        s=make_schedule(self.d)
        self.assertEqual(len(s),18);self.assertEqual([x['round'] for x in s],[1]*9+[2]*9)
        self.assertTrue(all(set(x['conditions'])=={'F','R'} for x in s))
        for block in range(6):
            pp=[x for x in s if x['block']==block]
            self.assertEqual([x['repeat'] for x in pp],[1,2,3]);self.assertEqual(len({x['panel_id'] for x in pp}),1)
        self.assertEqual(len({n for x in s for n in x['names']}),36)
    def test_paired_test_matches_explicit_enumeration_with_zero_pairs(self):
        for ds in [(1,1,0),(-2,1,2,0),(0,0,0),(1,-1,2)]:
            exact=sum(abs(sum(d*s for d,s in zip(ds,signs)))>=abs(sum(ds)) for signs in itertools.product([-1,1],repeat=len(ds)))/2**len(ds)
            self.assertEqual(exact_paired_p(ds),exact)
        self.assertEqual(exact_paired_p([1]*6),2/64)
    def test_missing_calls_cannot_become_majority_or_partial_success(self):
        for rule in ['single','agree','majority']:
            self.assertEqual(decision(['A',None,'A'],rule),(None,'required_call_missing'))
        self.assertEqual(decision(['A','B','C'],'majority'),(None,'tie'))
        self.assertEqual(decision(['A','B','A'],'majority'),('A','accepted'))
    def test_gate_correct_yield_cannot_exceed_its_base(self):
        for base,check,truth in itertools.product('ABC',repeat=3):
            out,_=decision([base,check],'agree')
            self.assertLessEqual(int(out==truth),int(base==truth))
    def test_policy_counts_keep_correct_rejection_and_failure_distinct(self):
        rows=[{'action':'accepted_correct','base_correct':True,'output':'A','reason':'accepted'},
              {'action':'rejected','base_correct':True,'output':None,'reason':'disagreement'},
              {'action':'rejected','base_correct':False,'output':None,'reason':'disagreement'},
              {'action':'rejected','base_correct':None,'output':None,'reason':'required_call_missing'}]
        m=summarize_policy(rows)
        self.assertEqual((m['planned'],m['accepted'],m['accepted_correct'],m['rejected_base_correct'],m['rejected_base_error'],m['base_missing']),(4,1,1,1,1,1))
        self.assertEqual(m['usable_correct_fraction'],.25);self.assertEqual(m['coverage'],.25)
        self.assertEqual(m['error_capture'],1);self.assertEqual(m['false_rejection'],.5)
    def test_full_prepared_scoring_with_synthetic_outputs(self):
        from src.experiments.policy_validation_v1 import prepare,validate
        from src.analysis.score_policy_validation_v1 import score
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)/'study'
            with patch('src.experiments.policy_validation_v1.subprocess.check_output',return_value='synthetic-cli-version'):
                prepare(base,ROOT/'data/policy_hydraulic_v1','hydraulic')
            p,g,pairs=validate(base)
            truth=json.loads((base/'answers/truth.json').read_text())['labels']
            rows=[];origin=datetime(2026,1,1,tzinfo=timezone.utc)
            folder=base/'predictions'/p['model'];folder.mkdir(parents=True)
            for pair in pairs:
                for slot,(name,c) in enumerate(zip(pair['names'],pair['conditions'])):
                    predictions=[{'id':x['id'],'predicted':truth[x['id']]} for x in g[name]['samples']]
                    start=origin+timedelta(seconds=pair['index']*180+.1*slot)
                    rows.append({'group':name,'status':'complete','condition':c,'repeat':pair['repeat'],'n':18,
                                 'usage':{'input_tokens':100,'output_tokens':20},'sessions':[name],
                                 'predictions':predictions,'started_at':start.isoformat(),
                                 'completed_at':(start+timedelta(seconds=10)).isoformat(),'elapsed_seconds':10})
                    (folder/f'{name}_dispatch.json').write_text(json.dumps({'monotonic_offset':pair['index']*180+.1*slot,'target_offset':pair['index']*180}))
            (base/'predictions/frozen_manifest.json').write_text('{}')
            with patch('src.analysis.score_policy_validation_v1.audit',return_value=(p,g,pairs,rows,{})),redirect_stdout(io.StringIO()):
                result=score(base)
            self.assertEqual(result['totals']['valid_prediction_slots'],648)
            self.assertEqual(result['context']['two_sided_sharp_null_p'],1)
            self.assertEqual(result['policies']['Same']['planned'],54)
            self.assertEqual(result['policies']['View']['accepted_correct'],54)
            self.assertEqual(result['policies']['Majority_F']['planned_requests'],18)
            self.assertEqual(result['policies']['Same']['error_capture'],None)
if __name__=='__main__':unittest.main()
