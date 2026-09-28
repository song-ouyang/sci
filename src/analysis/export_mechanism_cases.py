"""Export auditable extension cases and independently cross-check aggregate scores."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.mechanism_extension import BASE,SOURCE,MODELS,CONDITIONS,audit_outputs,digest

def main():
    p,groups,records,_=audit_outputs(BASE)
    report=json.loads((BASE/'analysis/scores.json').read_text())
    source=json.loads((SOURCE/'data/datasets.json').read_text())['datasets']['tep']['test']
    answers={s['id']:s for s in source}
    lookup={};cases=[]
    for r in records:
        predictions={x['id']:x['predicted'] for x in (r['predictions'] or [])}
        lookup[(r['model'],r['condition'],r['repeat'])]=predictions
        for s in groups[r['group']]['samples']:
            answer=answers[s['id']]
            cases.append({'model':r['model'],'condition':r['condition'],'repeat':r['repeat'],'request':r['group'],
                          'id':s['id'],'source_group':answer['group'],'true_label':answer['label'],
                          'predicted':predictions.get(s['id']),'available':s['id'] in predictions,
                          'correct':None if s['id'] not in predictions else predictions[s['id']]==answer['label'],
                          'response_status':r['status']})
    paired=[]
    for agg in report['paired_contrasts']:
        m,r,a,b=agg['model'],agg['repeat'],agg['left'],agg['right']
        ids=p['context_subset_ids'] if b in {'subset_batch','single_panel'} else [s['id'] for s in groups['tep_clean_r1']['samples']]
        left=lookup[(m,a,r)]
        right=({i:v for j in range(1,8) for i,v in lookup[(m,f'single_{j:02d}',r)].items()}
               if b=='single_panel' else lookup[(m,b,r)])
        valid=[]
        for i in ids:
            label=answers[i]['label'];available=i in left and i in right
            row={'model':m,'repeat':r,'left_condition':a,'right_condition':b,'id':i,'source_group':answers[i]['group'],
                 'true_label':label,'left_predicted':left.get(i),'right_predicted':right.get(i),'pair_available':available,
                 'prediction_flipped':None if not available else left[i]!=right[i],
                 'correctness_difference':None if not available else int(left[i]==label)-int(right[i]==label)}
            paired.append(row)
            if available:valid.append(row)
        if len(valid)!=agg['paired_n']:raise ValueError('Available paired denominator differs')
        if valid:
            difference=sum(x['correctness_difference'] for x in valid)/len(valid)
            flips=sum(x['prediction_flipped'] for x in valid)
            if abs(difference-agg['accuracy_difference'])>1e-12 or flips!=agg['flips']:
                raise ValueError('Independent paired recomputation differs')
    for agg in report['metrics']:
        rows=[x for x in cases if x['model']==agg['model'] and x['request']==agg['group']]
        valid=[x for x in rows if x['available']]
        if valid:
            correct=sum(x['correct'] for x in valid)
            if correct!=agg['correct'] or abs(correct/len(valid)-agg['accuracy'])>1e-12:
                raise ValueError('Independent accuracy recomputation differs')
        elif agg['accuracy'] is not None:raise ValueError('Unavailable request scored')
    folder=BASE/'analysis/cases';folder.mkdir(exist_ok=True,parents=True)
    for name,rows in [('predictions.csv',cases),('paired_transitions.csv',paired)]:
        with (folder/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={'protocol_sha256':digest(BASE/'protocol.json'),'scores_sha256':digest(BASE/'analysis/scores.json'),
              'frozen_manifest_sha256':digest(BASE/'predictions/frozen_manifest.json'),'answers_sha256':digest(SOURCE/'data/datasets.json'),
              'exporter_sha256':digest(Path(__file__)), 'independent_accuracy_and_pair_checks':'passed',
              'requested_rows':len(cases),'paired_rows':len(paired),'unique_tep_source_groups':len({r['source_group'] for r in cases}),
              'files':{f.name:digest(f) for f in folder.glob('*.csv')}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Exported and independently verified',len(cases),'prediction rows and',len(paired),'paired rows')

if __name__=='__main__':main()
