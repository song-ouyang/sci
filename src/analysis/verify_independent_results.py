"""Recompute all new classification and paired counts from case-level exports."""
import csv
import json
from pathlib import Path
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiments.independent_validation import BASE,audit_outputs,digest

def main():
    p,g,_,_=audit_outputs(BASE)
    assert digest(BASE/'data_audit.json')==p['data_audit_sha256']
    s=json.loads((BASE/'analysis/scores.json').read_text())
    rows=list(csv.DictReader((BASE/'analysis/cases.csv').open()))
    assert len(rows)==p['requested_classifications']
    index={(r['model'],r['condition'],int(r['repeat']),r['id']):r for r in rows}
    assert len(index)==len(rows)
    def subset(m,c,r,ids=None):return [x for x in rows if x['model']==m and x['condition']==c and int(x['repeat'])==r and (ids is None or x['id'] in ids)]
    def accuracy(rr):return sum(x['predicted']==x['true_label'] for x in rr)/len(rr)
    for q in s['metrics']:
        rr=subset(q['model'],q['condition'],q['repeat'])
        assert len(rr)==q['n']
        if q['status']!='complete':
            assert all(not x['predicted'] for x in rr) and q['accuracy'] is None
            continue
        assert all(x['predicted'] for x in rr)
        correct=sum(x['predicted']==x['true_label'] for x in rr)
        assert correct==q['correct'] and abs(accuracy(rr)-q['accuracy'])<1e-12
        labels=q['class_order'];cm=[[sum(x['true_label']==a and x['predicted']==b for x in rr) for b in labels] for a in labels]
        assert cm==q['confusion_matrix']
        f=[]
        for i in range(len(labels)):
            den=sum(cm[i])+sum(row[i] for row in cm)
            f.append(2*cm[i][i]/den if den else 0)
        assert abs(sum(f)/len(f)-q['macro_f1'])<1e-12
    for q in s['paired']:
        ids=([(x['late'],x['early']) for x in p['early_late_pairs']] if q['kind']=='same_run_phase' else [(i,i) for i in (p['main_ids'] if q['kind']=='mapping' else p['focal_ids'])])
        pairs=[(index[(q['model'],q['left'],q['repeat'],a)],index[(q['model'],q['right'],q['repeat'],b)]) for a,b in ids]
        available=[(a,b) for a,b in pairs if a['predicted'] and b['predicted']]
        assert len(available)==q['paired_n']
        if not available:assert q['difference'] is None;continue
        flips=sum(a['predicted']!=b['predicted'] for a,b in available)
        diff=sum(int(a['predicted']==a['true_label'])-int(b['predicted']==b['true_label']) for a,b in available)/len(available)
        assert flips==q['flips'] and abs(diff-q['difference'])<1e-12
    for q in s['focal']:
        rr=[x for x in subset(q['model'],q['condition'],q['repeat'],p['focal_ids']) if x['predicted']]
        assert len(rr)==q['n']
        if rr:assert sum(x['predicted']==x['true_label'] for x in rr)==q['correct']
    a=json.loads((BASE/'answers/records.json').read_text());d=json.loads((BASE/'data_audit.json').read_text())
    assert len(a)==242 and len({x['group'] for x in a})==172
    for x in a:
        numeric=[float(line.rsplit(': ',1)[1]) for line in x['prompt'].split('\n\n')[1].splitlines()]
        assert numeric==x['features'], 'Baseline/text numeric mismatch'
    focal=[x for x in a if x['id'] in p['focal_ids']]
    assert Counter(x['label'] for x in focal)==Counter(p['focal_class_counts'])
    tep={int(x['group'].rsplit('-',1)[1]) for x in a if x['group'].startswith('TEP/')}
    assert len(tep)==112 and tep.isdisjoint(set(d['controlled_excluded_run_ids'])|set(d['historical_excluded_run_ids']))
    for q in p['early_late_pairs']:
        aa=next(x for x in a if x['id']==q['early']);bb=next(x for x in a if x['id']==q['late'])
        assert aa['group']==bb['group'] and aa['label']==bb['label']
    result=dict(status='pass',metric_rows=len(s['metrics']),paired_rows=len(s['paired']),focal_rows=len(s['focal']),case_rows=len(rows),new_unique_sources=172,data_audit_sha256=digest(BASE/'data_audit.json'),scores_sha256=digest(BASE/'analysis/scores.json'),human_approval='pending')
    (BASE/'analysis/independent_count_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
