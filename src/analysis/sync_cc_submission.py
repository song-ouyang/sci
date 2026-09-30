"""Synchronize the current CC manuscript into QREI; offline, no inference.

The editorial sources are results/industrial_interface_focused_overleaf/*.tex.
This entry point deliberately does not call the older focused-paper generator.
Use --compile to compile this multi-file LaTeX project with cached Tectonic.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import shutil
import subprocess
import hashlib

ROOT = Path(__file__).resolve().parents[2]
CC = ROOT / 'results/industrial_interface_focused_overleaf'
Q = ROOT / 'results/qrei_submission'

def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def fnum(v):
    x = float(v) * 100
    return f'{x:.1f}' if x >= 0 else f'$-{abs(x):.1f}$'

def refresh_statistics():
    p = CC / 'main.tex'
    s = p.read_text()
    abba = rows(ROOT / 'results/statistics_v2/abba_contrast.csv')[0]
    old = next(x for x in s.splitlines() if x.startswith('ABBA/BAAB: first vs last (B) &'))
    new = ('ABBA/BAAB: first vs last (B) & 98.8 / 94.0 & 4.8 ['
           + fnum(abba['acc_diff_ci_low']) + ', ' + fnum(abba['acc_diff_ci_high'])
           + r'] & 0.50 & 0.67 & 0.003 & 0 \\')
    s = s.replace(old, new)
    mapping = {
        'Canonical vs intact-pair reversal': 'Intact-pair reversal',
        'Canonical vs descriptor rotation': 'Descriptor rotation',
        'Canonical vs identical duplicate': 'Identical duplicate (focal)',
        'Focal+A vs identical duplicate': 'Identical duplicate (focal+A)',
        'A: focal first vs last': 'A: front vs back',
        'B: focal first vs last': 'B: front vs back',
        'Focal first: A vs B': 'Front: A vs B',
        'Focal last: A vs B': 'Back: A vs B',
        'Focal only vs focal+A': 'Focal only vs focal+A (front)',
    }
    data = {r['contrast']: r for r in rows(ROOT / 'results/statistics_v2/hydraulic_contrasts.csv')}
    a = s.index(r'\label{tab:hydraulic}'); b = s.index(r'\end{table}', a)
    table = s[a:b]
    for label, key in mapping.items():
        r = data[key]
        line = next(x for x in table.splitlines() if x.startswith(label + ' &'))
        cols = line.split(' & ')
        cols[2] = fnum(r['acc_diff']) + ' [' + fnum(r['acc_diff_ci_low']) + ', ' + fnum(r['acc_diff_ci_high']) + ']'
        table = table.replace(line, ' & '.join(cols))
    s = s[:a] + table + s[b:]
    # Directly regenerate policy rows from corrected chronology, retaining subset rows.
    policy = {(r['study'], r['policy']): r for r in rows(ROOT / 'results/error_structure_v3/decision_policies.csv')}
    study_map = {'TEP context (84)': 'TEP', 'Hydraulic, five-block (72)': 'Hydraulic (5 blocks)',
                 'Hydraulic, new blocks (108)': 'Hydraulic (new blocks)',
                 'Paderborn bearing (144)': 'Paderborn bearing',
                 'Hydraulic, prospective (162)': 'Hydraulic (prospective)'}
    a=s.index(r'\label{tab:policies}');b=s.index(r'\end{table}',a)
    lines=s[a:b].splitlines();study=None
    for i,line in enumerate(lines):
        if ' & ' not in line: continue
        cells=line.split(' & ')
        if cells[0] in study_map: study=study_map[cells[0]]
        elif not line.startswith(' & '): continue
        if study is None or cells[1] not in {'Single','Vote','Vote (tie-first)','Canary'}:continue
        pol={'Single':'single','Vote':'vote','Vote (tie-first)':'vote_tie_first','Canary':'canary'}[cells[1]]
        if study=='Hydraulic (prospective)' and pol=='vote':pol='vote_tie_first';cells[1]='Vote (tie-first)'
        r=policy[(study,pol)]
        err='--' if not r['accepted_error_rate'] else f"{100*float(r['accepted_error_rate']):.1f}"
        suffix=r'\\\addlinespace' if 'addlinespace' in line else r'\\'
        lines[i]=' & '.join([cells[0],cells[1],f"{100*float(r['coverage']):.1f}",err,
                            f"{100*float(r['usable_correct_yield']):.1f}",f"{float(r['calls_per_window']):.2f} {suffix}"])
    s=s[:a]+'\n'.join(lines)+'\n'+s[b:]
    p.write_text(s)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compile',action='store_true');args=parser.parse_args()
    refresh_statistics()
    s=(CC/'main.tex').read_text()
    title=re.search(r'\\title\{([^\n]+)\}',s).group(1)
    abstract=s.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0].strip()
    plain=re.sub(r'\\([%&])',r'\1',abstract).replace('~',' ')
    wc=len(plain.split())
    assert wc<=230, f'Abstract exceeds QREI 230-word limit: {wc}'
    keywords=re.search(r'\\noindent\\textbf\{Keywords:\} (.+)',s).group(1)
    assert len(keywords.split(';'))<=5
    q=s.replace(r'\documentclass[11pt,a4paper]',r'\documentclass[12pt,a4paper]')
    q=q.replace(r'\usepackage[margin=1in]{geometry}',r'\usepackage[margin=2.5cm]{geometry}'+'\n'+r'\usepackage{setspace}\doublespacing'+'\n'+r'\usepackage[switch]{lineno}\linenumbers')
    # QREI's main-text instructions include author names and affiliations.
    # Correspondence is on the separate title page; avoid the title's symbolic
    # footnote font (not present in the frozen local TeX cache).
    q=re.sub(r'\\thanks\{[^}]*\}', '', q).replace('$^{a,}$', '$^{a}$')
    a=q.index(r'\author{');b=q.index(r'\date{}',a)
    q=q[:a]+q[a:b].replace(r'\small',r'\footnotesize')+q[b:]
    # Keep dense evidence tables legible in the double-spaced manuscript.
    q=q.replace(r'\begin{table}[htbp]\centering\small',r'\begin{table}[htbp]\centering\footnotesize\singlespacing')
    (Q/'manuscript.tex').write_text(q)
    shutil.copy2(CC/'supplement.tex',Q/'supporting_information.tex')
    for p in CC.glob('*.tex'):
        if p.name not in {'main.tex','supplement.tex'}:shutil.copy2(p,Q/p.name)
    shutil.copy2(CC/'references.bib',Q/'references.bib')
    shutil.copytree(CC/'figures',Q/'figures',dirs_exist_ok=True)
    tp=(Q/'titlepage.tex').read_text()
    tp=re.sub(r'(\\noindent\{\\Large\\bfseries ).+?(\\par\})',lambda m:m[1]+title+m[2],tp)
    tp=re.sub(r'(\\noindent\\textbf\{Keywords:\} ).+',lambda m:m[1]+keywords,tp)
    nfig=s.count(r'\begin{figure}');ntab=s.count(r'\begin{table}')
    for p in CC.glob('canary_*main.tex'):
        nfig+=p.read_text().count(r'\begin{figure}');ntab+=p.read_text().count(r'\begin{table}')
    tp=re.sub(r'abstract \d+ words; \d+ figures; \d+ tables;',f'abstract {wc} words; {nfig} figures; {ntab} tables;',tp)
    (Q/'titlepage.tex').write_text(tp)
    meta=(Q/'submission_metadata.txt').read_text()
    meta=re.sub(r'Title:\n[^\n]+',lambda m:'Title:\n'+title,meta)
    meta=re.sub(r'Abstract \(\d+ words\):\n.*?\n\nKeywords:\n[^\n]+',lambda m:f'Abstract ({wc} words):\n{plain}\n\nKeywords:\n{keywords}',meta,flags=re.S)
    if 'Revision approval:' not in meta:meta+='\nRevision approval: Updated 2026-09-30; authors must review this corrected version before submission.\n'
    (Q/'submission_metadata.txt').write_text(meta)
    if args.compile:
        exe=shutil.which('tectonic') or '/opt/homebrew/bin/tectonic'
        for folder,name in [(CC,'main'),(CC,'supplement'),(Q,'manuscript'),(Q,'supporting_information'),(Q,'titlepage'),(Q,'cover_letter')]:
            log=subprocess.run([exe,'--only-cached','--keep-logs',name+'.tex'],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (folder/(name+'.build.txt')).write_text(log.stdout)
            print(name,log.returncode,flush=True)
            if log.returncode:raise RuntimeError(log.stdout[-4000:])
        (Q/'upload').mkdir(exist_ok=True)
        for name in ['manuscript','supporting_information','titlepage','cover_letter']:shutil.copy2(Q/(name+'.pdf'),Q/'upload'/(name+'.pdf'))
        for i,name in enumerate(['framework','repeats','forest','timing','hydraulic','error_structure'],1):
            shutil.copy2(CC/'figures/v2'/(name+'.pdf'),Q/'upload/figures'/f'Figure{i}_{name}.pdf')
    approval_path=Q/'author_approval.json'
    approval=json.loads(approval_path.read_text()) if approval_path.exists() else {'status':'pending'}
    record={'editorial_source':str(CC.relative_to(ROOT)),'target':'Quality and Reliability Engineering International',
            'route':'subscription','abstract_words_whitespace':wc,'keywords':keywords.split('; '),
            'compiled_this_run':args.compile,'author_approval_of_revision':approval['status'],
            'author_approval_record':str(approval_path.relative_to(ROOT)) if approval_path.exists() else None,
            'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [CC,Q] for pattern in ['*.tex','*.bib'] for p in folder.glob(pattern)}}
    if args.compile:
        record['pdf_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [CC/'main.pdf',CC/'supplement.pdf',*[Q/'upload'/(n+'.pdf') for n in ['manuscript','supporting_information','titlepage','cover_letter']]]}
    (Q/'sync_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Synchronized CC -> QREI:',wc,'abstract words')

if __name__=='__main__':main()
