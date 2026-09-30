"""Verify an exported repository in a disposable copy, with no new inference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize(obj):
    if isinstance(obj, dict):
        return {k:normalize(v) for k,v in obj.items() if k not in {'created_utc','generated_at'}}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--interpreter',type=Path,required=True)
    args=parser.parse_args()
    repo=args.repo.resolve();dest=args.destination.resolve()
    if dest == repo or repo in dest.parents:
        raise RuntimeError('Disposable destination must be outside the source repository.')
    manifest=json.loads((repo/'PUBLICATION_MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(repo/name)!=expected:
            raise RuntimeError('Export manifest mismatch: '+name)
    if dest.exists():
        raise RuntimeError('Use a new disposable destination.')
    dest.mkdir(parents=True)
    for name in [*manifest['files'],'PUBLICATION_MANIFEST.json']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe manifest path: '+name)
        target=dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/relative,target)
    science={str(p.relative_to(dest)):p.read_bytes()
             for folder in ['results/statistics_v2','results/error_structure_v3',
                            'results/canary_controls_v2','results/canary_stability_v2/analysis']
             for p in (dest/folder).glob('*') if p.suffix in {'.json','.csv'}}
    tex={str(p.relative_to(dest)):p.read_bytes()
         for p in (dest/'results/industrial_interface_focused_overleaf').glob('*') if p.suffix in {'.tex','.bib'}}
    report={'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'manifest_files_verified':len(manifest['files']),
            'scope':'Internal offline reconstruction from the exact public export; no independent human review.'}
    proc=subprocess.run([str(args.interpreter.absolute()),'src/analysis/reproduce_public_revision.py','--compile'],
                        cwd=dest,env=dict(os.environ,MPLBACKEND='Agg',PYTHONNOUSERSITE='1'),
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (dest/'public_reproduction.log').write_text(proc.stdout)
    report['returncode']=proc.returncode
    mismatches=[]
    for name,old in science.items():
        new=(dest/name).read_bytes()
        same=normalize(json.loads(old))==normalize(json.loads(new)) if name.endswith('.json') else old==new
        if not same:mismatches.append(name)
    text_changes=[n for n,b in tex.items() if (dest/n).read_bytes()!=b]
    immutable_changes=[n for n,h in manifest['immutable_files'].items() if sha(dest/n)!=h]
    report.update({'scientific_files_compared':len(science),'scientific_mismatches':mismatches,
                   'canonical_tex_bib_compared':len(tex),'canonical_tex_bib_mismatches':text_changes,
                   'immutable_files_compared':len(manifest['immutable_files']),'immutable_mismatches':immutable_changes,
                   'complete':proc.returncode==0 and not mismatches and not text_changes and not immutable_changes,
                   'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    (dest/'public_reproduction_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['complete']:raise RuntimeError('Exported-repository reproduction failed.')

if __name__=='__main__':main()
