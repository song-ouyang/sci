"""Development replay of deterministic input checks, not mitigation inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.interface_contract import canonicalize, define_panel, render_requests


def main():
    source = ROOT / 'results/revision_v3/data/datasets.json'
    ds = json.loads(source.read_text())['datasets']['tep']
    registry = [dict(channel_id=i, descriptor=d) for i, d in zip(ds['feature_names'], ds['feature_display_names'])]
    samples = [dict(id=r['id'], readings=[dict(channel_id=f['channel_id'], descriptor=f['descriptor'], value=f'{v:.4f}')
               for f, v in zip(registry, r['features'])]) for r in ds['test']]
    results = []
    for mode in ['clean', 'correct_pair_reversal', 'descriptor_rotation', 'same_unit_swap', 'hidden_numeric_swap']:
        accepted = 0; canonical_equal = 0
        for s in samples:
            changed = copy.deepcopy(s); fields = changed['readings']
            if mode == 'correct_pair_reversal': fields.reverse()
            if mode == 'descriptor_rotation':
                descriptions = [f['descriptor'] for f in fields]
                for i, f in enumerate(fields): f['descriptor'] = descriptions[(i+1) % len(fields)]
            if mode == 'same_unit_swap':
                fields[7]['descriptor'], fields[11]['descriptor'] = fields[11]['descriptor'], fields[7]['descriptor']
            if mode == 'hidden_numeric_swap': fields[7]['value'], fields[11]['value'] = fields[11]['value'], fields[7]['value']
            try:
                result = canonicalize(changed, registry)
            except ValueError:
                continue
            accepted += 1
            canonical_equal += int(result == s)
        results.append(dict(condition=mode, records=len(samples), accepted=accepted,
                            rejected=len(samples)-accepted, byte_value_identity_to_clean=canonical_equal))
    panel = define_panel([s['id'] for s in samples], 7, registry)
    stable = render_requests(samples, registry, panel) == render_requests(list(reversed(samples)), registry, panel)
    out = ROOT / 'results/interface_contract_development'
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'offline_development_check', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'source_records': len(samples), 'new_independent_sources': 0, 'new_model_requests': 0,
              'trusted_identity_assumption': 'Source field IDs survive independently of corrupted display descriptors.',
              'results': results, 'fixed_panel_arrival_order_invariant': stable,
              'diagnostic_accuracy_improvement': 'not_evaluated',
              'limitations': ['Constructed metadata corruptions only; no field deployment evidence.',
                             'Internally consistent value/channel swaps are not detectable.',
                             'Deterministic canonicalization is a software property, not learned robustness.',
                             'Current source labels were available to the analyst; this is not a held-out efficacy test.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    (out / 'panel.json').write_text(json.dumps(panel, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
