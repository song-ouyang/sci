"""Finish only never-attempted jobs after a model worker stops on semantic failure.

Existing successful or invalid-response outcomes are terminal. This helper never
retries or replaces them and uses the unchanged, hashed inference implementation.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run_revision_v3 import run_group, sha, DEFAULT_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=DEFAULT_MODELS, required=True)
    args_cli = parser.parse_args()
    base = ROOT/'results/revision_v3'
    protocol_path = base/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    input_path = base/'inference/batches.json'
    if sha(input_path.read_bytes()) != protocol['input_sha256']:
        raise ValueError('Frozen input changed')
    groups = json.loads(input_path.read_text())['groups']
    args = SimpleNamespace(output=base/'predictions', prompts_dir=base/'inference/prompts',
                           runner_sha256=sha((ROOT/'src/experiments/run_revision_v3.py').read_bytes()),
                           protocol_sha256=sha(protocol_path.read_bytes()),
                           input_sha256=sha(input_path.read_bytes()), effort=protocol['reasoning_effort'],
                           timeout=900, retries=2, prompt_payloads={})
    cli = shutil.which('codex')
    if not cli:
        raise ValueError('Missing Codex CLI')
    args.cli_version = subprocess.check_output([cli,'--version'],text=True).strip()
    schedule = json.loads((args.output/'session_schedule.json').read_text())[args_cli.model]
    for name in schedule:
        metadata_path = args.output/args_cli.model/f'{name}_metadata.json'
        if metadata_path.exists():
            status = json.loads(metadata_path.read_text())['status']
            if status not in {'complete','invalid_response'}:
                raise ValueError(f'Nonterminal existing job must be investigated: {name}/{status}')
            print(f'KEEP TERMINAL {args_cli.model} {name}: {status}',flush=True)
            continue
        if list((args.output/args_cli.model).glob(f'{name}_attempt*')):
            raise ValueError(f'Existing attempt without terminal metadata; not safe to continue: {name}')
        raw = (args.prompts_dir/f'{name}.txt').read_bytes()
        if sha(raw) != protocol['prompt_sha256'][name]:
            raise ValueError('Prompt changed')
        args.prompt_payloads[name] = raw.decode()
        print(f'CONTINUE NEVER-ATTEMPTED {args_cli.model} {name}',flush=True)
        run_group(args,cli,args_cli.model,name,groups[name])


if __name__ == '__main__':
    main()
