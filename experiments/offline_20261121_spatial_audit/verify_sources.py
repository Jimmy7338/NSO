#!/usr/bin/env python3
"""Bind the local spatial-audit inputs to the completed public experiment commit."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COLD = ROOT.parent / 'same_goal_cold_holdout_20261121'
COMMIT = '4c388a5c06d51e53251a9c901bbefd511d5b2a95'
REPOSITORY = 'Jimmy7338/NSO'
PREFIX = 'experiments/offline_20261121_same_goal_cold_holdout'
GITHUB_BLOBS = {
    'EPISODE_INDEX.json': 'd73d88d1c95e46c3a54281012c55e363feffc9cc',
    'PROTOCOL.json': 'fd6a244070d24625e74302d6ed04d9dba3c410d4',
    'measurement/results/episodes.csv': 'f102e46b8baa01f34db50053a72db8262b55fcff',
    'measurement/results/contrasts.csv': '363b13b21bf138157a91f3d6dd14260135cd0c8b',
    'runner/holdout_freeze.json': '89074c68d7587cb0f11cb1d10baba0263bef6d97',
    'measurement/EVALUATOR_FREEZE.json': '1a8e68fe1a9b176457c27613a4d41564e8a53862',
    'measurement/evaluate.py': '48d872fdf3c6636d2349e4e8de1f1903d2db14f4',
    'measurement/results/checkpoints.csv': 'ced780ab027a293bc2443a50f99a4eafa34d2f55',
    'measurement/results/engineering_gate.json': '5235f65cf96e0a0c43484d43765afdc703ab3e27',
    'measurement/layouts.csv': '7ef105c14e1d609ed73a960ae347dbe58674c3a9',
}


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_sha(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Archived source checksum mismatch: {path}')
    return expected


def git_blob(path):
    payload = path.read_bytes()
    return hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()


def verify(cold):
    checkout = cold / 'publication/payload'
    published_inputs = {}
    for relative, expected in GITHUB_BLOBS.items():
        path = checkout / relative
        if git_blob(path) != expected:
            raise ValueError(f'Published file differs from GitHub commit: {relative}')
        original = cold / relative
        if relative != 'EPISODE_INDEX.json' and git_blob(original) != expected:
            raise ValueError(f'Live workspace file differs from published file: {relative}')
        published_inputs[relative] = {'git_blob_sha1': expected, 'sha256': sha256(path)}

    protocol = read(checkout / 'PROTOCOL.json')
    index = read(checkout / 'EPISODE_INDEX.json')
    freeze = read(checkout / 'measurement/EVALUATOR_FREEZE.json')
    sources = {name: check_sha(Path(name), digest) for name, digest in freeze['inputs_sha256'].items()}
    indexed = {tuple(Path(entry['run_directory']).parts[-2:]): entry for entry in index['bundles']}
    selections = {}
    for layout in protocol['layouts']:
        clocks = ['terminal']
        if layout == 'ordinary_cycle_16':
            clocks += ['paid_256', 'paid_320', 'paid_512']
        for arm in ('fullG', 'fullS'):
            entry = indexed[(layout, arm)]
            run = cold / 'runner/runs' / layout / arm
            if run.resolve() != Path(entry['run_directory']).resolve():
                raise ValueError(f'Archive mount differs from recorded working station path: {run}')
            copied = entry['archived_files_sha256']
            for relative in ('manifest.json', 'steps.json', 'global_choices.json', 'local_choices.json', 'audit.json'):
                path = run / relative
                selections[str(path)] = check_sha(path, copied[relative])
            manifest = read(run / 'manifest.json')
            for clock in clocks:
                for filename in ('checkpoint.json', 'belief.npz', 'native_mesh.ply'):
                    relative = f'{clock}/{filename}'
                    path = run / relative
                    if filename.endswith('.ply'):
                        info = entry['native_artifacts_on_host'][relative]
                        if path.stat().st_size != info['bytes']:
                            raise ValueError(f'Native mesh size has changed: {path}')
                        expected = info['sha256']
                    else:
                        expected = copied[relative]
                    if expected != manifest['files_sha256'][relative]:
                        raise ValueError(f'Run and publication index disagree: {path}')
                    selections[str(path)] = check_sha(path, expected)

    chain = cold / 'runner/runs/ordinary_chain_15'
    cycle = cold / 'runner/runs/ordinary_cycle_16'
    chain15_identity = selections[str(chain / 'fullG/terminal/native_mesh.ply')] == selections[
        str(chain / 'fullS/terminal/native_mesh.ply')]
    cycle16_terminal_equals_512 = {
        arm: all(selections[str(cycle / arm / 'terminal' / name)]
                 == selections[str(cycle / arm / 'paid_512' / name)] for name in ('native_mesh.ply', 'belief.npz'))
        for arm in ('fullG', 'fullS')
    }
    if not chain15_identity or not all(cycle16_terminal_equals_512.values()):
        raise ValueError('Identity controls do not match the closed-loop archive')
    return {'repository': REPOSITORY, 'commit': COMMIT, 'repository_path': PREFIX,
            'published_inputs': published_inputs, 'frozen_evaluator_and_reference_inputs_sha256': sources,
            'audited_primary_run_assets_sha256': selections,
            'chain15_terminal_mesh_bitwise_identical': chain15_identity,
            'cycle16_terminal_same_mesh_and_belief_as_paid_512': cycle16_terminal_equals_512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cold-root', type=Path, default=COLD)
    parser.add_argument('--output', type=Path, default=ROOT / 'source_hashes.json')
    settings = parser.parse_args()
    result = verify(settings.cold_root.resolve())
    settings.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f"Verified {len(result['published_inputs'])} GitHub blobs, "
          f"{len(result['frozen_evaluator_and_reference_inputs_sha256'])} evaluator/source inputs, "
          f"{len(result['audited_primary_run_assets_sha256'])} archived run assets.")


if __name__ == '__main__':
    main()
