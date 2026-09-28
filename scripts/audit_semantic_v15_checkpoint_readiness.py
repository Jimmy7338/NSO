#!/usr/bin/env python3
"""Prove checkpoint restoration and RGB isolation without future outcomes."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.virtual3d_inspection_v4 import read_inspection_markers_rgb
from nso.cpu_sensor_contract_v10 import digest
from nso.decision_replay_v13 import load_packet
from scripts import collect_semantic_v15_histories as collector
from scripts.collect_semantic_gain_v13_history import sha, write


def verify(folder):
    inventory = json.loads((folder / 'artifact_hashes.json').read_text())
    for name, expected in inventory.items():
        if sha(folder / name) != expected:
            raise ValueError(f'changed evidence: {folder / name}')
    return len(inventory)


class CheckpointReached(Exception):
    pass


def restore_one(output, declaration):
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'changed source: {name}')
    protocol = manifest['protocol']
    history = ROOT / protocol['histories']
    hp = json.loads((history / 'manifest.json').read_text())['protocol']
    seed, step = declaration['structure_seed'], declaration['action_id']
    checkpoints = json.loads((history / f'structure_{seed}/all_decisions.json').read_text())
    expected = next(r for r in checkpoints if r['action_id'] == step)
    receipts = []
    original = collector.DecisionCaptureRuntimeV14

    class RestoreProbe(original):
        def choose_goal(self, scene_idx, proposed_local, bounds):
            result = super().choose_goal(scene_idx, proposed_local, bounds)
            observed = self.decision_snapshots[-1]
            if observed['action_id'] == step:
                if digest(observed) != digest(expected):
                    raise ValueError('full before/after checkpoint replay differs')
                if self.states[0]['pending'] is not None:
                    raise ValueError('probe passed action authorization boundary')
                receipts.append(dict(**declaration, status='passed',
                    decision_ordinal=observed['ordinal'],
                    prefix_paid_actions=self.states[0]['packet'].action_id,
                    state_before_sha256=observed['state_before']['sha256'],
                    state_after_choice_sha256=observed['state_after_choice']['sha256'],
                    candidate_pool_sha256=observed['candidate_sha256'],
                    candidate_count=len(observed['candidates']),
                    pending_action=None, future_actions=0))
                raise CheckpointReached()
            if observed['action_id'] > step:
                raise ValueError('missed checkpoint')
            return result

    collector.DecisionCaptureRuntimeV14 = RestoreProbe
    try:
        try:
            # Replay mode validates every regenerated prefix packet/action and
            # writes only at episode completion, which this probe never reaches.
            collector.run_seed(history, hp, seed, replay=True)
        except CheckpointReached:
            pass
    finally:
        collector.DecisionCaptureRuntimeV14 = original
    if len(receipts) != 1:
        raise ValueError('checkpoint was not restored exactly once')
    write(output / f'restore_{seed}_{step}.json', receipts[0])
    print(json.dumps(receipts[0]), flush=True)


def rgb_isolation(history, pairs):
    rows = []
    for pair in pairs:
        if not pair['matched']:
            continue
        different_pixels = 0
        for step in range(pair['action_id'] + 1):
            frames = [load_packet(history / f'structure_{seed}/packets/{step:04d}.npz').frame
                      for seed in (pair['base_seed'], pair['alternative_seed'])]
            labels = [read_inspection_markers_rgb(f.color_rgb) for f in frames]
            masks = [a != 0 for a in labels]
            np.testing.assert_array_equal(masks[0], masks[1])
            neutral = [f.color_rgb.copy() for f in frames]
            for a, mask in zip(neutral, masks):
                a[mask] = 153
            np.testing.assert_array_equal(neutral[0], neutral[1])
            different_pixels += int(np.count_nonzero(np.any(frames[0].color_rgb != frames[1].color_rgb, axis=2)))
        rows.append(dict(base_seed=pair['base_seed'], alternative_seed=pair['alternative_seed'],
                         action_id=pair['action_id'], frames_checked=pair['action_id'] + 1,
                         marker_masks_equal=True, neutral_rgb_exact=True,
                         changed_marker_pixel_observations=different_pixels))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--restore-index', type=int)
    args = p.parse_args()
    if args.restore_index is not None:
        m = json.loads((args.output / 'manifest.json').read_text())
        restore_one(args.output, m['protocol']['checkpoints'][args.restore_index])
        return
    pp = ROOT / 'configs/virtual3d/semantic_v15_checkpoint_readiness.json'
    protocol = json.loads(pp.read_text())
    history, pairing = (ROOT / protocol[k] for k in ('histories', 'pairing'))
    counts = {str(folder.relative_to(ROOT)): verify(folder) for folder in (history, pairing)}
    hm = json.loads((history / 'manifest.json').read_text())
    if hm['status'] != 'complete':
        raise ValueError('complete histories required')
    frozen = dict(hm['source_sha256'])
    for path in (pp, Path(__file__).resolve()):
        frozen[str(path.relative_to(ROOT))] = sha(path)
    for name, expected in frozen.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'changed source: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    m = dict(status='running', protocol=protocol, source_sha256=frozen,
             created_utc=datetime.now(timezone.utc).isoformat(), input_artifacts_checked=counts)
    write(args.output / 'manifest.json', m)
    with zipfile.ZipFile(args.output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name in frozen:
            z.write(ROOT / name, name)
    try:
        pairs = json.loads((pairing / 'result.json').read_text())['rows']
        rgb = rgb_isolation(history, pairs)
        write(args.output / 'rgb_isolation.json', rgb)
        for index in range(len(protocol['checkpoints'])):
            subprocess.run([sys.executable, str(Path(__file__)), '--output', str(args.output),
                            '--restore-index', str(index)], check=True)
        for name, expected in frozen.items():
            if sha(ROOT / name) != expected:
                raise ValueError(f'changed source: {name}')
        for folder in (history, pairing):
            verify(folder)
        write(args.output / 'summary.json', dict(status='passed',
              separate_process_checkpoints_restored=len(protocol['checkpoints']),
              rgb_pairs_checked=len(rgb), marker_masks_and_neutral_rgb_exact=True,
              original_evidence_unchanged=True, future_actions=0, future_rewards=0,
              training_allowed=False, semantic_efficacy_proven=False, parent_groups=1))
        m['status'] = 'complete'
    except Exception as error:
        m.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'manifest.json', m)
        write(args.output / 'artifact_hashes.json', {str(p.relative_to(args.output)): sha(p)
              for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
