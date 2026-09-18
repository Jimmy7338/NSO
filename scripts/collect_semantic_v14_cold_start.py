#!/usr/bin/env python3
"""Cold-start geometry history with V14 observed features and separate replay."""
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

from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from nso.cpu_sensor_contract_v10 import GridTransform, digest
from nso.decision_replay_v13 import decision_state, load_packet, save_packet
from nso.semantic_opportunities_v14 import candidate_capacity, observed_descriptors, route_instance_features
from scripts.collect_semantic_gain_v13_history import packet, sha, start, write


def execute(folder, protocol, replay=False):
    scene_protocol = json.loads(Path(protocol['scene_protocol']).read_text())
    declared = next(c for c in scene_protocol['contexts'] if c['id'] == protocol['context'])
    settings = {**scene_protocol['shared_conditions'],
                **{k: v for k, v in declared.items() if k not in ('id', 'seed')}}
    # The existing world limit remains unchanged; the planner has the smaller
    # declared pilot budget and must reserve a return to its real initial pose.
    world = InspectionWorldV4(InspectionConfigV4(**settings), seed=declared['seed'],
                              semantic_condition=protocol['semantic_condition'])
    config = {**protocol, 'parent': declared['id'], 'candidate_cap': candidate_capacity(0)}
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    first = packet(world, config, None).validate(transform, world.config)
    if replay:
        assert first.sha256() == load_packet(folder / 'packets/0000.npz').sha256()
    else:
        (folder / 'packets').mkdir()
        save_packet(folder / 'packets/0000.npz', first)
    runtime = start(config, world, transform, first)
    backend = runtime.components._cpu_backend
    actions, decisions = [], []
    expected_actions = json.loads((folder / 'actions.json').read_text()) if replay else None
    expected_decisions = json.loads((folder / 'decisions.json').read_text()) if replay else None
    while not runtime.states[0]['closed']:
        state, bs = runtime.states[0], backend.scenes[0]
        capture = not state['active_actions'] and state['phase'] != 'return' and bs['ledger'].remaining_budget > 0
        checkpoint = None
        if capture:
            backend.args.cpu_max_candidates = candidate_capacity(len(bs['assets']), protocol['coverage_slots'])
            checkpoint = dict(action_id=world.step_count, state_sha256=decision_state(runtime)['sha256'],
                              observed_instances=len(bs['assets']), candidate_cap=backend.args.cpu_max_candidates)
        action = runtime.next_local_action(0)
        if checkpoint is not None:
            selection = bs.get('last_selection', {})
            candidates = selection.get('candidates', [])
            if candidates and candidates[0]['states'][0] != [*world.position, world.heading]:
                raise ValueError('stale candidate pool at decision')
            assets, descriptors = observed_descriptors(runtime, candidates)
            features = route_instance_features(state['mapper'], candidates, assets, descriptors)
            checkpoint.update(candidates=candidates, candidate_sha256=digest(candidates),
                descriptors=descriptors, features=features, next_action=action,
                candidate_audit=selection.get('candidate_audit', {}))
            if replay:
                assert digest(checkpoint) == digest(expected_decisions[len(decisions)]), 'decision replay mismatch'
            decisions.append(checkpoint)
            if not replay:
                write(folder / 'decisions.json', decisions)
            print(f'{"REPLAY" if replay else "COLLECT"} decision step={world.step_count} '
                  f'instances={len(assets)} candidates={len(candidates)}', flush=True)
        if action is None:
            break
        frame, collision, done = world.step(action)
        observed = packet(world, config, action, frame, collision, done).validate(transform, world.config)
        row = dict(action_id=world.step_count, action=action, packet_sha256=observed.sha256(),
                   pose=[*world.position, world.heading], collision=bool(collision))
        if replay:
            assert row == expected_actions[len(actions)], 'physical/action replay mismatch'
            assert observed.sha256() == load_packet(folder / f'packets/{world.step_count:04d}.npz').sha256()
        else:
            save_packet(folder / f'packets/{world.step_count:04d}.npz', observed)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=observed)
        actions.append(row)
        if not replay:
            write(folder / 'actions.json', actions)
        if world.step_count % 16 == 0:
            print(f'{"REPLAY" if replay else "COLLECT"} paid={world.step_count}', flush=True)
    terminal = runtime.sensor_episode_summary(0)
    result = dict(paid_actions=len(actions), collisions=world.collisions,
        returned_to_initial_pose=(*world.position, world.heading) == (*first.position, first.heading),
        within_budget=world.step_count <= protocol['total_budget'],
        termination=terminal['termination'], decisions=len(decisions),
        observed_instance_counts=[d['observed_instances'] for d in decisions],
        candidate_counts=[len(d['candidates']) for d in decisions],
        candidates_omitted_by_cap=sum(len(d['candidate_audit'].get('omitted_by_cap', [])) for d in decisions),
        module_names=sorted({c['module'] for c in backend.calls}),
        final_state_sha256=decision_state(runtime)['sha256'],
        model_trained=False, semantic_efficacy_proven=False, external_prefix_actions=0)
    if replay:
        assert len(actions) == len(expected_actions) and len(decisions) == len(expected_decisions)
        assert result == json.loads((folder / 'history_summary.json').read_text()), 'terminal replay mismatch'
    else:
        write(folder / 'history_summary.json', result)
        write(folder / 'terminal.json', terminal)
    if not (result['returned_to_initial_pose'] and result['within_budget'] and
            world.collisions == 0 and not result['termination']['failed']):
        raise ValueError('history safety/return preflight failed; evidence retained')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    folder = args.output
    if args.replay:
        manifest = json.loads((folder / 'manifest.json').read_text())
        for name, expected in manifest['source_sha256'].items():
            if sha(Path(name)) != expected:
                raise ValueError(f'frozen source changed: {name}')
        result = execute(folder, manifest['protocol'], replay=True)
        write(folder / 'replay.json', dict(status='passed', **result))
        return
    folder.mkdir(parents=True, exist_ok=False)
    protocol_path = Path('configs/virtual3d/semantic_v14_cold_start_preflight.json')
    protocol = json.loads(protocol_path.read_text())
    sources = sorted({str(Path(__file__).relative_to(ROOT)), str(protocol_path), protocol['scene_protocol'],
                      'scripts/collect_semantic_gain_v13_history.py',
                      *[str(p) for base in ('nso', 'env', 'utils') for p in Path(base).rglob('*.py')]})
    frozen = {name: sha(Path(name)) for name in sources}
    manifest = dict(status='running', protocol=protocol, source_sha256=frozen,
                    created_utc=datetime.now(timezone.utc).isoformat(), experiment='history_preflight_only')
    with zipfile.ZipFile(folder / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in frozen:
            archive.write(name, name)
    write(folder / 'manifest.json', manifest)
    try:
        execute(folder, protocol)
        for name, expected in frozen.items():
            if sha(Path(name)) != expected:
                raise ValueError(f'source changed: {name}')
        subprocess.run([sys.executable, str(Path(__file__)), '--output', str(folder), '--replay'], check=True)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write(folder / 'manifest.json', manifest)
        write(folder / 'artifact_hashes.json', {str(p.relative_to(folder)): sha(p)
              for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
