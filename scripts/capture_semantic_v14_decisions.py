#!/usr/bin/env python3
"""Extract all live global boundaries during exact physical history replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.components import NSO_Components
from nso.cpu_sensor_contract_v10 import digest
from nso.decision_capture_v14 import DecisionCaptureRuntimeV14
from scripts import collect_semantic_v14_cold_start as driver
from scripts.collect_semantic_gain_v13_history import sha, write


def capture(source, protocol):
    instances = []
    def start(config, world, transform, first):
        args = SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
            use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
            cpu_score_mode=config['score_mode'], cpu_disable_feedback=False,
            cpu_max_candidates=config['candidate_cap'], cpu_coverage_slots=config['coverage_slots'],
            cpu_planner_revision=config['planner_revision'], cpu_measured_novelty_floor=.25,
            run_id='v13-history-smoke')
        components = NSO_Components(args)
        components.initialize('cpu', 1, *world.shape, *world.shape)
        runtime = DecisionCaptureRuntimeV14(components, 1, world.shape)
        runtime.start_sensor_episode(0, config=world.config, transform=transform, packets=[first],
            total_budget=config['total_budget'], return_anchor=(*first.position, first.heading))
        instances.append(runtime)
        return runtime
    original = driver.start
    driver.start = start
    try:
        terminal = driver.execute(source, protocol, replay=True)
    finally:
        driver.start = original
    runtime = instances[0]
    snapshots = runtime.decision_snapshots
    calls = json.loads((source / 'module_calls.json').read_text())
    selections = [c for c in calls if c['method'] == 'select_topo_target']
    if len(snapshots) != len(selections):
        raise ValueError('global plan count differs from frozen module trace')
    for snapshot, selection in zip(snapshots, selections):
        assert snapshot['action_id'] == selection['action_id']
        assert snapshot['candidate_sha256'] == digest(selection['outputs']['candidates'])
        assert snapshot['selected_option'] == selection['outputs']['selected']
    outer = json.loads((source / 'decisions.json').read_text())
    for old in outer:
        matches = [s for s in snapshots if s['action_id'] == old['action_id']]
        assert len(matches) == 1
        current = matches[0]
        assert current['state_before']['sha256'] == old['state_sha256']
        assert current['candidate_sha256'] == old['candidate_sha256']
        assert current['features'] == old['features']
    return snapshots, terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    source, output = args.source, args.output
    sm = json.loads((source / 'manifest.json').read_text())
    if sm['status'] != 'complete':
        raise ValueError('completed source history required')
    for name, expected in sm['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    inventory = json.loads((source / 'artifact_hashes.json').read_text())
    for name, expected in inventory.items():
        assert sha(source / name) == expected, name
    if args.verify:
        m = json.loads((output / 'manifest.json').read_text())
        for name, expected in m['source_sha256'].items():
            assert sha(Path(name)) == expected, name
        rows, terminal = capture(source, sm['history_protocol'])
        assert rows == json.loads((output / 'all_decisions.json').read_text())
        assert terminal == json.loads((output / 'terminal.json').read_text())
        write(output / 'verification.json', dict(status='passed', exact_all_decision_snapshots=len(rows),
            exact_physical_actions=terminal['paid_actions'], future_rewards_read=False,
            same_observer_implementation=True, separate_process=True))
        return
    output.mkdir(parents=True, exist_ok=False)
    names = sorted({str(Path(__file__).relative_to(ROOT)), 'scripts/collect_semantic_v14_cold_start.py',
        'scripts/collect_semantic_gain_v13_history.py', *sm['source_sha256'],
        'nso/decision_capture_v14.py'})
    frozen = {name: sha(Path(name)) for name in names}
    m = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
        schema='v14_all_global_decision_capture/1', source=str(source), source_sha256=frozen,
        input_inventory_sha256=sha(source / 'artifact_hashes.json'), observation_only=True,
        model_trained=False, future_rewards_read=False)
    write(output / 'manifest.json', m)
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in frozen:
            archive.write(name, name)
    try:
        rows, terminal = capture(source, sm['history_protocol'])
        write(output / 'all_decisions.json', rows)
        write(output / 'terminal.json', terminal)
        subprocess.run([sys.executable, str(Path(__file__)), '--source', str(source),
                        '--output', str(output), '--verify'], check=True)
        summary = []
        outer_steps = {r['action_id'] for r in json.loads((source / 'decisions.json').read_text())}
        for r in rows:
            assets = r['descriptors'][0]['observed_assets'] if r['descriptors'] else []
            confident = [a for a in assets if a['semantic_confidence'] > 0 and a['class_vote'] != 0]
            target_candidates = [c for c in r['candidates'] if c.get('asset_index') is not None]
            confident_ids = {a['asset_index'] for a in confident}
            summary.append(dict(action_id=r['action_id'], remaining_budget=r['remaining_budget'],
                internal_recovery_decision=r['action_id'] not in outer_steps,
                candidates=len(r['candidates']), observed_instances=r['observed_instance_count'],
                confident_instances=len(confident),
                observed_vote_signs=sorted({1 if a['class_vote'] > 0 else -1 for a in confident}),
                inspection_candidates=len(target_candidates),
                confident_inspection_candidates=sum(c['asset_index'] in confident_ids for c in target_candidates),
                state_before_sha256=r['state_before']['sha256'], candidate_sha256=r['candidate_sha256']))
        write(output / 'summary.json', dict(status='complete', decisions=len(rows),
            candidate_descriptors=sum(len(r['candidates']) for r in rows), rows=summary,
            full_trajectory_unchanged=True, semantic_efficacy_proven=False,
            information_value_evaluated=False, parent_groups=1))
        for name, expected in frozen.items():
            assert sha(Path(name)) == expected, name
        for name, expected in inventory.items():
            assert sha(source / name) == expected, name
        m['status'] = 'complete'
    except Exception as error:
        m.update(status='failed', error=repr(error))
        raise
    finally:
        write(output / 'manifest.json', m)
        write(output / 'artifact_hashes.json', {str(p.relative_to(output)):sha(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
