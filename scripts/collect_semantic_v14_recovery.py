#!/usr/bin/env python3
"""Reuse the sealed cold-start driver with an explicit recovery runtime factory."""
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
from nso.recovery_runtime_v14 import RecoveryRuntimeV14
from scripts import collect_semantic_v14_cold_start as driver
from scripts.collect_semantic_gain_v13_history import sha, write


def run(folder, history_protocol, replay):
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
        runtime = RecoveryRuntimeV14(components, 1, world.shape)
        runtime.start_sensor_episode(0, config=world.config, transform=transform, packets=[first],
            total_budget=config['total_budget'], return_anchor=(*first.position, first.heading))
        instances.append(runtime)
        return runtime
    original = driver.start
    driver.start = start
    try:
        result = driver.execute(folder, history_protocol, replay=replay)
    finally:
        driver.start = original
    runtime = instances[0]
    recovery = [r for r in runtime.audit if r['event'] == 'v14_route_denied']
    if replay:
        assert recovery == json.loads((folder / 'recovery_events.json').read_text())
        write(folder / 'replay.json', dict(status='passed', recovery_events_exact=True, **result))
    else:
        write(folder / 'recovery_events.json', recovery)
        write(folder / 'runtime_audit.json', runtime.audit)
        write(folder / 'module_calls.json', runtime.components._cpu_backend.calls)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    folder = args.output
    if args.replay:
        m = json.loads((folder / 'manifest.json').read_text())
        for name, expected in m['source_sha256'].items():
            assert sha(Path(name)) == expected, name
        run(folder, m['history_protocol'], True)
        return
    folder.mkdir(parents=True, exist_ok=False)
    protocol_path = Path('configs/virtual3d/semantic_v14_recovery_comparison.json')
    protocol = json.loads(protocol_path.read_text())
    history = json.loads(Path(protocol['history_protocol']).read_text())
    baseline = Path(protocol['baseline'])
    old_manifest = json.loads((baseline / 'manifest.json').read_text())
    assert old_manifest['status'] == 'complete' and old_manifest['protocol'] == history
    for name, expected in old_manifest['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    for name, expected in json.loads((baseline / 'artifact_hashes.json').read_text()).items():
        assert sha(baseline / name) == expected, name
    sources = sorted({str(Path(__file__).relative_to(ROOT)), str(protocol_path), protocol['history_protocol'],
        history['scene_protocol'], 'scripts/collect_semantic_v14_cold_start.py',
        'scripts/collect_semantic_gain_v13_history.py', 'tests/virtual3d/test_recovery_runtime_v14.py',
        *[str(p) for base in ('nso','env','utils') for p in Path(base).rglob('*.py')]})
    frozen = {name: sha(Path(name)) for name in sources}
    manifest = dict(status='running', protocol=protocol, history_protocol=history,
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
        baseline_inventory_sha256=sha(baseline / 'artifact_hashes.json'), training_allowed=False)
    with zipfile.ZipFile(folder / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(name, name)
    write(folder / 'manifest.json', manifest)
    try:
        run(folder, history, False)
        subprocess.run([sys.executable, str(Path(__file__)), '--output', str(folder), '--replay'], check=True)
        old_actions = json.loads((baseline / 'actions.json').read_text())
        new_actions = json.loads((folder / 'actions.json').read_text())
        assert old_actions[:53] == new_actions[:53], 'common physical prefix changed'
        events = json.loads((folder / 'recovery_events.json').read_text())
        assert events and events[0]['action_id'] == 53 and events[0]['recovery_accepted']
        for name, expected in frozen.items():
            assert sha(Path(name)) == expected, name
        manifest.update(status='complete', common_53_action_prefix_exact=True,
                        first_denial_recovered=True)
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write(folder / 'manifest.json', manifest)
        write(folder / 'artifact_hashes.json', {str(p.relative_to(folder)): sha(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
