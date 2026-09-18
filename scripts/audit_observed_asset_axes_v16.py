#!/usr/bin/env python3
"""Audit face-axis correction on every recorded V15 decision asset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from nso.observed_asset_axes_v16 import reorient_observed_asset
from scripts.audit_semantic_v15_feedback_observability import sha, write, require


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    h = json.loads((a.source / 'manifest.json').read_text())
    require(h['status'] == 'complete', 'complete histories required')
    for n, expected in json.loads((a.source / 'artifact_hashes.json').read_text()).items():
        require(sha(a.source / n) == expected, 'history artifact changed')
    a.output.mkdir(parents=True, exist_ok=False)
    names = ['nso/observed_asset_axes_v16.py', 'tests/virtual3d/test_observed_asset_axes_v16.py',
        'nso/competition_candidates_v8_1.py', 'nso/response_features_v7.py',
        'scripts/audit_semantic_v15_feedback_observability.py', str(Path(__file__).relative_to(ROOT))]
    frozen = {n: sha(ROOT / n) for n in names}
    with zipfile.ZipFile(a.output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(ROOT / n, n)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.virtual3d.test_observed_asset_axes_v16', '-v'],
                           text=True, capture_output=True)
    (a.output / 'tests.txt').write_text(tests.stdout + tests.stderr)
    require(tests.returncode == 0, 'tests failed')
    rows = []; decisions = 0
    for seed in h['protocol']['structure_seeds']:
        f = a.source / f'structure_{seed}'
        calls = json.loads((f / 'module_calls.json').read_text())
        for d in json.loads((f / 'all_decisions.json').read_text()):
            decisions += 1
            call = next(c for c in reversed(calls) if c['method'] == 'update_semantic' and c['action_id'] == d['action_id'])
            assets = {x['group']: x for x in call['outputs']['measured_assets']}
            for original in d['candidate_audit']['measured_assets']:
                live = assets[original['group']]
                for field, value in original.items():
                    require(value == live[field], 'module/candidate asset history differs: ' + field)
                corrected = reorient_observed_asset(live)
                n = np.asarray(live['normal_out_xy']); n /= np.linalg.norm(n)
                angle = lambda front: float(np.degrees(np.arccos(np.clip(np.asarray(front) @ n, -1., 1.))))
                require(corrected['class_vote'] == live['class_vote'] and
                        corrected['marked_points'] == live['marked_points'], 'semantic input changed')
                rows.append(dict(structure_seed=seed, action_id=d['action_id'], decision_ordinal=d['ordinal'],
                    observed_group=original['group'], normal_fallback=live['normal_fallback'],
                    normal_sign_depth_support=live['normal_sign_depth_support'],
                    normal_out_xy=n.tolist(), old_front=original['front_axis'],
                    corrected_front=corrected['front_axis'].tolist(),
                    old_angle_degrees=angle(original['front_axis']),
                    corrected_angle_degrees=angle(corrected['front_axis']),
                    old_width=original['measured_width_m'], old_depth=original['measured_depth_m'],
                    corrected_width=corrected['measured_width_m'], corrected_depth=corrected['measured_depth_m']))
    for n, expected in frozen.items(): require(sha(ROOT / n) == expected, 'audit source changed')
    write(a.output / 'asset_axes.json', rows)
    result = dict(status='passed', tests_passed=4, decisions=decisions, asset_state_records=len(rows),
        old_angle_above_60_degrees=sum(r['old_angle_degrees'] > 60 for r in rows),
        corrected_angle_above_60_degrees=sum(r['corrected_angle_degrees'] > 60 for r in rows),
        corrected_max_angle_degrees=max(r['corrected_angle_degrees'] for r in rows),
        fallback_records=sum(r['normal_fallback'] for r in rows),
        parent_layouts=1, candidates_regenerated=False, runtime_policy_modified=False,
        semantic_labels_unchanged=True, hidden_geometry_used=False, semantic_efficacy_proven=False)
    write(a.output / 'result.json', result)
    write(a.output / 'manifest.json', dict(status='complete', source_sha256=frozen,
        source_history=str(a.source), input_inventory_sha256=sha(a.source / 'artifact_hashes.json')))
    write(a.output / 'artifact_hashes.json', {q.name: sha(q) for q in sorted(a.output.iterdir())
        if q.is_file() and q.name != 'artifact_hashes.json'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
