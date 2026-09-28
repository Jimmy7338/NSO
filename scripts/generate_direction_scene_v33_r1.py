#!/usr/bin/env python3
"""Generate the single approved V33 geometric correction; no visibility or DP."""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.box_union_geometry_v30 import BoxV30, surface_audit_v30, union_volume_v30
from nso.direction_scene_v33 import (rotate_box_v33, rotate_xy_v33,
    complete_safe_cells_v33, graph_v33, prefix_states_v33, exact_asset_faces_v33)

SOURCE = ROOT/'configs/virtual3d/v33_direction_scene_20260917.json'
OUTPUT = ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json'
R0 = ROOT/'audit_results/v33_direction_information_r0_20260917'
SOURCE_SHA = '4bdcf84cab68e205e160f25182cf2ce09cc221fea9a5593d70b57cf03c213390'
MANIFEST_SHA = 'ba038cc054b980ba900ae4644cd0d809b0154e9dd35d97e866e022763f2abbf7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_r0():
    assert sha(SOURCE) == SOURCE_SHA
    assert sha(R0/'manifest.json') == MANIFEST_SHA
    manifest = json.loads((R0/'manifest.json').read_text())
    inventory = json.loads((R0/'artifact_hashes.json').read_text())
    assert {str(p.relative_to(R0)) for p in R0.rglob('*') if p.is_file()
            and p.name != 'artifact_hashes.json'} == set(inventory)
    for name, expected in inventory.items():
        assert sha(R0/name) == expected, name
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    with zipfile.ZipFile(R0/'sources.zip') as archive:
        assert set(archive.namelist()) == set(manifest['source_sha256'])
        for name, expected in manifest['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    return manifest


def differences(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), path
        return [d for key in a for d in differences(a[key], b[key], (*path, key))]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, (*path, i))]
    return [] if a == b else [dict(path=list(path), before=a, after=b)]


def corrected(old):
    new = deepcopy(old)
    new['version'] = 'v33-direction-service-scene-r1'
    new['geometry_freeze_scope'] = 'one defect-grounded r1 after retained r0 observability failure; no further correction or reward/budget sweep'
    allowed = {('version',), ('geometry_freeze_scope',)}
    for pi, parent in enumerate(new['parents']):
        turns = parent['device_frame']['quarter_turns_ccw']
        assert parent['geometry_facts']['ribs_local_y'][0] == [1.6, 1.8]
        parent['geometry_facts']['ribs_local_y'][0][0] = 1.5
        allowed.add(('parents', pi, 'geometry_facts', 'ribs_local_y', 0, 0))
        for hi, hypothesis in enumerate(parent['hypotheses']):
            boxes = hypothesis['assets'][0]['boxes']
            guard = rotate_box_v33(boxes[0], -turns)
            rib = rotate_box_v33(boxes[3], -turns)
            assert guard == [-2.75, 2.75, 1.1, 1.5, 0., 2.]
            assert rib[2:] == [1.6, 1.8, .4, 1.8]
            guard[5] = 1.8
            rib[2] = 1.5
            boxes[0] = rotate_box_v33(guard, turns)
            boxes[3] = rotate_box_v33(rib, turns)
            base = ('parents', pi, 'hypotheses', hi, 'assets', 0, 'boxes')
            allowed.add((*base, 0, 5))
            allowed.add((*base, 3, 2 if turns == 0 else 1))
    changes = differences(old, new)
    assert {tuple(d['path']) for d in changes} == allowed
    return new, changes


def material(asset):
    audit = surface_audit_v30(exact_asset_faces_v33(asset))
    volume = union_volume_v30([BoxV30(tuple(b), asset['id']) for b in asset['boxes']])
    assert audit['closed_oriented_two_manifold'], audit
    assert abs(volume-audit['signed_volume']) < 1e-10
    return dict(vertical_area_m2=sum(f.area for f in exact_asset_faces_v33(asset, vertical_only=True)),
                full_exterior_area_m2=audit['area'], volume_m3=volume,
                closed_oriented_two_manifold=True)


def check_geometry(old, new):
    rows = []
    for before, after in zip(old['parents'], new['parents']):
        turns = after['device_frame']['quarter_turns_ccw']
        graph = graph_v33(after)
        assert graph == graph_v33(before)
        poses, edges, anchor = graph
        seen, queue = {anchor}, deque([anchor])
        while queue:
            for _, nxt in edges[queue.popleft()]:
                if nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
        assert len(seen) == len(poses)
        prefix = prefix_states_v33(after)
        assert prefix == prefix_states_v33(before)
        assert len(prefix) == 19 and prefix[0] == prefix[-1]
        summaries = []
        for prior_h, h in zip(before['hypotheses'], after['hypotheses']):
            boxes = h['assets'][0]['boxes']
            cells = complete_safe_cells_v33(after['local_grid'], boxes, turns, after['robot_radius_m'])
            assert cells == tuple(map(tuple, after['nav_cells']))
            one_h = deepcopy(after); one_h['hypotheses'][0] = deepcopy(h)
            assert graph_v33(one_h) == graph
            # Continuous shield containment at the unchanged paid-prefix origins.
            # This does not render depth, test every target, or evaluate reward.
            for pose in prefix:
                ox, oy = rotate_xy_v33(pose[:2], -turns)
                assert oy == 0
                for b in boxes[3:]:
                    for gx, gy, z in itertools.product(b[:2], b[2:4], b[4:6]):
                        x, y = rotate_xy_v33((gx, gy), -turns)
                        t = 1.1/y
                        assert -2.75 < ox+(x-ox)*t < 2.75
                        assert 0 < .9+(z-.9)*t < 1.8
            a, b = material(prior_h['assets'][0]), material(h['assets'][0])
            summaries.append(dict(hypothesis=h['id'], before=a, after=b,
                delta={key:b[key]-a[key] for key in ('vertical_area_m2','full_exterior_area_m2','volume_m3')}))
        assert all(abs(summaries[0]['after'][k]-summaries[1]['after'][k]) < 1e-10
                   for k in ('vertical_area_m2','full_exterior_area_m2','volume_m3'))
        r0 = json.loads((R0/(after['id']+'_geometry.json')).read_text())
        missing = []
        for t in r0['tables']:
            groups = {'narrow_slot': [], 'guard_top_back_band': []}
            for target in t['unobservable_targets']:
                x, y = rotate_xy_v33(target['point'][:2], -turns)
                z = target['point'][2]
                group = 'guard_top_back_band' if z > 1.8 else 'narrow_slot'
                assert 1.5-1e-9 <= y <= 1.6+1e-9
                groups[group].append(dict(local_point=[x,y,z],area_m2=target['area']))
            missing.append(dict(hypothesis=t['hypothesis'], groups={k:dict(count=len(v),
                area_m2=sum(p['area_m2'] for p in v), local_points=v) for k,v in groups.items()}))
        rows.append(dict(parent=after['id'], safe_cells=len(after['nav_cells']), poses=len(poses),
            all_safe_connected=True, complete_graph_unchanged=True, prefix_states_unchanged=True,
            continuous_prefix_rib_shadow=True, material=summaries, r0_missing=missing))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    if not args.generate:
        parser.print_help(); return
    if OUTPUT.exists(): raise FileExistsError('r1 config already exists; no replacement')
    if shutil.disk_usage(ROOT).free < 64*1024**2: raise OSError('64 MiB reserve required')
    manifest = verify_r0()
    old = json.loads(SOURCE.read_text())
    new, changes = corrected(old)
    rows = check_geometry(old, new)
    assert not any(m in sys.modules for m in ('nso.direction_information_v33',
        'nso.finite_belief_solver_v33', 'env.virtual3d', 'nso.mapping3d'))
    verify_r0()
    payload = (json.dumps(new, indent=2, allow_nan=False)+'\n').encode()
    with OUTPUT.open('xb') as handle: handle.write(payload)
    print(json.dumps(dict(status='geometry_checks_passed_not_full_visibility',
        generator_sha256=sha(Path(__file__)), output=str(OUTPUT.relative_to(ROOT)),
        output_sha256=sha(OUTPUT), protected_r0_sources=len(manifest['source_sha256']),
        changes=changes, parents=rows, revision_count=1,
        full_visibility_computed=False, DP_instances=0, worlds=0, Q_evaluations=0),
        allow_nan=False, separators=(',',':')))


if __name__ == '__main__': main()
