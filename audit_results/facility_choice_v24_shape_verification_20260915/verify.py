#!/usr/bin/env python3
"""Read-only sealed-byte/mesh-array/arithmetic audit; no backend or world import."""
import sys
sys.dont_write_bytecode = True
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
from datetime import datetime, timezone
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
SHAPE = ROOT/'audit_results/facility_choice_v24_shape_prefix_20260915'
LIMIT = 100*1024
RESERVE = 32*1024**2

def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()
def inventory(root):
    items = read(root/'artifact_hashes.json')
    paths = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root/'artifact_hashes.json'}
    assert paths == set(items), root
    for name, h in items.items():
        p = root/name
        assert p.resolve().is_relative_to(root.resolve()) and sha(p) == h, p
    return len(items), sha(root/'artifact_hashes.json')
def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('vertices', 'triangles')}
def mh(a): return {k:array_hash(v) for k,v in a.items()}
def add(items):
    v, t, offset = [], [], 0
    for a in items:
        v.append(a['vertices']); t.append(a['triangles']+offset); offset += len(a['vertices'])
    return dict(vertices=np.concatenate(v), triangles=np.concatenate(t).astype(np.int32))
def bounds(a):
    if not len(a['triangles']): return None
    p = a['vertices'][np.unique(a['triangles'])]
    return np.stack([p.min(0), p.max(0)])
def near(a,b): assert abs(a-b) <= 2e-14, (a,b)
def windows():
    # Read literal public front coordinates from the already frozen source.
    # Bounds reproduce its declared evaluator-only formula; no world is built.
    tree = ast.parse((ROOT/'env/facility_choice_v24.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='LAYOUTS_V24' for t in n.targets))
    result = {}
    for key, value in zip(assignment.value.keys, assignment.value.values):
        fronts = ast.literal_eval(next(x.value for x in value.keywords if x.arg=='fronts'))
        result[ast.literal_eval(key)] = [np.array([[cx-(.92 if i==0 else 1.2), front-.12, .01],
            [cx+(1.2 if i==0 else .92), front+1.52, 1.72]]) for i,(cx,front) in enumerate(fronts)]
    return result
def quality_arithmetic(score):
    assert len(score['instances']) == 2
    for tag in ('02cm','05cm','10cm'):
        qualities, f1s = [], []
        for row in score['instances']:
            p = list(row['projections'].values())
            assert len(p) == 3
            q = float(np.mean([min(x[tag]['f1'], x['iou']) for x in p]))
            f = float(np.mean([x[tag]['f1'] for x in p]))
            near(q, row[tag]['outline_quality']); near(f, row[tag]['outline_f1'])
            qualities.append(q); f1s.append(f)
        near(float(np.mean(qualities)), score[tag]['outline_macro_quality'])
        near(float(np.mean(f1s)), score[tag]['outline_macro_f1'])
        near(score['coverage_2d']*score[tag]['outline_macro_quality'], score[tag]['joint_outline'])
    assert score['missing_asset_count'] == sum(x['missing'] for x in score['instances'])
    near(score['completion_fraction'], float(np.mean([x['completed'] for x in score['instances']])))
    assert score['eligible'] == bool(score['coverage_2d']>=.8 and score['returned'] and not score['collisions'] and not score['failed'] and score['primitive_budget_compliant'] is True)
def pair_bundle(case):
    return dict(instances=[{k:r[k] for k in ('observed_slot','observed_frames','ground','completion','geometry_arrays','mesh_sha256','no_new_inferred_triangles','completed_equals_measured')} for r in case['instances']],
        seed_geometry=[None if r['seed'] is None else {k:r['seed'][k] for k in ('action_id','observed_slot','observed_seed_xyz','pixel','component_pixels','distance_to_component_median_m')} for r in case['instances']],
        representation_mesh_sha256=case['representation_mesh_sha256'])

def main():
    assert not (OUT/'verification.json').exists(), 'Use a new audit root; do not overwrite sealed verification'
    assert shutil.disk_usage(OUT).free > RESERVE+LIMIT
    old_count, old_inventory = inventory(SOURCE)
    new_count, new_inventory = inventory(SHAPE)
    original = read(SOURCE/'manifest.json'); manifest = read(SHAPE/'manifest.json'); result = read(SHAPE/'result.json')
    assert original['status']==manifest['status']==result['status']=='complete'
    assert len(original['source_sha256'])==145 and len(manifest['source_sha256'])==147
    for name,h in manifest['source_sha256'].items(): assert sha(ROOT/name)==h, name
    for name,h in manifest['input_sha256'].items(): assert sha(ROOT/name)==h, name
    assert sha(SHAPE/'sources.zip')==manifest['source_archive_sha256']
    assert sha(SOURCE/'sources.zip')==original['source_archive_sha256']
    for path, expected in [(SOURCE/'sources.zip', original['source_sha256']),
                           (SHAPE/'sources.zip', {k:v for k,v in manifest['source_sha256'].items() if k not in original['source_sha256']})]:
        with zipfile.ZipFile(path) as z:
            assert set(z.namelist())==set(expected)
            for k,h in expected.items(): assert hashlib.sha256(z.read(k)).hexdigest()==h
    assert (result['saved_packets_processed'],result['backend_observe_calls'],result['backend_snapshots'])==(980,1960,8)
    assert result['new_sensor_frames']==result['new_physical_actions']==result['new_tsdf_fusions']==0
    summaries, case_inventories, pids = [], [], []
    atlas = windows()
    for c in result['cases']:
        index = c['index']; folder = SOURCE/f'case_{index:02d}'
        saved=read(folder/'result.json'); receipt=read(folder/'verification.json'); timing=read(folder/'timing.json')
        case_inventories.append(inventory(folder)[0])
        assert receipt['status']=='passed' and receipt['physical_process_id']==timing['process_id'] and receipt['physical_process_id']!=receipt['replay_process_id']
        pids += [receipt['physical_process_id'],receipt['replay_process_id']]
        paid = 234 if index<2 else 254
        assert saved['paid_actions']==c['endpoint_action_id']==paid and c['saved_packets_processed']==paid+1
        assert [r['action_id'] for r in saved['trace']]==list(range(paid+1))
        assert digest([r['packet_sha256'] for r in saved['trace']])==c['input_packet_sequence_sha256']
        assert read(SHAPE/f'case_{index:02d}.json')==c
        assert c['observed_track_summary']==saved['marker_tracks']
        near(c['prefix_coverage'], saved['final_coverage_2d'])
        assert c['prefix_returned']==saved['returned_to_anchor']
        source_raw=ROOT/c['raw_source']['path']
        assert sha(source_raw)==c['raw_source']['sha256']
        with np.load(source_raw,allow_pickle=False) as z:
            assert {k:array_hash(z[k]) for k in ('vertices','triangles','vertex_colors')}==c['raw_source']['array_sha256']==saved['final_mesh_sha256']
        raw=arrays(source_raw); models=[]; slot_reports=[]
        for row in c['instances']:
            slot=row['observed_slot'];seed=row['seed'];action=seed['action_id'];pixel=seed['pixel']
            assert row['observed_frames']==paid+1
            first=next(r['action_id'] for r in saved['trace'] if any(x['observed_track_index']==slot for x in r['marker_components']))
            assert action==first and seed['packet_sha256']==saved['trace'][action]['packet_sha256']
            with np.load(folder/'packets'/f'{action:04d}.npz',allow_pickle=False) as z:
                depth=z['frame__depth_m'];K=z['frame__intrinsic'];T=z['frame__world_from_camera'];rgb=z['frame__color_rgb']
                rr,cc=pixel;assert depth[rr,cc]>.15 and list(rgb[rr,cc]) in [[40,100,220],[220,60,40]]
                # Same arithmetic order as the documented real-pixel backprojection.
                d=depth[rr,cc]
                local=np.array([(cc-K[0,2])*d/K[0,0],(rr-K[1,2])*d/K[1,1],d])
                hit=local@T[:3,:3].T+T[:3,3]
                assert np.allclose(hit,seed['observed_seed_xyz'],rtol=0,atol=2e-15)
            matches=[i for i,b in enumerate(atlas[c['parent']]) if np.all(hit>=b[0]) and np.all(hit<=b[1])]
            assert matches==row['fixed_seed_association']['matches']==[slot]
            measured=arrays(SHAPE/row['mesh_files']['observed_mesh']); inferred=arrays(SHAPE/row['mesh_files']['inferred_mesh'])
            completed=add([measured,inferred])
            for key,model in [('observed_mesh',measured),('inferred_mesh',inferred),('completed_mesh',completed)]:
                assert mh(model)==row['mesh_sha256'][key]
                assert {k:len(a) for k,a in model.items()}==row['mesh_sizes'][key]
            assert len(inferred['triangles'])==0 and not row['completion']['accepted']
            assert row['completion']['reason']=='enclosure_contradicts_observed_free_rays'
            assert row['no_new_inferred_triangles'] and row['completed_equals_measured'] and mh(completed)==mh(measured)
            a=row['completion']['support_audit'];near(row['completion']['free_ray_conflicts']/a['free_ray_box_hits'],a['free_ray_conflict_fraction'])
            geometry=row['geometry_arrays']
            assert geometry['measured_points_xyz']['count']==sum(geometry[k]['count'] for k in ('ground_points_xyz','cleaned_points_xyz','unassigned_points_xyz'))
            other=[x for x in row['completed_per_reference_window'] if x['id']!=slot]
            assert len(other)==1 and other[0]['missing'] and other[0]['05cm']['outline_quality']==0
            models.append(dict(measured=measured,inferred=inferred,completed=completed))
            slot_reports.append(dict(slot=slot,first_seed_action=action,seed_matches_reference_ids=matches,
                cleaned_point_count=geometry['cleaned_points_xyz']['count'],saved_measured_bounds_m=bounds(measured).tolist(),
                measured_triangles=len(measured['triangles']),other_reference_window_missing=True,
                inferred_triangles=0,completion_accepted=False,completion_reason=row['completion']['reason'],
                free_ray_conflict_fraction=a['free_ray_conflict_fraction'],free_ray_conflicts=row['completion']['free_ray_conflicts'],
                minimum_background_bracket_count=min(a['background_bracketed_edge_counts'].values())))
        b0,b1=[np.asarray(x['saved_measured_bounds_m']) for x in slot_reports]
        axis_gap=np.maximum(b1[0]-b0[1],b0[0]-b1[1])
        assert np.any(axis_gap>0), 'Actual saved instance AABBs overlap on all axes'
        assert c['instances'][0]['geometry_arrays']['cleaned_points_xyz']!=c['instances'][1]['geometry_arrays']['cleaned_points_xyz']
        assert not c['duplicate_seeded_components'] and c['all_observed_instances_nonempty'] and c['minimum_instance_separation_gate_passed'] and c['observed_instance_seed_and_separation_gate_passed']
        assoc=c['fixed_seed_association']
        assert assoc['seed_association_gate_passed'] and not assoc['missing_reference_ids'] and not assoc['duplicate_reference_ids'] and not assoc['extra_observed_track_receipts']
        assert assoc['observed_track_count']==assoc['expected_track_count']==2
        assembled=dict(raw=raw,**{name:add([m[name] for m in models]) for name in ('measured','inferred','completed')})
        assert {k:mh(a) for k,a in assembled.items()}==c['representation_mesh_sha256']
        for rep,score in c['representations'].items():
            quality_arithmetic(score)
            near(score['coverage_2d'],c['prefix_coverage'])
            assert score['returned'] and not score['collisions'] and not score['failed'] and score['budget_verified']
            assert score['completion_fraction']==0
            if rep=='inferred': assert score['05cm']['outline_macro_quality']==0 and score['missing_asset_count']==2
        assert c['representations']['completed']==c['representations']['measured']
        score=c['representations']
        summaries.append(dict(index=index,parent=c['parent'],assignment=c['assignment'],paid_prefix_actions=paid,
            coverage=c['prefix_coverage'],prefix_metric_eligible=score['completed']['eligible'],
            raw_Q=score['raw']['05cm']['outline_macro_quality'],measured_Q=score['measured']['05cm']['outline_macro_quality'],
            raw_J=score['raw']['05cm']['joint_outline'],measured_J=score['measured']['05cm']['joint_outline'],
            delta_Q=score['measured']['05cm']['outline_macro_quality']-score['raw']['05cm']['outline_macro_quality'],
            full_shape_completion_fraction=0,inferred_Q=0,completed_equals_measured=True,
            seed_association_passed=True,minimum_separation_passed=True,
            saved_instance_meshes_provably_AABB_disjoint=True,maximum_axis_separation_m=float(axis_gap.max()),slots=slot_reports))
    assert len(set(pids))==8
    for pair,(i,j) in zip(result['pairs'],[(0,1),(2,3)]):
        left,right=pair_bundle(result['cases'][i]),pair_bundle(result['cases'][j])
        assert left==right and pair['geometry_and_rejection_pair_identical']
        assert digest(left)==pair['left_geometry_sha256']==pair['right_geometry_sha256']
    # Recheck immutable inputs at the end, including all source paths.
    assert inventory(SOURCE)==(old_count,old_inventory) and inventory(SHAPE)==(new_count,new_inventory)
    for name,h in manifest['source_sha256'].items(): assert sha(ROOT/name)==h
    report=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),verified_source_count=147,
        original_artifacts_verified=old_count,processed_artifacts_verified=new_count,original_case_inventory_counts=case_inventories,
        original_inventory_sha256=old_inventory,processed_inventory_sha256=new_inventory,
        processed_result_sha256=sha(SHAPE/'result.json'),verifier_source_sha256=sha(Path(__file__).resolve()),
        verified_arrays_scope='16 saved instance meshes, 4 source TSDF meshes, and arithmetic-only concatenations for all representation hashes',
        paired_geometry_and_rejections_rechecked=True,no_duplicate_whole_component_or_saved_instance_mesh_detected=True,
        new_worlds_instantiated=0,new_backend_observations=0,new_backend_snapshots=0,new_tsdf_fusions=0,new_quality_evaluations=0,
        scope='Sealed artifact, source, real seed pixel, saved mesh separation, recorded metric arithmetic and result-consistency audit; not an independent backend rerun or task experiment',
        metric_limit='Q arithmetic is independently checked from sealed projection statistics; projection distances, point partition hashes and free-ray conflicts are not recomputed',
        qualification_limit='P00 eligibility is for the fixed scripted prefix only; inferred-only zero models can be mission-eligible with Q=0. Formal choice task and semantic efficacy remain untested',
        cases=summaries)
    encoded=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(encoded)+16384<LIMIT
    assert shutil.disk_usage(OUT).free-len(encoded)-16384>=RESERVE
    with (OUT/'verification.json').open('xb') as f: f.write(encoded); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k:report[k] for k in ('status','verified_source_count','original_artifacts_verified','processed_artifacts_verified','no_duplicate_whole_component_or_saved_instance_mesh_detected')}))

if __name__=='__main__': main()
