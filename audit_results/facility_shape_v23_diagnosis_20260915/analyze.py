#!/usr/bin/env python3
"""Read-only V23 saved-stage projection/mesh diagnosis; no sensing or fusion."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as PlotPolygon
import numpy as np
import open3d as o3d
import shapely
from env.facility_shape_probe_v23 import ShapeProbeWorldV23
from utils.facility_metrics_v19 import clip_mesh_to_bounds
from utils.facility_outline_v23 import (OutlineEvaluatorV23, PROJECTIONS,
    projected_envelope, polygon_parts, boundary_samples, compare_projection)
from utils.reconstruction_metrics import ray_scene

SOURCE = ROOT/'audit_results/facility_shape_v23_probe_20260915'
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def close(a, b):
    if a is None or b is None:
        assert a == b
    else:
        assert abs(a-b) < 1e-10, (a,b)


class StaticWorld(ShapeProbeWorldV23):
    """Allow only frozen static reference construction, never sensor calls."""
    def sense(self):
        raise AssertionError('No new sensors in this diagnosis')
    def scan(self):
        raise AssertionError('No new sensors in this diagnosis')
    def step(self, action):
        raise AssertionError('No new actions in this diagnosis')


def mesh_read(path):
    with np.load(path, allow_pickle=False) as a:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(a['vertices']),
                                       o3d.utility.Vector3iVector(a['triangles']))
    return mesh


def triangle_witness(scene, mesh, point):
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray([point], np.float32)), nthreads=1)
    identifier = int(closest['primitive_ids'].numpy()[0])
    vertices = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[identifier]]
    cross = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
    norm = float(np.linalg.norm(cross))
    return dict(source_triangle_index=identifier, source_triangle_vertices=vertices.tolist(),
        source_triangle_area_m2=norm/2, source_triangle_abs_normal_z=float(abs(cross[2])/norm),
        source_triangle_z_range=[float(vertices[:,2].min()),float(vertices[:,2].max())],
        nearest_original_mesh_point=closest['points'].numpy()[0].astype(float).tolist(),
        submitted_point=list(map(float,point)),
        note='Closest triangle in the ORIGINAL saved mesh; evaluator clipping only locates its witness')


def inspect_mesh(mesh, prediction):
    vertices = np.asarray(prediction.vertices)
    triangles = np.asarray(prediction.triangles)
    corners = vertices[triangles]
    cross = np.cross(corners[:,1]-corners[:,0], corners[:,2]-corners[:,0])
    norms = np.linalg.norm(cross, axis=1)
    areas = norms/2
    low = corners[:,:,2].max(axis=1) <= .05
    scene = ray_scene(mesh)
    extrema = []
    for axis in (0,1):
        for direction in ('min','max'):
            target = getattr(vertices[:,axis], direction)()
            matches = np.flatnonzero(np.isclose(vertices[:,axis], target, atol=1e-9, rtol=0))
            extrema.append(dict(axis='xy'[axis], direction=direction, coordinate=float(target),
                matching_vertex_count=len(matches), matching_vertex_z_range=[float(vertices[matches,2].min()), float(vertices[matches,2].max())],
                witness=triangle_witness(scene, mesh, vertices[matches[0]])))
    return dict(prediction_window_bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
        original_vertices=len(mesh.vertices), original_triangles=len(mesh.triangles),
        window_triangles=len(triangles), window_area_m2=float(areas.sum()),
        triangles_entirely_at_or_below_5cm=int(low.sum()), low_triangle_area_m2=float(areas[low].sum()),
        low_triangle_area_fraction=float(areas[low].sum()/areas.sum()),
        low_triangle_area_weighted_abs_normal_z=float(np.average(np.abs(cross[low,2])/norms[low],weights=areas[low])),
        extrema=extrema,
        low_height_threshold_is_diagnostic_only=True, no_low_triangles_removed=True)


def main():
    assert not (OUTPUT/'result.json').exists(), 'Do not overwrite sealed diagnosis'
    manifest = read(SOURCE/'manifest.json')
    assert manifest['status'] == 'complete'
    inventory = read(SOURCE/'artifact_hashes.json')
    for name, expected in inventory.items():
        assert sha(SOURCE/name) == expected, name
    for name, expected in manifest['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    assert sha(SOURCE/'sources.zip') == manifest['source_archive_sha256']
    aggregate = read(SOURCE/'result.json')
    assert aggregate['status'] == 'complete'
    rows, stages, sources = [], [], {}
    fig, axes = plt.subplots(4,3,figsize=(12.4,13.8))
    figure_row = 0
    for index, kind in enumerate(('simple','complex')):
        result = read(SOURCE/f'case_{index:02d}/result.json')
        verification = read(SOURCE/f'case_{index:02d}/verification.json')
        assert verification['status'] == 'passed' and verification['independent_process']
        assert verification['physical_process_id'] != verification['replay_process_id']
        world = StaticWorld(kind, sensor_model=manifest['config']['sensor_model'],noise_seed=manifest['config']['noise_seed'])
        evaluator = OutlineEvaluatorV23.from_world(world, **manifest['config']['outline'])
        assert evaluator.reference_signature == manifest['cases'][index]['outline_reference_signature']
        assert world.step_count == 0
        reference = evaluator.assets[0]
        for checkpoint in result['checkpoints']:
            row = checkpoint['outline']['instances'][0]
            stages.append(dict(kind=kind, stage=checkpoint['stage'], action_id=checkpoint['action_id'],
                Q=row['05cm']['outline_quality'], completed=row['completed'], dimensions=row['dimensions'],
                surface=checkpoint['surface']['instances'][0]['05cm'],
                surface_accuracy=checkpoint['surface']['instances'][0]['accuracy'],
                projections={name:dict(q=min(p['iou'],p['05cm']['f1']),iou=p['iou'],boundary_F1=p['05cm']['f1'],
                    hausdorff_upper_m=p['hausdorff_upper_m'],predicted_components=p['predicted_components'])
                    for name,p in row['projections'].items()}))
            if checkpoint['stage'] not in ('coarse','extra'):
                continue
            stage = checkpoint['stage']
            path = SOURCE/f'case_{index:02d}/meshes/{stage}.npz'
            sources[str(path.relative_to(ROOT))] = sha(path)
            mesh = mesh_read(path)
            prediction = clip_mesh_to_bounds(mesh, reference['bounds'])
            details = inspect_mesh(mesh, prediction)
            projections = {}
            for col,(name,xy) in enumerate(PROJECTIONS.items()):
                predicted = projected_envelope(prediction, xy)
                truth = reference['projections'][name]
                recomputed = compare_projection(predicted,truth,evaluator.spacing,evaluator.thresholds)
                expected = row['projections'][name]
                for key in ('iou','hausdorff_lower_m','hausdorff_upper_m','predicted_area_m2','reference_area_m2'):
                    close(recomputed[key],expected[key])
                close(recomputed['05cm']['f1'],expected['05cm']['f1'])
                parts=sorted(polygon_parts(predicted),key=lambda p:-p.area)
                # Inspect all connected projection components without deleting
                # any. No modified prediction or filtered score is produced.
                components=[dict(area_m2=float(p.area),boundary_length_m=float(p.length),bounds=list(p.bounds)) for p in parts]
                samples,weights,endpoints,spacing=boundary_samples(predicted,evaluator.spacing)
                candidates=np.vstack([samples,endpoints])
                distances=shapely.distance(shapely.points(candidates),truth.boundary)
                witness=candidates[int(np.argmax(distances))]
                values=np.asarray(prediction.vertices)
                offset=np.linalg.norm(values[:,xy]-witness,axis=1)
                nearest=int(np.argmin(offset))
                projections[name]=dict(original_metrics=expected, components=components,
                    directed_predicted_boundary_max_m=float(np.max(distances)), predicted_boundary_witness=witness.tolist(),
                    nearest_window_vertex_projection_offset_m=float(offset[nearest]),
                    source_triangle_witness=triangle_witness(ray_scene(mesh),mesh,values[nearest]) if offset[nearest]<1e-7 else None)
                ax=axes[figure_row,col]
                for p in polygon_parts(truth):
                    ax.add_patch(PlotPolygon(np.asarray(p.exterior.coords),closed=True,facecolor='#7dabd6',edgecolor='#244b70',alpha=.38,lw=1.6))
                for p in parts:
                    ax.add_patch(PlotPolygon(np.asarray(p.exterior.coords),closed=True,facecolor='#e98a36',edgecolor='#aa4a12',alpha=.55,lw=.55))
                bounds=reference['bounds'][:,xy]
                ax.set_xlim(bounds[0,0]-.08,bounds[1,0]+.08)
                ax.set_ylim(bounds[0,1]-.08,bounds[1,1]+.08)
                ax.set_aspect('equal'); ax.grid(alpha=.16)
                q=min(expected['iou'],expected['05cm']['f1'])
                ax.set_title(f'{kind}, {stage} ({checkpoint["action_id"]} actions) | {name.upper()}\nQv={q:.3f}; IoU={expected["iou"]:.3f}; boundary F1={expected["05cm"]["f1"]:.3f}',fontsize=10)
                ax.set_xlabel('xyz'[xy[0]]+' (m)'); ax.set_ylabel('xyz'[xy[1]]+' (m)')
            rows.append(dict(kind=kind,stage=stage,action_id=checkpoint['action_id'],
                saved_Q=row['05cm']['outline_quality'],saved_dimensions=row['dimensions'],
                mesh_diagnostics=details,projections=projections))
            figure_row+=1
    fig.suptitle('V23 actual saved TSDF projections: coarse scan and added side/rear views',fontsize=14,y=.993)
    fig.legend(handles=[Patch(facecolor='#7dabd6',edgecolor='#244b70',alpha=.38,label='Frozen evaluation reference'),
                        Patch(facecolor='#e98a36',edgecolor='#aa4a12',alpha=.55,label='Actual saved TSDF projection')],loc='upper center',bbox_to_anchor=(.5,.97),ncol=2,frameon=False)
    fig.text(.5,.008,'Evaluation window only; no triangle deletion, new fusion or 3D completion. GT is explanatory and never planner input.\nBounded 2D holes are filled only by the frozen metric; an open L-shaped footprint is not filled.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.945))
    fig.savefig(OUTPUT/'coarse_extra_projections.png',dpi=150)
    plt.close(fig)
    for name in ('manifest.json','result.json','artifact_hashes.json','sources.zip'):
        sources[str((SOURCE/name).relative_to(ROOT))]=sha(SOURCE/name)
    for i in range(2):
        for name in ('result.json','verification.json','artifact_hashes.json'):
            path=SOURCE/f'case_{i:02d}'/name;sources[str(path.relative_to(ROOT))]=sha(path)
    source_hashes={**manifest['source_sha256'],str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))}
    output=dict(status='complete_read_only_saved_mesh_diagnosis',input_inventory_entries_verified=len(inventory),
        frozen_source_entries_verified=len(manifest['source_sha256']),source_root=str(SOURCE),
        paired_history=aggregate['paired_history'],original_gates=aggregate['gates'],
        stages=stages,selected_stage_mesh_diagnostics=rows,input_sha256=sources,
        no_fusion=True,no_sensing=True,no_step=True,no_predicted_geometry_filter=True,
        gt_attribution_is_evaluation_only=True,shape_reference_rebuilt_statically_and_signature_verified=True)
    save(OUTPUT/'result.json',output)
    save(OUTPUT/'source_sha256.json',source_hashes)
    save(OUTPUT/'artifact_hashes.json',{p.name:sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(dict(status=output['status'],stages=len(stages),projection_checks=len(rows)*3,
        output=str(OUTPUT),source_sha256=sha(Path(__file__))),indent=2))


if __name__=='__main__':
    main()
