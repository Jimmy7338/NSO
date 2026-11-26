"""Build a frozen, source-only visibility reference with the original B11 metric."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import trimesh

from mechanism_world import ANALYSES, HERE, LAYOUTS, SPECS, build_scene
from prepare import digest, write

METRICS = ANALYSES / 'dual_template_exploration_20261119/audit/surface_metrics.py'
sys.path.insert(0,str(METRICS.parent))
from surface_metrics import fixed_camera_poses, ray_scene, subset_mesh, surface_samples, unique_sampling_faces, visible_reference


def source_mesh(layout):
    scene, records, meta = build_scene(layout)
    vertices, triangles, labels = [], [], []
    offset = 0
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        part = scene.geometry[key]
        points = trimesh.transform_points(part.vertices,transform)
        indices = np.asarray(part.faces,np.int32)
        vertices.append(points)
        triangles.append(indices+offset)
        labels.extend([meta[node]['source_instance_id']] * len(indices))
        offset += len(points)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vertices)),
                                    o3d.utility.Vector3iVector(np.concatenate(triangles)))
    return mesh, np.asarray(labels,np.int16), records


def prepare(layout):
    began=time.monotonic()
    protocol_path = HERE/'reference_protocol.json'
    rules = json.loads(protocol_path.read_text())
    initial = HERE/'prepared'/layout/'initial_state.json'
    truth = initial.parent/'evaluator_truth.npz'
    state = json.loads(initial.read_text())
    q = SPECS[layout]
    out = HERE/'evaluator_references'/layout
    out.mkdir(parents=True,exist_ok=False)
    with np.load(truth,allow_pickle=False) as values:
        reachable = values['reachable']
    config = SimpleNamespace(**state['actual_config'])
    poses, cells = fixed_camera_poses(reachable,config,state['anchor'][:2])
    mesh, labels, records = source_mesh(layout)
    source, source_labels, duplicates = unique_sampling_faces(mesh,labels)
    ray = ray_scene(mesh)
    reference, meta = {'camera_poses':poses}, {}
    seed0 = rules['base_seed'] + (1000 if q['parent']=='gallery' else 0)
    for name, target in (('all',None),('open31',31),('closed118',118)):
        selected = source if target is None else subset_mesh(source,source_labels==target)
        count = rules['global_area_samples'] if target is None else rules['per_target_area_samples']
        seed = seed0 if target is None else seed0+target
        points, _ = surface_samples(selected,count,seed)
        visible, first, rays = visible_reference(points,ray,poses,config,tolerance_m=rules['first_hit_tolerance_m'])
        assert visible.any(), (layout,name)
        reference[name] = points[visible].astype(np.float32)
        reference[name+'_first_pose'] = first[visible]
        area = selected.get_surface_area()
        meta[name] = dict(sampled_source_area_m2=area,drawn_samples=count,visible_samples=int(visible.sum()),
            estimated_visible_area_m2=float(area*visible.mean()),area_per_sample_m2=float(area/count),
            seed=seed,first_hit_rays=rays)
    np.savez_compressed(out/'fixed_reference.npz',**reference)
    assert o3d.io.write_triangle_mesh(str(out/'source_mesh.ply'),mesh)
    deps=[Path(__file__),HERE/'mechanism_world.py',protocol_path,truth,initial,METRICS]
    receipt=dict(status='fixed_source_only_surface_reference_prepared',layout=layout,
        source_inputs_sha256={str(p):digest(p) for p in deps},reference_npz_sha256=digest(out/'fixed_reference.npz'),
        source_mesh_sha256=digest(out/'source_mesh.ply'),metadata=meta,raw_source_triangles=len(mesh.triangles),
        duplicate_sampling_handling=duplicates,reachable_cells=int(reachable.sum()),camera_lattice_cells=cells,
        camera_pose_count=len(poses),source_object_records=records,actor_trajectory_read=False,actor_executed=False,
        reference_source_is_native_geometry_not_template=True, elapsed_s=time.monotonic()-began)
    write(out/'RECEIPT.json',receipt)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',required=True,choices=LAYOUTS)
    a=prepare(parser.parse_args().layout)
    print(json.dumps(dict(layout=a['layout'],status=a['status'],seconds=a['elapsed_s'],visible=a['metadata'])),flush=True)
