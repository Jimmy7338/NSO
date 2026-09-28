#!/usr/bin/env python3
"""Fixed-view, RGB-only marker feasibility; no navigation or efficacy claim."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.virtual3d_inspection_v4 import (InspectionConfigV4, InspectionWorldV4,
                                        read_inspection_markers_rgb, union_surface_from_boxes)


def digest(array):
    value=np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def majority_label(labels):
    """Summarize only labels found by the reader; abstain when no cue exists."""
    known=np.asarray(labels)[np.asarray(labels)>1]
    return None if not len(known) else int(Counter(known.tolist()).most_common(1)[0][0])


def observe_fixture(config,seed,condition):
    world=InspectionWorldV4(config,seed=seed,semantic_condition=condition)
    # Fixture positioning belongs to the experiment, never to the reader.
    world.position,world.heading=world.inspection_truth[0]['front_pose']
    frame=world.sense()
    decoded=read_inspection_markers_rgb(frame.color_rgb)
    physical=world.inspection_truth[0]['visible_category']
    truth=world.inspection_truth[0]['true_category']
    row=dict(seed=seed,hidden_type='layered' if truth==3 else 'solid',
        physical_marker_class=physical,hidden_geometry_class=truth,
        relation=config.semantic_relation,appearance=config.appearance,
        semantic_condition=condition,reader_label=majority_label(decoded),
        delivered_label=majority_label(frame.semantic),
        reader_visible_pixels=int(np.count_nonzero(decoded)),
        delivered_visible_pixels=int(np.count_nonzero(frame.semantic)),
        reader_matches_physical_class=majority_label(decoded)==physical,
        reader_predicts_hidden_class=majority_label(decoded)==truth,
        delivered_predicts_hidden_class=majority_label(frame.semantic)==truth,
        rgb_sha256=digest(frame.color_rgb),depth_sha256=digest(frame.depth_m),
        depth_mm_sha256=digest(np.rint(frame.depth_m*1000).astype(np.int32)),
        scan_sha256=digest(world.scan().ranges_m),
        pose_sha256=digest(frame.world_from_camera),
        geometry_sha256=digest(np.asarray(world.mesh.vertices))+':'+digest(np.asarray(world.mesh.triangles)))
    return world,frame,row


def prior_probe(world,frame):
    """Only mapper-derived hypotheses and candidates; no truth enters models."""
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3,ObjectCompletionModel
    from nso.route_coverage_v2 import orientation_graph
    from utils.grid_geometry import inflated_obstacles
    from scipy.sparse.csgraph import dijkstra
    mapper=SemanticHistoryMapperV3(world.shape,world.config,world.config.truncation_m)
    mapper.update(frame,world.scan())
    geometry=ObjectCompletionModel(mapper,semantic=False)
    semantic=ObjectCompletionModel(mapper,semantic=True)
    safe=~inflated_obstacles(mapper.belief!=0,world.config.robot_radius_m/world.config.resolution_m)
    # Camera pose is a measured input; infer its current grid position from it.
    cell=mapper.grid_cell(frame.world_from_camera[:3,3])
    distances=np.full(world.shape,-1.)
    if safe[cell]:
        graph,cells,ids=orientation_graph(safe)
        forward=frame.world_from_camera[:3,2]
        heading=int(np.argmax([forward[1],forward[0],-forward[1],-forward[0]]))
        all_costs=dijkstra(graph,directed=True,indices=int(ids[cell])*4+heading)
        cost=all_costs.reshape(-1,4).min(axis=1);valid=np.isfinite(cost)
        distances[tuple(cells[valid].T)]=cost[valid]
    common=lambda model:[dict(center=np.asarray(obj['center']).tolist(),
        prior=float(obj['shelf_prior_probability']),posterior=float(obj['shelf_probability']),
        observed_points=int(obj['observed_points'])) for obj in model.objects]
    points_equal=np.array_equal(geometry.points,semantic.points)
    candidates_equal=geometry.candidates(safe,distances)==semantic.candidates(safe,distances)
    return dict(geometry_objects=common(geometry),semantic_objects=common(semantic),
        hypothesis_points_equal=points_equal,candidates_equal=candidates_equal,
        candidate_count=len(geometry.candidates(safe,distances)),
        geometry_predicted_area_m2=float(geometry.weights.sum()),
        semantic_predicted_area_m2=float(semantic.weights.sum()),
        role='hypothesis sensitivity only; no new depth fused after scoring; not measured reconstruction gain')


def mesh_surface_audit(config):
    """Check exterior union faces and analytic area, not just duplicate counts."""
    world=InspectionWorldV4(replace(config,complex_fraction=0.),seed=731)
    vertices=np.asarray(world.mesh.vertices);triangles=np.asarray(world.mesh.triangles)
    groups={}
    for triangle in triangles:
        face=vertices[triangle]
        key=tuple(sorted(map(tuple,np.round(face,8))))
        area=float(np.linalg.norm(np.cross(face[1]-face[0],face[2]-face[0]))/2)
        count,_=groups.get(key,(0,area));groups[key]=(count+1,area)
    xyz=vertices[triangles];centers=xyz.mean(axis=1)
    normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    primitives=np.asarray(world._solid_primitives);lo=primitives[:,:3];hi=lo+primitives[:,3:6]
    inside=lambda points:np.any(np.all((points[:,None,:]>=lo-1e-10)&(points[:,None,:]<=hi+1e-10),axis=2),axis=1)
    internal=int(np.count_nonzero(inside(centers+1e-6*normals)))
    inward_errors=int(np.count_nonzero(~inside(centers-1e-6*normals)))
    checks=[]
    cube=(0,0,0,1,1,1,2)
    for name,boxes,expected_area,expected_volume in (
        ('identical_boxes',[cube,cube],6.,1.),
        ('contained_box',[cube,(.3,.3,.3,.4,.4,.4,2)],6.,1.),
        ('overlapping_boxes',[cube,(.5,0,0,1,1,1,2)],8.,1.5),
        ('touching_boxes',[cube,(1,0,0,1,1,1,2)],10.,2.),
        ('nonconvex_cross',[(0,0,0,2,1,1,2),(.5,-.5,0,1,2,1,2)],14.,3.)):
        mesh,_,audit=union_surface_from_boxes(boxes)
        area=float(mesh.get_surface_area());volume=audit['union_volume_m3']
        assert np.isclose(area,expected_area,atol=1e-8,rtol=0) and np.isclose(volume,expected_volume,atol=1e-8,rtol=0)
        checks.append(dict(case=name,expected_area_m2=expected_area,actual_area_m2=area,
            expected_volume_m3=expected_volume,actual_volume_m3=volume,watertight=mesh.is_watertight()))
    duplicate_excess=sum(count-1 for count,area in groups.values())
    assert duplicate_excess==0 and internal==0 and inward_errors==0 and world.mesh.is_watertight()
    return dict(scope='all-solid marked ambiguous fixture; exact union exterior audit',
        union_construction=world.surface_measure_audit,
        duplicate_triangle_groups=sum(count>1 for count,area in groups.values()),
        duplicate_triangle_excess=duplicate_excess,
        duplicate_area_lower_bound_m2=sum((count-1)*area for count,area in groups.values()),
        coordinate_matching_tolerance_m=1e-8,
        triangles_checked=len(triangles),internal_triangle_faces=internal,inward_orientation_errors=inward_errors,
        watertight=world.mesh.is_watertight(),analytic_surface_area_checks=checks,
        finding='Solid objects no longer duplicate their front board. All scene primitives also use a coordinate-plane solid union for the evaluator exterior; internal and partial coplanar overlaps are removed by construction.',
        verification='Every emitted triangle was checked: its outward-offset centroid is outside all primitives and its inward-offset centroid is inside. Primitive boundary planes partition constant occupancy, so no unrepresented boundary crosses an emitted elementary face.',
        sensor_consistency='Scene tests additionally compare primitive sensor first hits with union-mesh hits at front/rear RGB-D fixtures; no sensor backend replacement.',
        remaining_approximations=['primitive boundary coordinates rounded at 1e-9m',
            'finite area-sampled evaluator reference','reachable visibility lattice rather than all continuous camera poses',
            'synthetic axis-aligned physical scene and ideal marker RGB; not real-facility surface ground truth'],
        cue_implication='RGB-only information evidence remains a fixed-view feasibility result, not measured navigation/reconstruction gain.')


def run(output):
    if output.exists():raise FileExistsError(output)
    sources=('scripts/check_inspection_semantic_cue_v4.py','env/virtual3d_inspection_v4.py',
        'env/virtual3d_v2.py','env/virtual3d.py','utils/rgbd_contract.py','utils/grid_geometry.py',
        'nso/semantic_completion_v3.py','nso/mapping3d.py','nso/mapping3d_v2.py','nso/camera_mapping_v2.py',
        'nso/route_coverage_v2.py')
    before={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    seeds=(731,732,733,734)
    config=InspectionConfigV4(bays_per_side=1,depth_sigma_m=0.,dropout=0.,
        appearance='marked',semantic_source='rgb_marker',observation_condition='ambiguous')
    rows=[];pairs=[];priors=[];interventions=[]
    for seed in seeds:
        primary=[]
        for fraction in (0.,1.):
            world,frame,row=observe_fixture(replace(config,complex_fraction=fraction),seed,'aligned')
            rows.append(row);primary.append((world,frame,row))
            probe=prior_probe(world,frame);probe.update(seed=seed,hidden_type=row['hidden_type'])
            priors.append(probe)
            # RGB inference must remain independent of the simulator's label IDs.
            world._geometry_labels={gid:1 for gid in world._geometry_labels}
            row['reader_unchanged_after_gt_semantics_erased']=np.array_equal(
                frame.semantic,world.sense().semantic)
            for name,changed,condition in (
                ('missing_marker',replace(config,appearance='gray'),'aligned'),
                ('shuffled_labels',config,'shuffled'),('absent_labels',config,'absent'),
                ('independent_relation',replace(config,semantic_relation='independent'),'aligned'),
                ('reversed_relation',replace(config,semantic_relation='reversed'),'aligned')):
                other,other_frame,control=observe_fixture(replace(changed,complex_fraction=fraction),seed,condition)
                control['control']=name;rows.append(control)
                invariant=dict(seed=seed,hidden_type=row['hidden_type'],control=name,
                    same_geometry=row['geometry_sha256']==control['geometry_sha256'],
                    same_depth=np.array_equal(frame.depth_m,other_frame.depth_m),
                    same_rgb=np.array_equal(frame.color_rgb,other_frame.color_rgb))
                if condition in ('shuffled','absent'):
                    invariant['same_scan']=row['scan_sha256']==control['scan_sha256']
                    assert all(invariant[key] for key in ('same_geometry','same_depth','same_rgb','same_scan'))
                interventions.append(invariant)
        low,high=primary[0][1],primary[1][1]
        maximum=float(np.max(np.abs(low.depth_m-high.depth_m)))
        pair=dict(seed=seed,raw_depth_max_difference_m=maximum,
            raw_depth_bitwise_equal=np.array_equal(low.depth_m,high.depth_m),
            depth_equal_at_1mm=np.array_equal(np.rint(low.depth_m*1000),np.rint(high.depth_m*1000)),
            rgb_different=not np.array_equal(low.color_rgb,high.color_rgb),
            rgb_reader_distinguishes_classes=primary[0][2]['reader_label']==2 and primary[1][2]['reader_label']==3)
        assert maximum<=1e-6 and pair['depth_equal_at_1mm'] and pair['rgb_reader_distinguishes_classes']
        # If the front panel is removed, a geometric control should reveal detail.
        open_frames=[observe_fixture(replace(config,complex_fraction=fraction,
            observation_condition='open'),seed,'aligned')[1] for fraction in (0.,1.)]
        pair['open_control_depth_pixels_different_over_1cm']=int(np.count_nonzero(
            np.abs(open_frames[0].depth_m-open_frames[1].depth_m)>.01))
        pairs.append(pair)
    summary={}
    for name in ('primary','missing_marker','shuffled_labels','absent_labels','independent_relation','reversed_relation'):
        subset=[row for row in rows if row.get('control','primary')==name]
        summary[name]=dict(fixtures=len(subset),reader_nonabstain=int(sum(row['reader_label'] is not None for row in subset)),
            reader_physical_class_accuracy=float(np.mean([row['reader_matches_physical_class'] for row in subset])),
            reader_hidden_class_accuracy=float(np.mean([row['reader_predicts_hidden_class'] for row in subset])),
            delivered_hidden_class_accuracy=float(np.mean([row['delivered_predicts_hidden_class'] for row in subset])))
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
    result=dict(status='complete',scope='fixed-view sensor-information feasibility; development fixtures only',
        no_navigation_run=True,no_claim_of_reconstruction_or_planning_advantage=True,
        seeds=list(seeds),noise='ideal RGB material, zero depth noise/dropout; not real-camera robustness',
        reader_inputs=['current RGB image only'],semantic_source='rgb_marker',
        geometry_fixture_access='positioning and evaluation only; not supplied to RGB reader or shape model',
        depth_claim='Paired front views agree after 1mm quantization; raw float ray roundoff is reported, so no exact-real-number indistinguishability claim.',
        transfer_claim='Artificial equipment identifiers only; no open-vocabulary or natural-image recognition validation.',
        source_sha256=before,summary=summary,paired_front_views=pairs,
        mesh_surface_measure_audit=mesh_surface_audit(config),
        physical_intervention_checks=interventions,observations=rows,shape_prior_sensitivity=priors)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(output),summary=summary,paired_front_views=pairs),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
