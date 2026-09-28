#!/usr/bin/env python3
"""Read-only, frozen-source candidate audit; GT is opened only after pool sealing.

No navigation or new observations. Exhaustive known-safe poses diagnose proposal
loss. Optional evaluator values are ideal-valid-depth single-frame visibility,
not TSDF improvement, route utility, or an implementable GT planning policy.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def worker(source, output, snapshot):
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import open3d as o3d
    from dataclasses import replace
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse.csgraph import dijkstra
    from scripts.eval_counterfactual_views import candidate_routes, remap, grid_config
    from nso.semantic_completion_v3 import ObjectCompletionModel
    from nso.quality_coverage3d import QualityCoveragePolicy
    from nso.route_coverage_v2 import orientation_graph
    from env.virtual3d import camera_pose
    from env.virtual3d_inspection_v4 import InspectionConfigV4
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    from utils.reconstruction_metrics import ray_scene
    from utils.counterfactual_surface_visibility import reference_visible
    from utils.grid_geometry import inflated_obstacles
    import nso.semantic_completion_v3 as module
    assert Path(module.__file__).resolve().is_relative_to(snapshot)
    manifest=read(source/'artifact_hashes.json'); checked={}
    def checked_path(path):
        path=Path(path); key=str(path.relative_to(source)); value=sha(path)
        assert manifest[key]==value, key
        checked[str(path)]=value
        return path
    summary=read(checked_path(source/'summary.json'))
    config=read(checked_path(source/'config.json')); records=[]
    for history in summary['histories']:
        started=time.monotonic(); fixture=source/history['fixture']
        hdir=(source/history['predictions']).parent
        entry=read(checked_path(fixture/'fixture.json'))
        c=InspectionConfigV4(**entry['environment']); index=history['history_step']
        rows=read(checked_path(fixture/'prefix/records.json'))[:index+1]
        frames=[RGBDFrame.load(checked_path(fixture/'prefix/frames'/f'{i:04d}.npz')) for i in range(index+1)]
        scans=[PlanarScan.load(checked_path(fixture/'prefix/scans'/f'{i:04d}.npz')) for i in range(index+1)]
        shape=(round(c.height_m/c.resolution_m), round(c.width_m/c.resolution_m))
        mapper=remap(frames, scans, rows, c, shape); last=rows[-1]
        obs=mapper.observation(tuple(last['position']), last['heading'], index, last['collision'])
        routes,audit=candidate_routes(mapper, obs, config['branch_actions'], config['max_candidates'])
        original=read(checked_path(hdir/'candidates.json'))
        assert json.loads(json.dumps(routes))==original, 'candidate replay differs'
        assert json.loads(json.dumps(audit))==read(checked_path(hdir/'candidate_audit.json'))
        prediction=read(checked_path(hdir/'predictions.json'))
        model=ObjectCompletionModel(mapper, False); semantic=ObjectCompletionModel(mapper, True)
        policy=QualityCoveragePolicy(grid_config(c), c, 'coverage'); safe=policy._safe(obs)
        strict=~inflated_obstacles(mapper.belief!=0, c.robot_radius_m/c.resolution_m)
        graph,cells,ids=orientation_graph(safe); start=int(ids[obs.position])*4+obs.heading
        outward=dijkstra(graph, directed=True, indices=start)
        backward=dijkstra(graph.T.tocsr(), directed=True, indices=start)
        minimum=outward.reshape(-1,4).min(axis=1)
        distance=np.full(shape,-1.); valid=np.isfinite(minimum)
        distance[tuple(cells[valid].T)]=minimum[valid]
        reachable=safe&(distance>=0)
        retreat,nearest=distance_transform_edt(~reachable, return_indices=True)
        ring=[]; owners={}
        for oi,obj in enumerate(model.objects):
            for radius in (.9,1.4,2.):
                for az in np.arange(0,2*np.pi,np.pi/4):
                    xy=obj['center'][:2]+radius*np.array([np.cos(az),np.sin(az)])
                    r=shape[0]-1-int(np.floor(xy[1]/.2)); col=int(np.floor(xy[0]/.2))
                    rec={'object':oi,'radius_m':radius,'azimuth_rad':float(az),'raw_cell':[r,col]}
                    if not(0<=r<shape[0] and 0<=col<shape[1]): rec['status']='outside_map'
                    elif retreat[r,col]>2: rec.update(status='retreat_over_2cells',retreat_cells=float(retreat[r,col]))
                    else:
                        target=tuple(map(int,nearest[:,r,col]));owners.setdefault(target,set()).add(oi)
                        rec.update(status='accepted',cell=list(target),retreat_cells=float(retreat[r,col]))
                    ring.append(rec)
        all_ring=sorted(owners,key=lambda p:(distance[p],p))
        assert all_ring==model.candidates(safe,distance,limit=10**9)
        ring64=model.candidates(safe,distance,limit=64)
        heads={}; switched=[]
        for cell in ring64:
            xy=camera_pose(cell,0,c,shape[0])[:2,3]
            oi=min(range(len(model.objects)),key=lambda i:np.linalg.norm(model.objects[i]['center'][:2]-xy))
            delta=model.objects[oi]['center'][:2]-xy
            heading=int(np.argmax(np.array([[0,1],[1,0],[0,-1],[-1,0]])@delta))
            heads[cell]=heading
            if oi not in owners[cell]: switched.append({'cell':cell,'original_objects':sorted(owners[cell]),'new_object':oi})
        chosen_poses={tuple(r['pose']) for r in routes}
        path_poses={tuple(s) for r in routes for s in r['states'][1:]}
        frontier_cells=set(policy._candidates(obs,safe,distance))
        observed_mesh=mapper.mesh(); ray=ray_scene(observed_mesh) if len(observed_mesh.triangles) else None
        marked=np.array([obj['marker_supported'] for obj in prediction['objects']['S']])
        point_marked=marked[model.hypothesis_ids//2] if len(marked) else np.zeros(len(model.points),bool)
        candidates=[]
        for ci,cell in enumerate(cells):
            cell=tuple(map(int,cell))
            for heading in range(4):
                sid=ci*4+heading; cost=outward[sid]+backward[sid]
                if not(np.isfinite(cost) and outward[sid]>=1 and cost<=config['branch_actions']): continue
                pose=(*cell,heading); g=model.gain(cell,heading,ray)
                no_observed_occlusion=model.gain(cell,heading,None)
                mask=g>0
                row={'pose':pose,'round_trip_cost':int(cost),
                     'outbound_cost':int(outward[sid]),'ring_raw':cell in owners,
                     'ring64':cell in heads,'ring64_heading':heads.get(cell)==heading,
                     'frontier':cell in frontier_cells,'uniform':cell[0]%4==0 and cell[1]%4==0,
                     'selected_endpoint':pose in chosen_poses,'on_executed_route':pose in path_poses,
                     'G_hidden_m2':float(g.sum()),'S_hidden_m2':float(semantic.weights[mask].sum()),
                     'G_marked_hidden_m2':float(g[point_marked].sum()),
                     'S_marked_hidden_m2':float(semantic.weights[mask&point_marked].sum()),
                     'G_without_observed_occlusion_m2':float(no_observed_occlusion.sum()),
                     'G_by_object_m2':[float(g[model.hypothesis_ids//2==i].sum()) for i in range(len(model.objects))]}
                row['any_original_pool']=row['frontier'] or row['uniform'] or row['ring64_heading']
                candidates.append(row)
        target=output/history['fixture']/hdir.name;target.mkdir(parents=True)
        # Persist sensor-only complete pose inventory BEFORE any GT read.
        write(target/'sensor_only_poses.json', candidates)
        sensor_sha=sha(target/'sensor_only_poses.json')
        write(target/'ring_proposals.json',ring)
        # Evaluator-only static upper bound. No ideal observation is fused or
        # passed back to candidate generation/scoring. No GT-ranked routes run.
        ref=np.load(checked_path(fixture/'reference.npz'))
        seen=np.load(checked_path(hdir/'candidate_000/visibility.npz'))['prefix']
        truth=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(ref['vertices']),o3d.utility.Vector3iVector(ref['triangles']))
        truth_ray=ray_scene(truth); points=ref['points']; weights=ref['weights']; classes=ref['classes']
        for row in candidates:
            r,col,h=row['pose'];pose=camera_pose((r,col),h,c,shape[0])
            ideal=replace(frames[-1],depth_m=np.full_like(frames[-1].depth_m,c.max_depth_m),world_from_camera=pose)
            visible=reference_visible(points,ideal,truth_ray,pose,c.max_depth_m)&~seen
            row['GT_ideal_new_surface_m2']=float(weights[visible].sum())
            row['GT_ideal_new_object_m2']=float(weights[visible&(classes>1)].sum())
        stages={'all_known_safe_budget':lambda r:True,'ring_all_headings':lambda r:r['ring_raw'],
                'ring64_all_headings':lambda r:r['ring64'],'ring64_nearest_heading':lambda r:r['ring64_heading'],
                'any_original_pool':lambda r:r['any_original_pool'],
                'selected_endpoints':lambda r:r['selected_endpoint'],
                'all_executed_path_poses':lambda r:r['on_executed_route']}
        metrics=('G_hidden_m2','S_hidden_m2','G_marked_hidden_m2','S_marked_hidden_m2',
                 'GT_ideal_new_surface_m2','GT_ideal_new_object_m2')
        stage_stats={}
        for name,accept in stages.items():
            subset=[r for r in candidates if accept(r)]
            stage_stats[name]={'poses':len(subset)}
            for metric in metrics:
                best=max(subset,key=lambda r:(r[metric],-r['round_trip_cost']),default=None)
                stage_stats[name][metric]={'positive_poses':sum(r[metric]>1e-8 for r in subset),
                    'max':best[metric] if best else 0.,'best_pose':best['pose'] if best else None}
        outcome=read(checked_path(hdir/'outcomes.json'))
        visibility_loss=[r for r in candidates if r['G_without_observed_occlusion_m2']>1e-8 and r['G_hidden_m2']<=1e-8]
        rec={'fixture':history['fixture'],'step':index,'sigma_m':c.depth_sigma_m,'source_candidate_replay_exact':True,
             'sensor_only_pre_GT_sha256':sensor_sha,'objects':prediction['objects']['S'],
             'ring_counts':{status:sum(r['status']==status for r in ring) for status in ('accepted','outside_map','retreat_over_2cells')},
             'ring_unique_cells':len(all_ring),'ring64_cells':len(ring64),'nearest_reassociation':switched,
             'safe_cells':int(safe.sum()),'reachable_safe_cells':int(reachable.sum()),
             'strict_start_safe':bool(strict[obs.position]),'current_pose_exception_used':not bool(strict[obs.position]),
             'known_free_cells':int((mapper.belief==0).sum()),'stages':stage_stats,
             'observed_mesh_removes_all_hidden_poses':len(visibility_loss),
             'original_pool_audit':audit,'original_selected':prediction['selected'],
             'original_best_actual_area_rate_candidate':max(outcome,key=lambda r:r['area_per_action'])['candidate_id'],
             'elapsed_s':time.monotonic()-started}
        with gzip.open(target/'pose_diagnostics.json.gz','wt') as stream:json.dump(candidates,stream,allow_nan=False)
        write(target/'summary.json',rec);records.append(rec)
        print(history['fixture'],index,'poses',len(candidates),'elapsed',round(rec['elapsed_s'],1),flush=True)
        del mapper,model,semantic,ray,truth_ray
    write(output/'summary.json',{'histories':records,'new_navigation_branches':0,
          'GT_scope':'ideal-valid-depth static single-frame visibility after fixed sensor-only pose enumeration; not reconstruction or executed route utility'})
    assert all(sha(path)==value for path,value in checked.items())
    write(output/'input_hashes.json',checked)


def run(source,output):
    source=source.resolve(); output=output.resolve();meta=read(source/'metadata.json')
    assert meta['status']=='complete' and read(source/'verification.json')['status']=='passed_full'
    assert not output.exists() and not output.is_relative_to(source)
    output.mkdir(parents=True); started=time.monotonic()
    record={'status':'running','workers':1,'source':str(source),'source_archive_sha256':sha(source/'sources.zip'),
            'script_sha256':sha(__file__),'new_navigation_branches':0,'disk_estimate_mib':20,'runtime_estimate_minutes':[2,6]}
    write(output/'metadata.json',record)
    try:
        with tempfile.TemporaryDirectory(prefix='nso_candidate_audit_') as name:
            snapshot=Path(name)
            with zipfile.ZipFile(source/'sources.zip') as archive:
                for member in archive.infolist():
                    target=(snapshot/member.filename).resolve()
                    if not target.is_relative_to(snapshot):raise ValueError('unsafe archive path')
                    if not member.is_dir():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(member))
            for name,value in meta['source_sha256'].items():assert sha(snapshot/name)==value,name
            env=os.environ|{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--source',str(source),
                            '--output',str(output),'--snapshot',str(snapshot)],env=env,cwd=snapshot,check=True)
        record.update(status='complete',elapsed_s=time.monotonic()-started)
    except Exception as error:
        record.update(status='failed',error=repr(error));write(output/'metadata.json',record);raise
    write(output/'metadata.json',record)
    write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.source,args.output,args.snapshot)
    else:run(args.source,args.output)
