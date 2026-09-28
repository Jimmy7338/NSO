#!/usr/bin/env python3
"""Reconstruct fixed checkpoints of eight saved V36 trajectories on CPU.

No World or planner is constructed, and no sensor is queried. Stage predictions
are sealed before reference access. Existing prefix/final results are checked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/nso-article-reconstruction-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'audit_results/article_stage_20260928/reconstruction_checkpoints'
FIGURES=ROOT/'docs/thesis/figures/article_reconstruction_20260928'
SOURCE=ROOT/'audit_results/v36_online_confirmation_20260918'
SCENE=ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json'
STEPS=(18,24,30,36,42)
DISPLAY=(18,30,42)
MAX_BYTES=250*1024**2
MIN_FREE=1024**3
COUNTS=dict(new_worlds=0,new_sensor_queries=0,new_policy_runs=0,
            saved_packet_loads=0,mapper_updates=0,tsdf_integrations=0,
            mesh_extractions=0,offline_surface_evaluations=0)
INPUTS={}
CHECKS=[]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def checked(path, expected=None):
    path=Path(path)
    h=sha(path)
    if expected is not None and h!=expected:
        raise ValueError('Input hash mismatch: '+str(path))
    INPUTS[rel(path)]=dict(sha256=h,bytes=path.stat().st_size,sealed_hash_verified=expected is not None)
    return path


def read(path, expected=None):
    return json.loads(checked(path,expected).read_text())


def size():
    return sum(p.stat().st_size for folder in (BASE,FIGURES) if folder.exists()
               for p in folder.rglob('*') if p.is_file())


def guard(extra=0):
    if size()+extra>MAX_BYTES or shutil.disk_usage(ROOT).free-extra<MIN_FREE:
        raise RuntimeError('Checkpoint output cap or 1 GiB free reserve reached')


def write(path, value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    guard(len(data));path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def arrays(path, **values):
    import numpy as np
    stream=io.BytesIO();np.savez_compressed(stream,**values)
    data=stream.getvalue();guard(len(data));path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as output:output.write(data)


def check(value, message):
    if not value:raise AssertionError(message)
    CHECKS.append(message)


def source_closure():
    paths={Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix=='.py' and '.venv' not in path.parts:
                paths.add(path)
    return {rel(p):sha(p) for p in sorted(paths)}


def import_runtime():
    import numpy as np
    import open3d as o3d
    from env.information_pixel_v34 import InformationConfigV34, _mesh_from_faces, sensor_counts_v34
    from nso.box_union_geometry_v30 import BoxV30,union_exterior_faces_v30
    from nso.decision_replay_v13 import load_packet,array_hash
    from nso.cpu_sensor_contract_v10 import GridTransform
    from nso.sensor_contract_v34 import validate_sensor_packet_v34
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from nso.surface_measurement_v34 import SurfaceMeasurementV34,extract_observed_asset_mesh
    return locals()


def software_versions(runtime):
    versions={}
    for name in ('numpy','scipy','open3d','matplotlib'):
        try:versions[name]=dict(version=importlib.metadata.version(name),source='distribution metadata')
        except importlib.metadata.PackageNotFoundError:
            module=runtime['o3d'] if name=='open3d' else __import__(name)
            versions[name]=dict(version=module.__version__,source='bundled module.__version__; distribution metadata unavailable')
    return versions


def prepare(runtime):
    protocol=read(BASE/'protocol.json')
    check(protocol['paid_action_checkpoints']==list(STEPS) and protocol['cases']==list(range(8)),
          'Predeclared complete eight-by-five checkpoint design')
    check(not (BASE/'freeze.json').exists(),'Exclusive new reconstruction batch')
    config=read(SOURCE/'config.json')
    scene=read(SCENE,'55bc23b57de6e13472970eb1b6670a6e6040db74d3b61e155fe84f9fa08cd7c9')
    check(len(config['physical_cases'])==8,'Eight original physical cases')
    inventory=[]
    for case in config['physical_cases']:
        folder=SOURCE/f"case{case['index']:02d}"
        seal=read(folder/'main_seal.json')
        trace=read(folder/'trace.json',seal['trace.json'])
        read(folder/'result.json',seal['result.json'])
        checked(folder/'evaluation_floor.npz',seal['evaluation_floor.npz'])
        for stage in ('prefix','final'):
            for suffix in ('maps.npz','extracted.npz','crop.json'):
                name=f'{stage}_{suffix}';checked(folder/name,seal[name])
        check(len(trace)==43 and [r['paid'] for r in trace]==list(range(43)),f"case{case['index']:02d}: all saved steps present")
        for step in range(43):
            name=f'packets/{step:03d}.npz'
            checked(folder/name,seal[name])
        inventory.append(dict(case=case,packet_count=43,seal_sha256=sha(folder/'main_seal.json')))
    check(not any(runtime['sensor_counts_v34']().values()),'No simulation calls during runtime import or preflight')
    write(BASE/'freeze.json',dict(schema='article.reconstruction.freeze.v1',protocol_sha256=sha(BASE/'protocol.json'),
        source_sha256=source_closure(),input_sha256=INPUTS,inventory=inventory,
        versions=software_versions(runtime),
        planned_offline_evaluations=40,planned_saved_packet_reintegrations=344,
        new_worlds=0,new_policy_runs=0,exclusive_create=True))
    return config,scene


def mesh_arrays(mesh):
    import numpy as np
    return {name:np.asarray(getattr(mesh,name)).copy() for name in ('vertices','triangles','vertex_colors')}


def mesh_from_arrays(path,runtime):
    np,o3d=runtime['np'],runtime['o3d']
    with np.load(path,allow_pickle=False) as data:
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(data['vertices'])
        mesh.triangles=o3d.utility.Vector3iVector(data['triangles'])
        if len(data['vertex_colors']):mesh.vertex_colors=o3d.utility.Vector3dVector(data['vertex_colors'])
    return mesh


def evaluate_reference(parent,hypothesis,acquisition,runtime):
    np=runtime['np'];Box=runtime['BoxV30']
    shift=np.asarray(acquisition['translation'])
    boxes=[Box(tuple((np.asarray(b).reshape(3,2)+shift[:,None]).ravel()),asset['id'])
           for asset in parent['hypotheses'][hypothesis]['assets'] for b in asset['boxes']]
    full=runtime['_mesh_from_faces'](runtime['union_exterior_faces_v30'](boxes,vertical_only=False))
    vertical=runtime['_mesh_from_faces'](runtime['union_exterior_faces_v30'](boxes,vertical_only=True))
    return runtime['SurfaceMeasurementV34'](full,vertical)


def parity(measurement, old, belief, old_maps, crop, old_crop, label, runtime):
    np=runtime['np']
    with np.load(old_maps,allow_pickle=False) as saved:
        check(np.array_equal(belief,saved['belief']),label+': exact saved occupancy equality')
    check(measurement['prediction_geometry_sha256']==old['prediction_geometry_sha256'],label+': exact canonical cropped geometry SHA')
    check(measurement['reference']==old['reference'],label+': identical reference geometry, areas and sampling seeds')
    check(crop==json.loads(old_crop.read_text()),label+': identical crop audit')
    delta={}
    for field in ('C_map','main_joint'):
        delta[field]=float(measurement[field]-old[field])
    for threshold in ('02cm','05cm','10cm'):
        for field in ('precision','recall','f1','joint'):
            delta[f'{threshold}.{field}']=float(measurement[threshold][field]-old[threshold][field])
    check(all(abs(v)<=1e-10 for v in delta.values()),label+': numeric parity within fixed absolute 1e-10')
    check(all(measurement[k]==old[k] for k in ('eligible','returned','collisions','paid_actions','budget')),label+': qualification parity')
    return dict(passed=True,absolute_tolerance=1e-10,maximum_absolute_difference=max(abs(x) for x in delta.values()),
                differences=delta,exact_occupancy=True,exact_canonical_geometry_sha=True,exact_reference=True)


def run_case(case,config,scene,runtime):
    np=runtime['np'];index=case['index']
    acquisition=config['parents'][case['parent']]
    parent=next(p for p in scene['parents'] if p['id']==case['parent'])
    source=SOURCE/f'case{index:02d}';output=BASE/f'case{index:02d}'
    check(not output.exists(),f'case{index:02d}: exclusive output')
    output.mkdir()
    write(output/'started.json',dict(case=case,source=rel(source),checkpoints=STEPS,
        wall_clock_start_ns=time.time_ns(),new_worlds=0,new_policy_runs=0))
    def timeout(*_):raise TimeoutError('Declared 300-second per-case cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    start=time.monotonic()
    try:
        trace=read(source/'trace.json')
        old=read(source/'result.json')
        c=runtime['InformationConfigV34'](width_m=acquisition['raster_shape'][1]*acquisition['raster_resolution_m'],
            height_m=acquisition['raster_shape'][0]*acquisition['raster_resolution_m'])
        check(c.voxel_m==.04 and c.max_depth_m==4. and c.resolution_m==.2,'Original mapper configuration retained')
        mapper=runtime['ObservedRuntimeMapperV10'](tuple(acquisition['raster_shape']),c,truncation_m=.12)
        transform=runtime['GridTransform'](tuple(acquisition['raster_shape']),.2)
        checkpoints=[];distance=0.;previous=None
        for step in range(43):
            guard()
            packet=runtime['load_packet'](source/'packets'/f'{step:03d}.npz');COUNTS['saved_packet_loads']+=1
            runtime['validate_sensor_packet_v34'](packet,transform,c)
            check(packet.sha256()==trace[step]['packet_sha256'],f'case{index:02d}/{step}: original packet logical identity')
            check(packet.action_id==step and packet.action==trace[step]['action'],f'case{index:02d}/{step}: unchanged executed action')
            check(not np.any(packet.frame.semantic),f'case{index:02d}/{step}: no semantic GT fused')
            position=packet.frame.world_from_camera[:2,3]
            if previous is not None:distance+=float(np.linalg.norm(position-previous))
            previous=position.copy()
            mapper.update(packet.frame,packet.scan)
            COUNTS['mapper_updates']+=1;COUNTS['tsdf_integrations']+=1
            check(runtime['array_hash'](mapper.belief)==trace[step]['measured_map_sha256'],
                  f'case{index:02d}/{step}: every measured map matches the original trace')
            if step not in STEPS:continue
            raw=mapper.mesh();COUNTS['mesh_extractions']+=1
            mesh,crop=runtime['extract_observed_asset_mesh'](raw,acquisition['public_bounds'])
            arrays(output/f'step{step:02d}_mesh.npz',**mesh_arrays(mesh))
            arrays(output/f'step{step:02d}_maps.npz',belief=mapper.belief,camera_seen=mapper.camera_seen,visible=mapper.visible)
            write(output/f'step{step:02d}_crop.json',crop)
            checkpoints.append(dict(step=step,translation_m=distance,
                returned=trace[step]['pose_v33']==trace[0]['pose_v33'],
                collisions=int(trace[step]['collisions_so_far']),source_packet_count=step+1,
                extracted_triangles=len(np.asarray(mesh.triangles))))
        # No future predicted mesh, geometry reference or final quality informs any saved action.
        predicted_files=sorted(p for p in output.iterdir() if p.name.startswith('step'))
        write(output/'prediction_freeze.json',dict(before_reference_access=True,
            files={p.name:sha(p) for p in predicted_files},checkpoints=checkpoints,
            source_trace_sha256=sha(source/'trace.json'),paths_changed=False))
        del mapper
        with np.load(source/'evaluation_floor.npz',allow_pickle=False) as data:
            reachable=data['reachable'].copy()
        evaluator=evaluate_reference(parent,case['hypothesis'],acquisition,runtime)
        stage_rows=[];parities=[]
        for point in checkpoints:
            step=point['step'];mesh=mesh_from_arrays(output/f'step{step:02d}_mesh.npz',runtime)
            with np.load(output/f'step{step:02d}_maps.npz',allow_pickle=False) as data:
                belief=data['belief'].copy()
            coverage=float(np.mean(belief[reachable]!=-1))
            if COUNTS['offline_surface_evaluations']>=40:raise RuntimeError('40-call surface evaluation cap')
            COUNTS['offline_surface_evaluations']+=1
            measured=evaluator.evaluate(mesh,coverage,returned=point['returned'],collisions=point['collisions'],
                failed=False,paid_actions=step,budget=42)
            write(output/f'step{step:02d}_measurement.json',measured)
            if step in (18,42):
                name='prefix' if step==18 else 'final'
                crop=json.loads((output/f'step{step:02d}_crop.json').read_text())
                receipt=parity(measured,old['stages'][name]['measurement'],belief,source/f'{name}_maps.npz',
                    crop,source/f'{name}_crop.json',f'case{index:02d}/{name}',runtime)
                parities.append(dict(step=step,**receipt))
            stage_rows.append(dict(case_index=index,parent=case['parent'],hypothesis=case['hypothesis'],mode=case['mode'],
                paid_actions=step,translation_m=point['translation_m'],saved_frames=point['source_packet_count'],
                C_map=coverage,precision=measured['05cm']['precision'],recall=measured['05cm']['recall'],
                f1=measured['05cm']['f1'],joint=measured['05cm']['joint'],
                returned=point['returned'],eligible_as_terminal=measured['eligible'],
                measurement_scope='saved_trajectory_intermediate_checkpoint' if step<42 else 'saved_trajectory_endpoint_remeasurement',
                mesh_path=rel(output/f'step{step:02d}_mesh.npz'),measurement_path=rel(output/f'step{step:02d}_measurement.json')))
        check(not any(runtime['sensor_counts_v34']().values()),f'case{index:02d}: all simulator counters remain zero')
        result=dict(status='completed',case=case,checkpoints=stage_rows,parities=parities,
            elapsed_s=time.monotonic()-start,new_worlds=0,new_policy_runs=0,
            offline_surface_evaluations=5,source_frames_reintegrated=43)
        write(output/'result.json',result)
        write(output/'seal.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        print(json.dumps({'case':index,'status':'completed','prefix_and_final_parity':True,
            'elapsed_s':result['elapsed_s']}),flush=True)
        return result
    except BaseException:
        write(output/'failure.json',dict(status='failed',traceback=traceback.format_exc(),counts=COUNTS,
            current_sources=source_closure(),automatic_retry=False))
        raise
    finally:signal.alarm(0)


def run():
    if not sys.dont_write_bytecode:raise RuntimeError('Use python -B')
    guard();runtime=import_runtime();config,scene=prepare(runtime)
    results=[]
    try:
        for case in config['physical_cases']:
            results.append(run_case(case,config,scene,runtime))
        freeze=read(BASE/'freeze.json')
        for path,value in freeze['source_sha256'].items():
            check(sha(ROOT/path)==value,'Unchanged reconstruction source '+path)
        for path,value in freeze['input_sha256'].items():
            check(sha(ROOT/path)==value['sha256'],'Unchanged source input '+path)
        check(COUNTS['offline_surface_evaluations']==40 and COUNTS['mapper_updates']==344,'Complete declared 40-stage/344-frame offline work')
        rows=[r for result in results for r in result['checkpoints']]
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        data=buffer.getvalue().encode();guard(len(data));(BASE/'checkpoint_metrics.csv').write_bytes(data)
        write(BASE/'result.json',dict(status='completed',counts=COUNTS,cases=8,checkpoint_count=40,
            original_independent_trajectories=8,new_independent_samples=0,
            parity_checked_stages=16,maximum_parity_absolute_difference=max(p['maximum_absolute_difference'] for x in results for p in x['parities']),
            checks_passed=len(CHECKS),checks=CHECKS,metrics_file='checkpoint_metrics.csv',
            total_case_elapsed_s=sum(r['elapsed_s'] for r in results),protocol_sha256=sha(BASE/'protocol.json')))
        write(BASE/'seal.json',{str(p.relative_to(BASE)):sha(p) for p in sorted(BASE.rglob('*')) if p.is_file()})
    except BaseException:
        write(BASE/'failure.json',dict(status='stopped',counts=COUNTS,completed_cases=len(results),
            traceback=traceback.format_exc(),automatic_retry=False))
        raise


def load_plot_data():
    result=read(BASE/'result.json')
    if result['status']!='completed':raise ValueError('Complete verified offline results required')
    seal=read(BASE/'seal.json')
    for name,value in seal.items():checked(BASE/name,value)
    with (BASE/'checkpoint_metrics.csv').open() as handle:rows=list(csv.DictReader(handle))
    for r in rows:
        for key in ('case_index','hypothesis','paid_actions','saved_frames'):r[key]=int(r[key])
        for key in ('translation_m','C_map','precision','recall','f1','joint'):r[key]=float(r[key])
    return result,rows


def save_figure(fig,name):
    import matplotlib.pyplot as plt
    for extension in ('pdf','svg','png'):
        guard(15*1024**2)
        path=FIGURES/f'{name}.{extension}'
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None}
        fig.savefig(path,dpi=260,facecolor='white',metadata=metadata)
    plt.close(fig)


def plot():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    if FIGURES.exists():raise RuntimeError('Exclusive-create figure output required')
    guard();FIGURES.mkdir(parents=True)
    result,rows=load_plot_data()
    colors={'G':'#0072B2','S':'#D55E00'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.6,
        'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'article-reconstruction-20260928'})
    pairs=[('P00',0),('P00',1),('P01',0),('P01',1)]
    fields=['C_map','precision','recall','f1','joint']
    titles=[r'$C_{\mathrm{map}}$','Precision @ 5 cm','Recall @ 5 cm',r'$F_1$ @ 5 cm',r'$J_5=C_{\mathrm{map}}F_1$']
    def series(p,h,m):return sorted([r for r in rows if (r['parent'],r['hypothesis'],r['mode'])==(p,h,m)],key=lambda r:r['paid_actions'])
    fig,axes=plt.subplots(4,5,figsize=(12.6,9.6))
    fig.subplots_adjust(left=.075,right=.985,top=.90,bottom=.135,hspace=.77,wspace=.24)
    distance_table=[]
    for row,(p,h) in enumerate(pairs):
        for column,field in enumerate(fields):
            ax=axes[row,column]
            for mode in ('S','G'):
                values=series(p,h,mode)
                ax.plot(STEPS,[v[field] for v in values],color=colors[mode],ls='--' if mode=='G' else '-',
                    marker='o' if mode=='S' else 's',mfc='white',ms=3.2,lw=1.35,label=mode,zorder=4 if mode=='G' else 3)
            ax.set(xlim=(17,43),ylim=(0,1.04),xticks=STEPS,yticks=[0,.5,1])
            ax.spines[['top','right']].set_visible(False);ax.grid(color='#E9EEF0',lw=.6)
            ax.tick_params(length=2,labelsize=6.8)
            if row==0:ax.set_title(titles[column],fontsize=9,pad=9)
            if column==0:ax.set_ylabel(f'{p} / h{h}',fontsize=9,fontweight='bold')
            if row==3:ax.set_xlabel('Paid actions',fontsize=7)
        distances=[' / '.join(f"{r['translation_m']:.0f}" for r in series(p,h,m)) for m in ('G','S')]
        position=axes[row,0].get_position()
        y=position.y0-.038
        fig.text(.51,y,f'{p}/h{h}  cumulative distance (m), at actions 18 / 24 / 30 / 36 / 42:   G {distances[0]}   |   S {distances[1]}',
            ha='center',fontsize=7,color='#52606A')
        distance_table.append(dict(parent=p,hypothesis=h,steps=STEPS,G=distances[0],S=distances[1]))
    fig.text(.075,.966,'Coverage and surface quality along the same executed routes',fontsize=14,fontweight='bold',color='#22333D')
    fig.text(.075,.938,'40 fixed offline checkpoints · 8 original trajectories · identical TSDF and surface evaluator · no new policy runs',fontsize=8.5,color='#52606A')
    handles=[Line2D([],[],color=colors[m],ls='--' if m=='G' else '-',marker='s' if m=='G' else 'o',mfc='white',label=m) for m in ('G','S')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.043),ncols=2,frameon=False,fontsize=8)
    fig.text(.075,.016,'Markers are actual remeasured checkpoints; connecting segments are visual guides. Intermediate scores do not imply terminal qualification.\n'
        'Both t=18 and t=42 reproduce the original saved maps, canonical surface geometry and numerical scores within the declared tolerance.',fontsize=7,color='#52606A')
    save_figure(fig,'checkpoint_quality_components')

    # Actual extracted TSDF only. All four pairs and three declared display times.
    scene=read(SCENE);parents={p['id']:p for p in scene['parents']};config=read(SOURCE/'config.json')
    fig,axes=plt.subplots(4,6,figsize=(13.8,10.6))
    fig.subplots_adjust(left=.045,right=.99,top=.91,bottom=.095,wspace=.055,hspace=.30)
    mesh_display=[]
    for row,(p,h) in enumerate(pairs):
        parent=parents[p]
        azimuth=np.deg2rad(55 if h else 125);elevation=np.deg2rad(24)
        eye=np.array([np.cos(azimuth)*np.cos(elevation),np.sin(azimuth)*np.cos(elevation),np.sin(elevation)])
        horizontal=np.array([-np.sin(azimuth),np.cos(azimuth),0]);vertical=np.cross(eye,horizontal)
        for stage_index,step in enumerate(DISPLAY):
            for mi,mode in enumerate(('G','S')):
                ax=axes[row,stage_index*2+mi]
                measured=next(r for r in rows if (r['parent'],r['hypothesis'],r['mode'],r['paid_actions'])==(p,h,mode,step))
                path=ROOT/measured['mesh_path']
                with np.load(path,allow_pickle=False) as data:
                    vertices=data['vertices'].copy()-np.asarray(config['parents'][p]['translation'])
                    triangles=data['triangles'].copy()
                vertices-=np.asarray(parent['device_frame']['origin_xyz'])
                for _ in range(parent['device_frame']['quarter_turns_ccw']):
                    x,y=vertices[:,0].copy(),vertices[:,1].copy();vertices[:,0],vertices[:,1]=y,-x
                centered=vertices-np.array([0,2.6,.9])
                projection=np.column_stack([centered@horizontal,centered@vertical,centered@eye])[triangles]
                order=np.argsort(projection[:,:,2].mean(axis=1),kind='stable')
                facets=vertices[triangles];normals=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0])
                lengths=np.linalg.norm(normals,axis=1)
                normals=np.divide(normals,lengths[:,None],out=np.zeros_like(normals),where=lengths[:,None]>0)
                light=np.array([.45 if h else -.45,.6,1.]);light/=np.linalg.norm(light)
                shade=.60+.40*np.abs(normals@light)
                surface_colors=np.clip(np.array([.404,.506,.557])[None,:]*shade[:,None]+.10,0,1)
                ax.add_collection(PolyCollection(projection[order,:,:2],facecolors=surface_colors[order],edgecolors='none',
                    linewidth=0,antialiased=False,rasterized=True))
                ax.set(xlim=(-3.9,3.9),ylim=(-2.15,2.15),aspect='equal');ax.set_axis_off()
                ax.set_title(f'{p}/h{h} · {mode} · t={step}',loc='left',fontsize=7.5,fontweight='bold',color=colors[mode],pad=4)
                ax.text(.02,.015,r'$F_1$'+f" {measured['f1']:.3f}  |  "+r'$J_5$'+f" {measured['joint']:.3f}",
                    transform=ax.transAxes,fontsize=6.5,color='#40515B')
                ax.plot([-3.65,-2.65],[-1.8,-1.8],color='#40515B',lw=1)
                ax.text(-3.15,-1.63,'1 m',ha='center',fontsize=6)
                mesh_display.append(dict(parent=p,hypothesis=h,method=mode,paid_actions=step,
                    mesh_path=measured['mesh_path'],mesh_sha256=sha(path),triangles_drawn=len(triangles),
                    projection='orthographic',azimuth_deg=55 if h else 125,elevation_deg=24,
                    limits=[[-3.9,3.9],[-2.15,2.15]],all_saved_triangles=True))
    fig.text(.045,.969,'Actual TSDF surfaces at three declared checkpoints',fontsize=14,fontweight='bold',color='#22333D')
    fig.text(.045,.941,'All four G/S conditions · same camera and metric scale within each pair across time · no completion or smoothing',fontsize=8.5,color='#52606A')
    for stage_index,step in enumerate(DISPLAY):
        left=axes[0,2*stage_index].get_position().x0;right=axes[0,2*stage_index+1].get_position().x1
        fig.text((left+right)/2,.915,f'{step} paid actions',ha='center',fontsize=9,fontweight='bold')
    fig.text(.045,.042,'Every panel projects the saved extracted TSDF mesh. The full four-pair display retains coincident h0 results.\n'
        'Neutral shading indicates illumination, not geometric error; geometry was never completed with a template. Intermediate stages are not new tasks.',fontsize=7,color='#52606A')
    save_figure(fig,'checkpoint_actual_meshes')
    captions=dict(
        checkpoint_quality_components=dict(
            zh='对原V36全部八条保存轨迹在18、24、30、36、42付费动作处重融合与离线评价。五列分别为实际二维覆盖、5cm表面精确率、召回率、F1和乘积J5；四行保留两布局双构型全部配对。横轴为原轨迹付费动作，下方列出各检查点真实累计平移距离。圆点/方点为实际离线测量，连接线仅帮助阅读，未插值生成其他时刻分数。中间状态不作为完成且合格任务；18/42步与原保存结果分别核对。40个检查点来自八条旧轨迹，不增加独立样本。',
            en='Offline reintegration and measurement of all eight saved V36 trajectories at 18, 24, 30, 36 and 42 paid actions. Columns separate actual planar coverage, 5 cm precision, recall, F1 and joint J5; rows retain all four paired conditions. The action axis is accompanied by actual cumulative translation distances. Markers are measured checkpoints and connecting segments only visual guides. Intermediate stages are not successful terminal episodes. Prefix and final stages are checked against original saved outputs; 40 checkpoints do not constitute additional independent trajectories.'),
        checkpoint_actual_meshes=dict(
            zh='固定显示18、30、42步的实际重融合TSDF，共四条件、每条件G/S配对。所有面板显示全部已保存提取网格三角面；同一构型各方法和时刻的相机、米制尺度、材质与光照相同。h0/h1分别使用后左/后右预定义视角；未补洞、平滑或加入模板表面。F1、J5由本轮离线评价直接读取，明暗只是光照。两个h0持平条件完整保留。',
            en='Actual reintegrated TSDF meshes at the fixed display checkpoints 18, 30 and 42 for all four G/S pairs. All extracted triangles are rendered; each configuration shares one camera, metric scale, material and lighting across methods and time. The h0/h1 views use predefined rear-left/rear-right cameras. No hole completion, smoothing or template surface is added. F1 and J5 come from the offline measurements; shading is illumination, not error. Both coincident h0 conditions remain visible.'))
    write(FIGURES/'captions.json',captions)
    write(FIGURES/'display_geometry.json',dict(meshes=mesh_display,distance_table=distance_table))
    write(FIGURES/'manifest.json',dict(schema='article.reconstruction.figures.v1',source_result_sha256=sha(BASE/'result.json'),
        source_seal_sha256=sha(BASE/'seal.json'),source_script_sha256=sha(__file__),
        files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(FIGURES.iterdir()) if p.is_file()},
        new_worlds=0,new_policy_runs=0,new_measurements_in_plotting=0,original_trajectories=8,offline_checkpoints=40,
        all_pairs_shown=True,mesh_smoothing=False,mesh_completion=False,mesh_decimation=False,
        maximum_combined_bytes=MAX_BYTES,combined_bytes_before_manifest=size()))
    print(json.dumps(dict(status='figures_completed',output=rel(FIGURES),combined_bytes=size())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','plot'))
    arguments=parser.parse_args()
    run() if arguments.command=='run' else plot()
