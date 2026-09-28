#!/usr/bin/env python3
"""Render saved, sealed scene meshes from an explicit comparison snapshot.

No simulator, policy, fusion, surface-distance calculation or quality evaluation
is imported or run. All scene triangles are rendered; local crops are solely
for display and never replace the complete prediction used by the evaluator.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-article-scene-meshes-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET_PIN = 'a0319f7e5683a91a56effdbbab28c90e226ce5b0ac7178c2b2408cb8954ae14f'
METHODS = ('NBV', 'G', 'B', 'S')
COLORS = {'NBV': '#778493', 'G': '#4477AA', 'B': '#BB8822', 'S': '#228833'}
INK, MUTED = '#243A46', '#5F6D76'
ELEVATION, AZIMUTH = 53., -63.
HEIGHT_RANGE = (-.1, 2.4)
CMAP = LinearSegmentedColormap.from_list('height', ['#DBE3E7', '#77ACAF', '#2B7887', '#264A72'])
NORM = Normalize(*HEIGHT_RANGE)
SOURCES, DISPLAY = {}, []
MAX_OUTPUT = 35 * 1024**2


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def source(path):
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError('regular input file without links required: ' + str(path))
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    SOURCES[key] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return path


def read(path):
    return json.loads(source(path).read_text())


def child(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('plain relative input member required')
    return root / p


def verify_manifest(root):
    manifest = read(root/'manifest.json')
    for name, expected in manifest['files'].items():
        path = source(child(root, name))
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('input manifest mismatch: ' + str(path))
    return manifest


def mesh(path):
    with np.load(source(path), allow_pickle=False) as packed:
        vertices = np.array(packed['vertices'], dtype=float)
        faces = np.array(packed['triangles'], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('expected triangular three-dimensional mesh')
    if not np.isfinite(vertices).all() or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices))):
        raise ValueError('invalid finite mesh geometry')
    return dict(vertices=vertices, faces=faces, triangles=vertices[faces], path=str(path))


def load_inputs(args):
    snapshot = verify_manifest(args.analysis)
    slots = read(args.analysis/'slots.json')
    if not slots or {r['phase'] for r in slots} != {args.phase}:
        raise ValueError('explicit single phase must match every declared snapshot slot')
    if any(r['method'] not in METHODS for r in slots):
        raise ValueError('targeted ablations require a separate figure')
    keys = [(r['scene_id'], r['budget'], r['noise_seed'], r['method']) for r in slots]
    if len(keys) != len(set(keys)):
        raise ValueError('ambiguous duplicate declared condition')
    groups = sorted({key[:3] for key in keys})
    if len(groups) > 6:
        raise ValueError('at most six declared matched conditions per figure package')
    if any({r['method'] for r in slots if (r['scene_id'],r['budget'],r['noise_seed']) == g} != set(METHODS) for g in groups):
        raise ValueError('all four methods, including pending slots, must be declared')
    assets = read(args.assets/'manifest.json')
    if sha(args.assets/'manifest.json') != ASSET_PIN:
        raise ValueError('frozen static asset inventory mismatch')
    scenes, runs = {}, {}
    for scene_id, _, _ in groups:
        if scene_id in scenes:
            continue
        folder = child(args.assets, scene_id)
        for name in ('renderer_private/geometry.npz', 'evaluation_private/instances.json', 'public_workspace.json'):
            path = source(folder/name)
            if sha(path) != assets['artifact_sha256'][scene_id+'/'+name]:
                raise ValueError('frozen scene asset mismatch')
        scenes[scene_id] = dict(gt=mesh(folder/'renderer_private/geometry.npz'),
            instances=read(folder/'evaluation_private/instances.json')['private_instances'],
            workspace=read(folder/'public_workspace.json'))
    for slot in slots:
        if slot['status'] != 'reviewed_qualified':
            continue  # Never use later completion to silently upgrade this snapshot.
        run_id = slot['run_id']; review_dir = child(args.reviews, run_id)
        verify_manifest(review_dir); review = read(review_dir/'review.json')
        if (review.get('all_checks_passed') is not True or review.get('qualified') is not True
                or review['run_id'] != run_id or review['method'] != slot['method']
                or review['scene_id'] != slot['scene_id'] or review['protocol_sha256'] != slot['protocol_sha256']):
            raise ValueError('matching independently reviewed qualified episode required')
        for key, measured in [('C_nav','C_nav'), ('F1','Q'), ('J_nav','J_nav')]:
            if slot[key] != review['metrics'][measured]:
                raise ValueError('snapshot metric differs from its independent review')
        episode = child(args.episodes, run_id)
        original = read(episode/'artifact_manifest.json')
        if sha(episode/'artifact_manifest.json') != review['input_manifest_sha256']:
            raise ValueError('episode manifest differs from reviewed original')
        source_key = str((episode/'artifact_manifest.json').relative_to(ROOT)) if episode.is_relative_to(ROOT) else str(episode/'artifact_manifest.json')
        if source_key not in snapshot['sources'] or snapshot['sources'][source_key]['sha256'] != review['input_manifest_sha256']:
            raise ValueError('snapshot does not bind this episode manifest')
        for name in ('prediction/mesh.npz', 'prediction_seal.json', 'result.json', 'evaluation.json'):
            path = source(episode/name); row = original['files'][name]
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('sealed figure artifact changed: ' + name)
        with source(review_dir/'trajectory.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != slot['paid_actions']+1 or [int(r['paid_step']) for r in rows] != list(range(len(rows))):
            raise ValueError('noncontinuous reviewed route')
        xy = np.array([[float(r['x_m']), float(r['y_m'])] for r in rows])
        runs[run_id] = dict(mesh=mesh(episode/'prediction/mesh.npz'), review=review, slot=slot, xy=xy, rows=rows)
    return slots, groups, scenes, runs


def geometry_equal(a, b):
    return np.array_equal(a['vertices'], b['vertices']) and np.array_equal(a['faces'], b['faces'])


def face_colors(triangles):
    normal = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    length = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(length[:,None], 1e-30)
    lamp = np.array([-.4, -.7, 1.]); lamp /= np.linalg.norm(lamp)
    light = .62 + .38*np.abs(normal @ lamp)
    rgba = CMAP(NORM(triangles.mean(axis=1)[:,2])); rgba[:,:3] *= light[:,None]
    return rgba


def configure(ax, bounds, *, labels=True):
    ax.set_proj_type('ortho'); ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_xlim(bounds[0,0], bounds[1,0]); ax.set_ylim(bounds[0,1], bounds[1,1]); ax.set_zlim(bounds[0,2], bounds[1,2])
    ax.set_box_aspect(bounds[1]-bounds[0], zoom=1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor('#E2E8EB')
        axis._axinfo['grid'].update(color='#E0E6E9', linewidth=.45)
        axis.line.set_color('#BDCAD0')
    ax.tick_params(labelsize=6.4, pad=-1, length=2, colors=MUTED)
    if labels:
        ax.set_xlabel('x (m)', fontsize=7, labelpad=-2); ax.set_ylabel('y (m)', fontsize=7, labelpad=-2)
        ax.set_zlabel('z (m)', fontsize=7, labelpad=-3)
        ax.set_xticks([0,4,8]);ax.set_yticks([0,4,7.5]);ax.set_zticks([0,1.2,2.4])
    else:
        ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])


def draw_mesh(ax, item, bounds, identity, *, crop=None):
    triangles = item['triangles']
    if crop is not None:
        # Exact existing faces only: no invented cap, interpolation or geometry repair.
        keep = (triangles.min(axis=1) >= crop[0]-1e-12).all(axis=1) & (triangles.max(axis=1) <= crop[1]+1e-12).all(axis=1)
        triangles = triangles[keep]
    DISPLAY.append(dict(panel=identity, original_triangles=len(item['triangles']), drawn_triangles=len(triangles),
        display_only_crop=crop.tolist() if crop is not None else None, decimation=False, rasterized=True))
    if len(triangles):
        poly = Poly3DCollection(triangles, facecolors=face_colors(triangles), edgecolors='none',
            linewidths=0, antialiased=False, zsort='average', rasterized=True)
        ax.add_collection3d(poly)
    configure(ax, bounds, labels=crop is None)


def route(ax, run):
    xy = run['xy']; color = COLORS[run['slot']['method']]
    ax.plot(xy[:,0], xy[:,1], np.full(len(xy), .035), color='white', lw=2.7, zorder=30)
    ax.plot(xy[:,0], xy[:,1], np.full(len(xy), .035), color=color, lw=1.35, zorder=31)
    ax.scatter(xy[0,0], xy[0,1], .035, s=25, marker='D', c='white', edgecolors=INK, lw=.9, depthshade=False, zorder=32)


def pending(ax, status, bounds, *, local=False):
    configure(ax, bounds, labels=not local)
    ax.text2D(.5,.53,status.replace('_',' ').upper(), transform=ax.transAxes,
        ha='center', va='center', color=MUTED, fontsize=9, weight='bold')
    ax.text2D(.5,.42,'No qualified mesh in this snapshot', transform=ax.transAxes,
        ha='center', color=MUTED, fontsize=6.7)


def save(fig, output, name, exports):
    for ext in ('png','pdf','svg'):
        p=output/f'{name}.{ext}'
        metadata=({'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None)
        fig.savefig(p,dpi=220,facecolor='white',metadata=metadata,bbox_inches='tight',pad_inches=.09)
        exports.append(p.name)
        if sum(f.stat().st_size for f in output.iterdir() if f.is_file()) > MAX_OUTPUT:
            raise ValueError('figure package exceeds fixed 35 MiB output bound')
    plt.close(fig)


def colorbar(fig, rect):
    ax=fig.add_axes(rect)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=ax,orientation='horizontal')
    bar.set_label('Face-centroid height (m); not error',fontsize=7,color=MUTED,labelpad=2)
    bar.set_ticks([0,1.2,2.4]);bar.ax.tick_params(labelsize=6.5,length=2)
    bar.outline.set_visible(False)


def all_scenes(slots, groups, scenes, runs, bounds, output, exports):
    fig=plt.figure(figsize=(15.2,3.35*len(groups)+.8))
    fig.subplots_adjust(left=.012,right=.987,top=.93,bottom=.072,wspace=.015,hspace=.17)
    fig.text(.035,.975,'Actual scene reconstructions and executed routes',fontsize=15,weight='bold',color=INK)
    for row,group in enumerate(groups):
        scene_id,budget,noise=group
        for col,method in enumerate(('GT',)+METHODS):
            ax=fig.add_subplot(len(groups),5,row*5+col+1,projection='3d',computed_zorder=False)
            if method=='GT':
                draw_mesh(ax,scenes[scene_id]['gt'],bounds,f'{scene_id}/GT/full')
                title='Ground truth'; detail=f'{scene_id.replace("ART1_","")} | B{budget}, n{noise}'
            else:
                slot=next(r for r in slots if (r['scene_id'],r['budget'],r['noise_seed'],r['method'])==(*group,method))
                title=method
                if slot['run_id'] in runs:
                    run=runs[slot['run_id']];draw_mesh(ax,run['mesh'],bounds,f'{slot["run_id"]}/full');route(ax,run)
                    detail=f'F1 {slot["F1"]:.3f} | J {slot["J_nav"]:.3f} | {slot["path_length_m"]:.1f} m'
                else:
                    pending(ax,slot['status'],bounds);detail='Endpoint unavailable'
            ax.set_title(title,fontsize=11,color=COLORS.get(method,INK),weight='bold',pad=3)
            ax.text2D(.5,-.04,detail,transform=ax.transAxes,ha='center',fontsize=7.1,color=MUTED)
    fig.text(.035,.025,'All saved scene triangles; identical world scale and camera. Routes project onto the floor and overlay occlusion.\n'
        'GT and local crops are offline figure annotations. Pending slots are retained; no missing score is filled with zero.',fontsize=7.6,color=MUTED)
    colorbar(fig,[.715,.04,.23,.01])
    save(fig,output,'scene_meshes_all_declared',exports)


def preview(slots, groups, scenes, runs, bounds, output, exports):
    group=next((g for g in groups if g[0]=='ART1_AISLE_DEV'),None)
    if group is None:return
    matching={r['method']:r for r in slots if (r['scene_id'],r['budget'],r['noise_seed'])==group}
    fig=plt.figure(figsize=(12.8,5.0));fig.subplots_adjust(left=.018,right=.98,top=.85,bottom=.15,wspace=.04)
    fig.text(.04,.965,'Aisle mapping: sealed 3D endpoints',fontsize=16,color=INK,weight='bold')
    fig.text(.04,.916,f'{group[0]} | {group[1]} paid actions | noise seed {group[2]} | development snapshot',fontsize=9,color=MUTED)
    for index,method in enumerate(('GT','G','S')):
        ax=fig.add_subplot(1,3,index+1,projection='3d',computed_zorder=False)
        if method=='GT':
            draw_mesh(ax,scenes[group[0]]['gt'],bounds,'preview/GT');detail='Offline geometry; no executed route'
        elif matching[method]['run_id'] in runs:
            run=runs[matching[method]['run_id']];draw_mesh(ax,run['mesh'],bounds,f'preview/{method}');route(ax,run)
            detail=f'F1 {run["slot"]["F1"]:.3f} | {run["slot"]["path_length_m"]:.1f} m'
        else:
            pending(ax,matching[method]['status'],bounds);detail='No qualified endpoint'
        ax.set_title(f'({chr(97+index)}) '+('Ground truth' if method=='GT' else method),loc='left',fontsize=12,color=COLORS.get(method,INK),weight='bold')
        ax.text2D(.5,-.025,detail,ha='center',transform=ax.transAxes,fontsize=8,color=MUTED)
    equal=False
    if all(matching[m]['run_id'] in runs for m in ('G','S')):
        g,s=(runs[matching[m]['run_id']] for m in ('G','S'))
        equal=geometry_equal(g['mesh'],s['mesh']) and np.array_equal(g['xy'],s['xy'])
    note=('G and S: exactly identical saved vertices, triangles and XY routes.\nThis development endpoint does not establish a semantic advantage.' if equal else
        'Saved, independently reviewed endpoints from this declared condition.\nMissing measurements are not estimated.')
    fig.text(.04,.062,note,fontsize=8,color=INK,linespacing=1.4)
    fig.text(.04,.018,'Full meshes, fixed view and scale. Route overlays the floor; diamond: start / return. GT also includes unobservable surfaces.',fontsize=7.2,color=MUTED)
    colorbar(fig,[.77,.087,.19,.013]);save(fig,output,'aisle_gt_g_s_preview',exports)
    return equal


def closeups(slots,groups,scenes,runs,output,exports):
    for group in groups:
        selected=[r for r in slots if (r['scene_id'],r['budget'],r['noise_seed'])==group]
        if not any(r['run_id'] in runs for r in selected):continue
        instances=scenes[group[0]]['instances']
        boxes=[np.array(i['world_aabb_m'],dtype=float)+np.array([[-.18,-.18,-.18],[.18,.18,.18]]) for i in instances]
        extent=np.max([b[1]-b[0] for b in boxes],axis=0)
        fig=plt.figure(figsize=(13.8,3.*len(instances)+.6));fig.subplots_adjust(left=.045,right=.987,top=.925,bottom=.06,wspace=.01,hspace=.08)
        fig.text(.045,.975,f'{group[0].replace("ART1_","")}: every declared equipment instance',fontsize=15,color=INK,weight='bold')
        fig.text(.045,.945,'Identical camera and physical scale; fixed ground-truth boxes + 0.18 m are used for display only.',fontsize=8.5,color=MUTED)
        for row,(instance,crop) in enumerate(zip(instances,boxes)):
            center=crop.mean(axis=0);bounds=np.array([center-extent/2,center+extent/2])
            for col,method in enumerate(('GT',)+METHODS):
                ax=fig.add_subplot(len(instances),5,row*5+col+1,projection='3d',computed_zorder=False)
                if method=='GT':
                    draw_mesh(ax,scenes[group[0]]['gt'],bounds,f'{group[0]}/I{row+1}/GT',crop=crop)
                    detail=instance['category'].capitalize()
                else:
                    slot=next(r for r in selected if r['method']==method)
                    if slot['run_id'] in runs:
                        run=runs[slot['run_id']];draw_mesh(ax,run['mesh'],bounds,f'{slot["run_id"]}/I{row+1}',crop=crop)
                        result=next(r for r in run['review']['metrics']['per_instance'] if r['instance_id']==instance['instance_id'])
                        detail=f'P {result["precision"]:.3f} | R {result["recall"]:.3f} | F1 {result["f1"]:.3f}'
                    else:
                        pending(ax,slot['status'],bounds,local=True);detail='No qualified measurement'
                if row==0:ax.set_title('Ground truth' if method=='GT' else method,fontsize=11,weight='bold',color=COLORS.get(method,INK),pad=0)
                if col==0:ax.text2D(-.08,.5,f'I{instance["instance_id"]+1}',transform=ax.transAxes,fontsize=11,color=INK,weight='bold')
                ax.text2D(.5,.01,detail,ha='center',transform=ax.transAxes,fontsize=7.4,color=MUTED)
        fig.text(.045,.025,'Original triangles inside fixed display boxes only; complete predictions are used for scoring.\n'
            'P/R/F1 use the fixed observable reference. GT also includes unobservable surfaces; all four instances are retained.',fontsize=7.3,color=MUTED)
        colorbar(fig,[.80,.038,.17,.008])
        save(fig,output,f'{group[0].lower()}_b{group[1]}_n{group[2]}_all_instance_closeups',exports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','episodes','reviews','assets','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=('development','main'),required=True)
    args=parser.parse_args()
    for name in ('analysis','episodes','reviews','assets','output'):setattr(args,name,getattr(args,name).absolute())
    if args.output.exists():raise ValueError('new exclusive figure snapshot directory required')
    slots,groups,scenes,runs=load_inputs(args)
    source(__file__)
    vertices=[s['gt']['vertices'] for s in scenes.values()]+[r['mesh']['vertices'] for r in runs.values()]
    bounds=np.array([np.min([v.min(axis=0) for v in vertices if len(v)],axis=0),np.max([v.max(axis=0) for v in vertices if len(v)],axis=0)])
    bounds+=np.array([[-.1,-.1,-.05],[.1,.1,.1]])
    args.output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    exports=[]
    equal=preview(slots,groups,scenes,runs,bounds,args.output,exports)
    all_scenes(slots,groups,scenes,runs,bounds,args.output,exports)
    closeups(slots,groups,scenes,runs,args.output,exports)
    captions=dict(
        full_en='Full saved TSDF meshes and offline ground truth at the same world limits, orthographic view and physical scale. All scene triangles are rasterized without decimation. Executed XY routes are projected onto the floor and drawn over occlusion for legibility; diamonds mark start and return. Missing or unqualified endpoints remain explicitly labeled according to the supplied immutable comparison snapshot. Colors encode triangle-centroid height with geometric normal illumination, not measured errors. GT includes unobservable surfaces; metrics use the fixed observable reference.',
        full_zh='同一世界坐标范围、正交视角与物理尺度下的完整保存TSDF网格和离线真值。所有场景三角面均栅格化绘制，不抽稀。实际XY路线投影到地面并覆盖遮挡以便阅读，菱形表示起点和返航点。缺失或不合格终点按指定快照保留状态。颜色表示三角面中心高度并叠加真实面法向光照，不表示实测误差。GT包含不可观测表面，指标使用固定可观测参考。',
        closeups_en='Every declared instance is shown using its offline GT bounding box expanded by 0.18 m; only original triangles wholly contained in the display box are drawn. View, scale and shading are shared across methods. P/R/F1 are copied from the independent full-scene evaluation against the fixed observable reference. GT also includes unobservable surfaces; display crops do not enter scoring, training or planning.',
        closeups_zh='全部预声明实例按离线真值包围盒外扩0.18米展示，仅绘制完整位于展示框内的原三角面。各方法共享视角、尺度和配色。P/R/F1复制自针对固定可观测参考的独立全场景评价。GT也包含不可观测表面；局部展示裁剪不参与评分、训练或规划。',
        preview_en=('The saved AISLE G and S vertex arrays, triangle indices and XY routes are exactly identical. This panel demonstrates the reconstruction pipeline and does not establish a semantic advantage.' if equal else 'The AISLE panel displays qualified endpoints from the explicit snapshot; no missing endpoint is estimated.'),
        preview_zh=('AISLE中G和S保存的顶点数组、三角面索引及XY路线完全相同。此图展示重建流程，不构成语义优势证据。' if equal else 'AISLE图展示指定快照中的合格终点；缺失终点不作估计。'))
    data=dict(schema='article.mesh_display_geometry.v1',world_bounds=bounds.tolist(),camera=dict(elevation_deg=ELEVATION,azimuth_deg=AZIMUTH,projection='orthographic'),height_range_m=HEIGHT_RANGE,
        display_decimation=False,display_crop_margin_m=.18,aisle_g_s_mesh_and_xy_exactly_equal=equal,panels=DISPLAY,
        slots=[dict(run_id=r['run_id'],method=r['method'],scene_id=r['scene_id'],status=r['status']) for r in slots])
    for name,value in [('captions.json',captions),('display_geometry.json',data)]:
        (args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');exports.append(name)
    (args.output/'generator_source.py').write_bytes(Path(__file__).read_bytes());exports.append('generator_source.py')
    manifest=dict(schema='article.scene_mesh_figures.v1',phase=args.phase,analysis_snapshot=str(args.analysis),
        declared_slots=len(slots),qualified_displayed=len(runs),qualified_run_ids=sorted(runs),all_declared_slots_retained=True,
        new_worlds=0,new_policy_runs=0,new_sensor_queries=0,new_tsdf_integrations=0,new_surface_evaluations=0,
        sources=SOURCES,files={name:dict(bytes=(args.output/name).stat().st_size,sha256=sha(args.output/name)) for name in exports})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(args.output),qualified=len(runs),slots=len(slots),files=len(exports),aisle_g_s_identical=equal,output_bytes=sum(p.stat().st_size for p in args.output.iterdir()))))


if __name__=='__main__':main()
