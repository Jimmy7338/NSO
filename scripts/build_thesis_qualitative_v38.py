#!/usr/bin/env python3
"""Render saved V36 geometry/trajectories/meshes; never run an experiment."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-v38-qualitative-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/thesis/figures/v38_qualitative'
BASE = 'audit_results/v36_online_confirmation_20260918'
SCENE = 'configs/virtual3d/v33_direction_scene_r1_20260917.json'
RESERVE = 64 * 1024 * 1024
LIMIT = 2 * 1024 * 1024
COLORS = {'G': '#777D82', 'S': '#0072B2'}
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(relative):
    path = ROOT / relative
    SOURCES[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read(relative):
    return json.loads(source(relative).read_text())


def checked_space():
    if shutil.disk_usage(ROOT).free < RESERVE + LIMIT:
        raise RuntimeError('64 MiB reserve plus bounded 2 MiB output required')


def main():
    checked_space()
    OUT.mkdir(parents=True, exist_ok=True)
    config = read(BASE + '/config.json')['parents']['P00']
    scene = read(SCENE)
    parent = next(p for p in scene['parents'] if p['id'] == 'P00')
    boxes = parent['hypotheses'][1]['assets'][0]['boxes']
    assert config['forced_prefix'] == 18 and config['budget'] == 42
    shift = np.asarray(config['translation'])
    records = {}
    for mode, index in [('G', 2), ('S', 3)]:
        folder = f'{BASE}/case{index:02d}'
        trace = read(folder + '/trace.json')
        result = read(folder + '/result.json')
        crop = read(folder + '/final_crop.json')
        path = source(folder + '/final_extracted.npz')
        with np.load(path, allow_pickle=False) as arrays:
            vertices = arrays['vertices'].copy() - shift
            triangles = arrays['triangles'].copy()
        assert result['mode'] == mode and result['physical_case']['hypothesis'] == 1
        assert result['paid_actions'] == 42 and len(trace) == 43
        assert triangles.shape[0] == crop['kept_triangles']
        assert np.isfinite(vertices).all()
        assert trace[18]['pose_v33'] == trace[42]['pose_v33'] == [0, 0, 0]
        records[mode] = dict(trace=trace, result=result, crop=crop,
            vertices=vertices, triangles=triangles, source_folder=folder)

    font = findfont(FontProperties(family='Liberation Sans'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.5,
        'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': .6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'nso-v38-qualitative', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(7.05, 4.45), facecolor='white')
    ax = fig.add_axes([.075, .28, .32, .57])
    # Frozen actual geometry is shown only as offline context, not fed to planning.
    for bounds in sorted(boxes, key=lambda b: b[5]):
        x0, x1, y0, y1, z0, z1 = bounds
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
            facecolor='#E8EAEB' if z1 < 1 else '#CED3D6',
            edgecolor='#8E969B', linewidth=.55, zorder=1))
    low, high = np.asarray(config['public_bounds']) - shift
    padding = records['G']['crop']['padding_m']
    ax.add_patch(Rectangle(low[:2]-padding, *(high[:2]-low[:2]+2*padding),
        fill=False, edgecolor='#AAB0B3', linewidth=.65, linestyle=(0,(3,2)), zorder=2))
    prefix = np.asarray([r['pose_v33'][:2] for r in records['G']['trace'][:19]])
    assert np.array_equal(prefix, [r['pose_v33'][:2] for r in records['S']['trace'][:19]])
    ax.plot(prefix[:,0], prefix[:,1], color='#DDE0E2', lw=5, solid_capstyle='round', zorder=3)
    for mode, record in records.items():
        points = np.asarray([r['pose_v33'][:2] for r in record['trace'][18:]])
        ax.plot(points[:,0], points[:,1], color=COLORS[mode], lw=1.6,
            linestyle='--' if mode=='G' else '-', zorder=4)
        x = -3 if mode == 'G' else 3
        ax.annotate('', xy=(x,2.7), xytext=(x,1.7),
            arrowprops=dict(arrowstyle='-|>', lw=1.5, color=COLORS[mode]), zorder=5)
    ax.scatter([0],[0],s=24,c='#25292C',marker='D',zorder=6)
    ax.scatter([0],[1.1],s=22,c='#B37828',marker='s',edgecolors='white',linewidth=.5,zorder=6)
    ax.text(0, -.40, 'Anchor', ha='center', va='top', fontsize=8)
    ax.annotate('RGB cue', xy=(0,1.1), xytext=(.05,.68), ha='center', va='top',
        fontsize=8, color='#735018', arrowprops=dict(arrowstyle='-',lw=.5,color='#735018'))
    ax.set(xlim=(-3.55,3.55), ylim=(-.70,4.6), aspect='equal',
        xlabel='x (m)', ylabel='y (m)', xticks=(-3,0,3), yticks=(0,2,4))
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('(a) Saved scene and executed paths', loc='left', pad=7)
    handles=[Line2D([0],[0],color=COLORS['G'],lw=1.6,ls='--',label='G: active geometry'),
        Line2D([0],[0],color=COLORS['S'],lw=1.6,label='S: semantic belief'),
        Line2D([0],[0],color='#DDE0E2',lw=5,label='Shared 18-action prefix')]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.28),
        frameon=False,handlelength=2.1,labelspacing=.5,borderaxespad=0)
    fig.text(.075,.91,'P00 / h1: class predicts the useful side',fontsize=9,weight='bold')
    fig.text(.075,.105,'Both: 42 actions, exact return\nCoverage = 563 / 588',fontsize=8.2,linespacing=1.5)

    display = {}
    # Identical rear-right orthographic view exposes the task-relevant side.
    # Every saved triangle is rendered: no decimation, hole filling or GT overlay.
    for mode, ypos, label in [('G', .48, '(b)'), ('S', .045, '(c)')]:
        record = records[mode]
        mesh_ax = fig.add_axes([.425,ypos,.56,.435],projection='3d')
        facets = record['vertices'][record['triangles']]
        normals = np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0])
        lengths = np.linalg.norm(normals,axis=1)
        normals = np.divide(normals,lengths[:,None],out=np.zeros_like(normals),where=lengths[:,None]>0)
        light = np.asarray([.45,.6,1.]); light /= np.linalg.norm(light)
        shade = .55 + .45*np.abs(normals @ light)
        rgb = np.asarray(to_rgb(COLORS[mode]))
        facecolors = np.clip(rgb[None,:]*shade[:,None]+.10,0,1)
        collection = Poly3DCollection(facets,facecolors=facecolors,edgecolors='none',
            linewidths=0,antialiased=False,zsort='average',rasterized=True)
        mesh_ax.add_collection3d(collection)
        mesh_ax.set(xlim=(-3.1,3.1),ylim=(.85,3.7),zlim=(-.10,1.95))
        mesh_ax.set_box_aspect((6.2,2.85,2.05))
        mesh_ax.view_init(elev=24,azim=55,roll=0)
        mesh_ax.set_proj_type('ortho')
        mesh_ax.set_axis_off()
        metric = record['result']['stages']['final']['measurement']['05cm']
        mesh_ax.text2D(.03,.95,f'{label} {mode}: saved final TSDF',transform=mesh_ax.transAxes,
            va='top',fontsize=9,color=COLORS[mode],weight='bold')
        mesh_ax.text2D(.03,.08,f"F1@5 cm = {metric['f1']:.3f}     J5 = {metric['joint']:.3f}",
            transform=mesh_ax.transAxes,fontsize=8.5,va='bottom')
        display[mode]={'triangles_displayed':len(facets),'saved_metrics_copied':metric,
            'mesh_display_transform':'subtract the common public [3.5, 0.5, 0] translation; no alignment',
            'normals_used_for_display_shading_only':True}

    stem=OUT/'v38_saved_paths_and_tsdf'
    outputs=[]
    for suffix in ('pdf','svg','png'):
        checked_space()
        path=stem.with_suffix('.'+suffix)
        metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else {'Date':None} if suffix=='svg' else None
        fig.savefig(path,dpi=300,metadata=metadata)
        outputs.append(path)
    plt.close(fig)

    path=OUT/'trajectories.csv'
    with path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['method','paid_action_completed','x_m','y_m','heading_index','action_just_completed','next_action_selected','source_trace'])
        for mode, record in records.items():
            for row in record['trace']:
                writer.writerow([mode,row['paid'],*row['pose_v33'],row['action'],row['next_action'],record['source_folder']+'/trace.json'])
    outputs.append(path)
    SOURCES['scripts/build_thesis_qualitative_v38.py']={'sha256':sha(__file__),'bytes':Path(__file__).stat().st_size}
    assert all(sha(ROOT/rel)==value['sha256'] for rel,value in SOURCES.items())
    receipt={'version':'v38-saved-qualitative-1','created_date':'2026-09-20',
        'figure_width_inches':7.05,'figure_height_inches':4.45,'raster_dpi':300,
        'font_family':'Liberation Sans','font_path':font,
        'camera_shared':{'elevation_degrees':24,'azimuth_degrees':55,'roll_degrees':0,'projection':'orthographic'},
        'display':display,'sources':SOURCES,
        'outputs':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs},
        'caption_zh':'P00/h1 保存数据的定性展示。(a) 灰色设施几何为离线真值示意，虚框为双方共同公共 ROI；路径来自真实保存的动作位姿，粗浅灰线为共同18动作前缀，灰虚线G与蓝实线S为之后的实际路径（往返重叠段及原地转向不分开绘制）。橙色点表示人工RGB类别提示所在前表面；类别—观察侧关联是双方声明的公共先验。(b,c) 同一正交视角的实际终点TSDF公共ROI网格，显示全部保存三角形，不补洞、不重建、不加真值表面。颜色仅区别方法，明暗仅表示展示光照，不表示误差。保存评分直接抄录；两方法覆盖均563/588，均42动作返航。本例为h1正增量示例，h0零增量另在定量图完整报告。',
        'scope':{'offline_ground_truth_scene_context':True,'source_mesh_modified':False,
            'all_saved_triangles_rendered':True,'same_mesh_camera_and_limits':True,'mesh_decimation':False,
            'new_worlds':0,'new_sensor_packets':0,'new_planner_calls':0,'new_TSDF_integrations':0,
            'new_quality_evaluations':0,'new_counterfactual_outcomes':0,
            'ROI_outside_errors_excluded_from_saved_precision':True}}
    (OUT/'provenance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    total=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    if total>LIMIT: raise RuntimeError(f'figure pack exceeds 2 MiB: {total}')
    if shutil.disk_usage(ROOT).free<RESERVE: raise RuntimeError('reserve violated')
    print(json.dumps({'output':str(OUT),'total_bytes':total,'sources_verified':len(SOURCES),'displayed':display},indent=2))


if __name__ == '__main__':
    main()
