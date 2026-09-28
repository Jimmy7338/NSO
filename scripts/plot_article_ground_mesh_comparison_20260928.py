#!/usr/bin/env python3
"""Deterministic nine-panel display of sealed raw G meshes, without evaluation.

Three fixed development layouts; GT / original V1 / GroundV2. All stored
triangles are rendered. No simulator, controller, TSDF or metric implementation
is imported. The failed original CELL pipeline remains explicitly failed.
"""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLCONFIGDIR','/tmp/nso-ground-mesh-comparison-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import plot_article_scene_meshes_20260928 as draw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ROOT=draw.ROOT
STAGE=ROOT/'audit_results/article_stage_20260928'
MAX_BYTES=15*1024**2
FAMILIES=('AISLE','CELL','LOOP')
METHOD='G'
METRIC='article.common_numeric_face_evaluation.v1'
ELEVATION,AZIMUTH=53.,-63.
RENDER_DPI=210


def require(test,message):
    if not test:raise ValueError(message)


def load_inputs(analysis):
    manifest=draw.verify_manifest(analysis)
    require(manifest['schema']=='article.ground_comparison_manifest.v1','reviewed Ground comparison snapshot required')
    summary=draw.read(analysis/'summary.json')
    require(summary['findings']==0 and draw.read(analysis/'findings.json')==[] and summary['metric_version']==METRIC,'released consistent common-version comparison')
    with draw.source(analysis/'slots.csv').open(newline='') as stream:slots=list(csv.DictReader(stream))
    require(len(slots)==24,'all twelve paired cells retained in snapshot')
    pins=manifest['input_sha256']
    def pinned(path):
        path=Path(path);key=str(path.relative_to(ROOT));record=pins.get(key)
        require(record is not None and not record.get('mutable_snapshot'),'immutable snapshot input binding:'+key)
        require(draw.sha(draw.source(path))==record['sha256'],'snapshot input SHA:'+key)
        return draw.read(path)
    assets=STAGE/'scene_assets_v1';asset_manifest=draw.read(assets/'manifest.json')
    require(draw.sha(assets/'manifest.json')==draw.ASSET_PIN,'fixed article asset bundle')
    scenes=[];panel_records=[]
    for family in FAMILIES:
        scene='ART1_'+family+'_DEV';gt_path=assets/scene/'renderer_private/geometry.npz'
        require(draw.sha(draw.source(gt_path))==asset_manifest['artifact_sha256'][scene+'/renderer_private/geometry.npz'],'exact reference mesh asset')
        gt=draw.mesh(gt_path);panels=[dict(kind='GT',mesh=gt,slot=None)]
        for arm,phase in [('GroundOff','development_v1'),('GroundOn','ground_ablation_v2')]:
            candidates=[r for r in slots if r['scene_id']==scene and r['arm']==arm and r['method']==METHOD]
            require(len(candidates)==1,'fixed G has exactly one result per arm/layout')
            row=candidates[0];run=row['run_id'];episode=STAGE/phase/'episodes'/run
            require(row['review_passed']=='True' and row['motion_completion_verified']=='True'
                and row['quality_measurement_available']=='True' and row['metric_version']==METRIC,'reviewed saved motion and common quality required')
            require(row['budget']=='160' and row['noise_seed']=='92801','fixed declared development budget/seed')
            if arm=='GroundOn':
                review=pinned(STAGE/'episode_reviews_ground_v2'/run/'review.json')
                require(review['all_checks_passed'] and review['qualified'],'Ground terminal independently reviewed')
                episode_manifest=pinned(episode/'artifact_manifest.json')
                require(draw.sha(episode/'artifact_manifest.json')==review['input_manifest_sha256'],'reviewed episode seal')
                predicted=episode_manifest['files']['prediction/mesh.npz'];mesh_pin=predicted['sha256']
                metric=review['metrics']
                require(row['original_end_to_end_qualified']=='True','Ground original qualification preserved')
            else:
                common=STAGE/'common_evaluation_v1'/run
                measure=pinned(common/'measurement/result.json');captured=pinned(common/'prepared/input_snapshot.json')
                require(measure['measurement_version']==METRIC and measure['quality_measurement_available']
                    and measure['motion_completion_verified'],'common derived metric with complete saved motion')
                require(measure['original_prediction_seal']==captured['original_prediction_seal'],'derived metric raw prediction binding')
                predicted=captured['input_files']['prediction/mesh.npz'];mesh_pin=predicted['sha256']
                require(mesh_pin==measure['original_prediction_seal']['mesh.npz'],'raw complete mesh is the evaluated source')
                metric=measure['metrics']
                require(measure['original_end_to_end_status']==row['online_status']
                    and str(measure['original_end_to_end_qualified'])==row['original_end_to_end_qualified'],'original failure never upgraded by plotting')
                if family=='CELL':
                    require(row['online_status']=='attempt_failed' and measure['mode']=='new_derived_measurement'
                        and row['original_end_to_end_qualified']=='False','CELL original evaluation failure and later measurement retained')
            for csv_key,key in [('C_nav','C_nav'),('P','macro_precision'),('R','macro_completeness'),('F1','Q'),('J_nav','J_nav')]:
                require(float(row[csv_key])==metric[key],'display metric equals released snapshot and sealed review:'+csv_key)
            path=draw.source(episode/'prediction/mesh.npz')
            require(path.stat().st_size==predicted['bytes'] and draw.sha(path)==mesh_pin,'raw predicted geometry bytes unchanged')
            panels.append(dict(kind=arm,mesh=draw.mesh(path),slot=row))
        scenes.append(dict(family=family,scene_id=scene,panels=panels))
        for item in panels:
            row=item['slot'];panel_records.append(dict(scene_id=scene,kind=item['kind'],method=None if row is None else METHOD,
                run_id=None if row is None else row['run_id'],vertices=len(item['mesh']['vertices']),
                original_triangles=len(item['mesh']['faces']),displayed_triangles=len(item['mesh']['faces']),
                full_raw_mesh=True,decimation=False,smoothing=False,triangle_crop=False,semantic_weighting=False,
                original_online_status=None if row is None else row['online_status'],
                original_end_to_end_qualified=None if row is None else row['original_end_to_end_qualified']=='True',
                metric_version=None if row is None else row['metric_version'],
                quality_mode=None if row is None else row['quality_mode'],
                metrics=None if row is None else {k:float(row[k]) for k in ('C_nav','P','R','F1','J_nav')}))
    # Bounds cover every raw vertex from every panel; the same orthographic
    # view and metric scale apply across all nine panels, not just each row.
    all_meshes=[p['mesh'] for s in scenes for p in s['panels']]
    low=np.min([m['vertices'].min(axis=0) for m in all_meshes],axis=0)-.10
    high=np.max([m['vertices'].max(axis=0) for m in all_meshes],axis=0)+.10
    return scenes,panel_records,np.stack([low,high])


def panel_image(mesh,bounds):
    fig=plt.figure(figsize=(3.70,3.00));ax=fig.add_axes([.012,.015,.925,.965],projection='3d',computed_zorder=False)
    triangles=mesh['triangles']
    if len(triangles):
        ax.add_collection3d(Poly3DCollection(triangles,facecolors=draw.face_colors(triangles),edgecolors='none',
            linewidths=0,antialiased=False,zsort='average',rasterized=True))
    ax.set_proj_type('ortho');ax.view_init(elev=ELEVATION,azim=AZIMUTH)
    ax.set_xlim(*bounds[:,0]);ax.set_ylim(*bounds[:,1]);ax.set_zlim(*bounds[:,2]);ax.set_box_aspect(bounds[1]-bounds[0],zoom=1.0)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False;axis.pane.set_edgecolor('#E4E9EB');axis.line.set_color('#B6C4CC')
        axis._axinfo['grid'].update(color='#DFE6E9',linewidth=.45)
    ax.set_xticks([0,4,8]);ax.set_yticks([0,4,8]);ax.set_zticks([0,1.2,2.4])
    ax.tick_params(labelsize=6.4,pad=-1,colors=draw.MUTED,length=2)
    ax.set_xlabel('x (m)',fontsize=7,labelpad=-3,color=draw.MUTED)
    ax.set_ylabel('y (m)',fontsize=7,labelpad=-3,color=draw.MUTED)
    ax.set_zlabel('z (m)',fontsize=7,labelpad=-4,color=draw.MUTED)
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=RENDER_DPI,facecolor='white')
    plt.close(fig);buffer.seek(0);return plt.imread(buffer)


def render(scenes,records,bounds,output):
    fig=plt.figure(figsize=(12.0,10.8),facecolor='white')
    fig.text(.052,.980,'Observed reconstruction under a common geometric planner',fontsize=15,weight='bold',color=draw.INK,va='top')
    fig.text(.052,.951,'Three development layouts · method G · budget 160 · one noise realization (92801)',fontsize=9,color=draw.MUTED)
    centers=(.205,.521,.837)
    for x,title,detail in zip(centers,('Reference geometry','Original frontend','Ground association frontend'),
                            ('Offline ground truth','V1 · GroundOff','V2 · GroundOn')):
        fig.text(x,.918,title,ha='center',fontsize=11.2,color=draw.INK,weight='bold')
        fig.text(x,.901,detail,ha='center',fontsize=8.3,color=draw.MUTED)
    for row,scene in enumerate(scenes):
        top=.883-row*.257
        fig.text(.018,top-.097,scene['family'],rotation=90,ha='center',va='center',fontsize=10.5,weight='bold',color=draw.INK)
        for col,item in enumerate(scene['panels']):
            image=panel_image(item['mesh'],bounds)
            ax=fig.add_axes([.047+col*.316,top-.209,.310,.212]);ax.imshow(image);ax.axis('off')
            slot=item['slot'];faces=len(item['mesh']['faces'])
            if slot is None:
                text=f'{faces:,} triangles · full reference mesh'
            else:
                mark=' *' if slot['original_end_to_end_qualified']=='False' else ''
                text=f"C {float(slot['C_nav']):.3f}   F1 {float(slot['F1']):.3f}   J {float(slot['J_nav']):.3f}{mark}"
            fig.text(centers[col],top-.223,text,ha='center',fontsize=8.8,color=draw.INK)
            if slot is not None:
                fig.text(centers[col],top-.238,f'{faces:,} raw triangles · {slot["paid_actions"]} paid actions',ha='center',fontsize=7.5,color=draw.MUTED)
            print(json.dumps(dict(rendered=scene['scene_id']+'/'+item['kind'],triangles=faces)),flush=True)
    fig.text(.052,.090,'All saved triangles; one orthographic camera and metric scale. No smoothing, decimation or crop.',fontsize=8.1,color=draw.MUTED)
    fig.text(.052,.067,'* CELL / Original: online evaluation failed; saved motion completed. Displayed score is a separate\n  common-version derived measurement. The original episode remains failed.',fontsize=7.8,color='#825C36',linespacing=1.3)
    cb=fig.add_axes([.70,.032,.245,.011]);bar=fig.colorbar(plt.cm.ScalarMappable(norm=draw.NORM,cmap=draw.CMAP),cax=cb,orientation='horizontal')
    bar.set_ticks([-.1,1.2,2.4]);bar.ax.tick_params(labelsize=7,length=2);bar.outline.set_visible(False)
    bar.set_label('Height (m); not reconstruction error',fontsize=7.2,color=draw.MUTED,labelpad=1)
    fig.text(.052,.030,'Four-instance unweighted macro F1; J = C × F1. GT is for offline display only.',fontsize=7.8,color=draw.MUTED)
    for ext in ('png','pdf','svg'):
        metadata=({'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None)
        fig.savefig(output/f'ground_frontend_meshes_G.{ext}',dpi=210,facecolor='white',metadata=metadata)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=STAGE/'analysis_v1/ground12_complete')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/thesis/figures/article_ground_meshes_20260928')
    args=parser.parse_args();require(not args.output.exists(),'fresh figure directory; never overwrite prior draft')
    start=time.monotonic();scenes,records,bounds=load_inputs(args.analysis)
    args.output.mkdir(parents=True)
    print(json.dumps(dict(data_consumption_complete=True,raw_meshes_loaded=9,triangle_count=sum(r['original_triangles'] for r in records))),flush=True)
    render(scenes,records,bounds,args.output)
    captions='''# Figure caption / 图注

**English.** Complete saved meshes under the same geometry-only planner G in three development layouts, comparing the original V1 frontend with GroundV2. Each row shows reference geometry, original-frontend reconstruction and ground-association reconstruction. All nine panels share orthographic camera, world scale and height coloring; every original stored triangle is retained. Color indicates height, not error or semantic importance. C is measured navigational coverage, F1 the unweighted four-instance macro surface score, and J=C×F1 under article.common_numeric_face_evaluation.v1. These are development/component-ablation results, not unseen-layout confirmation or a semantic-method comparison. For CELL/V1, the original online evaluation failed on a numerical degenerate face; the motion is complete and the displayed common-version score was subsequently derived from the sealed prediction. Its original failed status is retained. The raw displayed mesh includes that face; no display crop, smoothing or decimation is used. Full-frame fusion can reconstruct surfaces of instances not registered by the online semantic frontend.

**中文。** 固定几何规划方法G在三种开发布局中的完整封存网格，对比原V1前端和GroundV2关联前端。每行依次为离线参考几何、原前端重建、新地面关联前端重建；九幅统一正交相机、米制尺度和高度配色，保留所有原始三角面。颜色表示高度，不表示误差或语义权重。C为实际导航覆盖率，F1为四实例无权重宏平均表面分数，J=C×F1，全部采用同一规范评价版本。这是开发场景的共同组件消融，不是未见布局验证，也不证明语义方法增益。CELL/V1原在线评价因数值退化面失败，实际运动完整；图中分数来自封存预测的后续共同版本测量，原失败资格不变。展示仍用含该面的原始网格，无裁剪、平滑、简化。整帧融合也可能重建在线语义前端未登记实例的表面。
'''
    (args.output/'captions.md').write_text(captions)
    (args.output/'panel_records.json').write_text(json.dumps(dict(panels=records,bounds_xyz_m=bounds.tolist(),
        projection='orthographic',elevation_deg=ELEVATION,azimuth_deg=AZIMUTH,render_dpi=RENDER_DPI,
        semantic_weighting=False,all_triangles_retained=True,display_gt_offline_only=True,
        new_worlds=0,new_tsdf_integrations=0,new_quality_evaluations=0),ensure_ascii=False,indent=2)+'\n')
    draw.source(Path(__file__));draw.source(Path(draw.__file__))
    for source_file in (Path(__file__),Path(draw.__file__)):
        target=args.output/'source_archive'/source_file.name;target.parent.mkdir(exist_ok=True)
        target.write_bytes(source_file.read_bytes())
    manifest=dict(schema='article.ground_raw_mesh_figure.v1',analysis=str(args.analysis.relative_to(ROOT)),
        sources=draw.SOURCES,files={str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=draw.sha(p)) for p in args.output.rglob('*') if p.is_file()},
        method_fixed_before_render=METHOD,layouts=list(FAMILIES),panels=9,elapsed_s=time.monotonic()-start,
        output_cap_bytes=MAX_BYTES,new_worlds=0,new_quality_evaluations=0,new_tsdf_integrations=0,
        metric_version=METRIC,raw_triangles=sum(r['original_triangles'] for r in records))
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    total=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file());require(total<=MAX_BYTES,'fixed 15MiB figure package cap')
    print(json.dumps(dict(output=str(args.output),bytes=total,elapsed_s=manifest['elapsed_s'])),flush=True)


if __name__=='__main__':main()
