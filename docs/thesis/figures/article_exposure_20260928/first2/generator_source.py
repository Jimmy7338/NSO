#!/usr/bin/env python3
"""Publication adapter for reviewed GroundV2/ExposureV3 snapshots only.

GroundV2 is Exposure Off; ExposureV3 is Exposure On. Both already use the same
ground-aware frontend. No World, policy replay, fusion or quality evaluation.
Rendering requires an explicit --render release and a fresh output directory.
"""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/nso-article-exposure-mpl')
from scripts import plot_article_ground_comparisons_20260928 as base

plt,np,scenes=base.plt,base.np,base.scenes
sha,load,save=base.sha,base.load,base.save
METHODS,FAMILIES,COLORS,INK=base.METHODS,base.FAMILIES,base.COLORS,base.INK
ARMS=('GroundV2','ExposureV3')
LABELS=dict(GroundV2='Exposure Off',ExposureV3='Exposure On')
VERSION='article.common_numeric_face_evaluation.v1'
MAX_OUTPUT_BYTES=12*1024**2


def require(condition,message):
    if not condition:raise ValueError(message)


def verified_snapshot(root):
    manifest=load(root/'manifest.json')
    require(manifest['schema']=='article.exposure_comparison_manifest.v1','Exposure paired analysis snapshot required')
    for name,pin in manifest['files'].items():
        path=root/name
        require(not path.is_symlink() and not Path(name).is_absolute() and '..' not in Path(name).parts
            and path.resolve().is_relative_to(root.resolve()),'safe snapshot member')
        require(path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],'altered snapshot: '+name)
    require({'slots.csv','trajectories.csv','paired_effects.csv','summary.json'}<=set(manifest['files']), 'required sealed tables')
    def rows(name):
        with (root/name).open(newline='') as stream:return list(csv.DictReader(stream))
    slots=rows('slots.csv');keys=[(r['arm'],r['scene_id'],r['method']) for r in slots]
    expected={(a,'ART1_'+f+'_DEV',m) for a in ARMS for f in FAMILIES for m in METHODS}
    require(len(slots)==24 and len(set(keys))==24 and set(keys)==expected,'all twenty-four declared slots, including missing')
    by={r['run_id']:r for r in slots};require(len(by)==24,'unique run names')
    traces={}
    for row in rows('trajectories.csv'):
        require(row['run_id'] in by,'declared trajectory identity');slot=by[row['run_id']]
        require(all(row[k]==slot[k] for k in ('arm','scene_id','method','original_end_to_end_qualified','motion_completion_verified')),
            'trajectory identity and qualification binding')
        require(slot['review_passed']=='True' and slot['trace_status']=='sealed_trace_available','independently reviewed saved trajectory')
        require(all(math.isfinite(float(row[k])) for k in ('x_m','y_m','yaw_rad','cumulative_path_m')),'finite physical pose')
        traces.setdefault(row['run_id'],[]).append(row)
    for run,values in traces.items():
        slot=by[run]
        require([int(r['paid_step']) for r in values]==list(range(int(slot['paid_actions'])+1)),'consecutive complete saved path')
        total=0.
        for i,row in enumerate(values):
            if i:total+=math.hypot(float(row['x_m'])-float(values[i-1]['x_m']),float(row['y_m'])-float(values[i-1]['y_m']))
            require(abs(total-float(row['cumulative_path_m']))<=1e-9,'actual path cumulative distance')
        require(abs(total-float(slot['path_length_m']))<=1e-9,'actual path matches reviewed cost')
    for row in slots:
        if row['quality_measurement_available']=='True':
            require(row['metric_version']==VERSION,'canonical metric only')
            require(all(0.<=float(row[k])<=1.+1e-12 for k in ('C_nav','P','R','F1','J_nav')),'bounded stored quality')
            require(abs(float(row['J_nav'])-float(row['C_nav'])*float(row['F1']))<=1e-12,'joint arithmetic')
    return slots,traces


def path_features(values):
    xy=np.asarray([[float(r['x_m']),float(r['y_m'])] for r in values]);yaw=np.asarray([float(r['yaw_rad']) for r in values])
    return dict(xy=xy,yaw=yaw,
        turns=[i for i,r in enumerate(values) if r['actual_sensor_action'] in ('turn_left','turn_right')],
        observations=[i for i,r in enumerate(values) if r['actual_sensor_action']=='observe' and int(r['paid_step'])>0])


def status_label(row):
    status=row['status']
    if status=='unstarted':return 'unstarted'
    if status=='running_reserved':return 'running'
    if status=='awaiting_independent_review':return 'awaiting review'
    if status=='failed_attempt_reviewed':return 'failed attempt'
    if status in ('analysis_binding_error','review_findings'):return 'review issue'
    if row['motion_completion_verified']=='False':return 'incomplete return'
    if row['original_end_to_end_qualified']=='False':return 'unqualified'
    return status.replace('_',' ')


def score_text(row):
    return f"{float(row['J_nav']):.3f}"+('†' if row['original_end_to_end_qualified']=='False' else '') if base.has_metric(row) else status_label(row)


def plot_path(ax,values,method,arm):
    features=path_features(values);xy,yaw=features['xy'],features['yaw'];on=arm=='ExposureV3'
    color=COLORS[method] if on else '#88959E';order=11 if on else 10
    ax.plot(*xy.T,color=color,lw=1.45 if on else 2.4,ls='-' if on else (0,(3,2)),alpha=1. if on else .78,zorder=order)
    if features['turns']:
        ax.scatter(*xy[features['turns']].T,s=12,facecolors='none',edgecolors=color,lw=.6,zorder=order+2)
    if features['observations']:
        ax.scatter(*xy[features['observations']].T,s=16,marker='s',facecolors=color if on else 'white',edgecolors=color,lw=.6,zorder=order+3)
    indices=np.arange(0,len(values),16)
    ax.quiver(*xy[indices].T,np.cos(yaw[indices]),np.sin(yaw[indices]),color=color,
        angles='xy',scale_units='xy',scale=3.5,width=.0035,zorder=order+4,alpha=1. if on else .55)
    ax.scatter(*xy[0],s=42,marker='D',facecolors='white',edgecolors=INK,lw=1.2,zorder=20)
    ax.scatter(*xy[-1],s=25,marker='x',color=color,lw=1.,zorder=21)


def routes(slots,traces,records,output):
    fig,axes=plt.subplots(3,4,figsize=(13.2,13.4))
    for i,family in enumerate(FAMILIES):
        for j,method in enumerate(METHODS):
            ax=axes[i,j];scene='ART1_'+family+'_DEV';scenes.plan_map(ax,records[scene],show_axes=False)
            pair={r['arm']:r for r in slots if r['scene_id']==scene and r['method']==method}
            for arm in ARMS:
                row=pair[arm]
                if row['run_id'] in traces:plot_path(ax,traces[row['run_id']],method,arm)
            off,on=pair['GroundV2'],pair['ExposureV3'];lines=[f"J  Off: {score_text(off)} | On: {score_text(on)}"]
            for short,row in [('Off',off),('On',on)]:
                if row['run_id'] in traces:
                    suffix='' if row['original_end_to_end_qualified']=='True' else ' †'
                    lines.append(f"{short}: {float(row['path_length_m']):.1f} m · {int(row['turns'])} turns · {int(row['explicit_observe_actions'])} obs{suffix}")
                else:lines.append(short+': '+status_label(row))
            ax.text(.5,-.18,'\n'.join(lines),transform=ax.transAxes,ha='center',va='top',fontsize=7.5,color=INK)
            if j==0:ax.set_ylabel(family+'\ny (m)',fontsize=10,color=INK)
            if i==2:ax.set_xlabel('x (m)',fontsize=10,color=INK)
            if i==0:ax.set_title(method,fontsize=13,color=COLORS[method],pad=10,weight='bold')
            ax.set_xticks([0,2,4,6,8]);ax.set_yticks([0,2,4,6,8]);ax.tick_params(labelsize=8,colors='#657581')
    handles=[base.Line2D([0],[0],color='#88959E',lw=2.4,ls='--',label='Exposure Off (GroundV2)'),
        base.Line2D([0],[0],color=INK,lw=1.45,label='Exposure On (ExposureV3)'),
        base.Line2D([0],[0],color=INK,ls='',marker='D',mfc='white',label='Start'),
        base.Line2D([0],[0],color=INK,ls='',marker='x',label='Recorded endpoint'),
        base.Line2D([0],[0],color=INK,ls='',marker='s',label='Extra paid observation')]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.998),frameon=False,fontsize=9)
    fig.subplots_adjust(top=.915,bottom=.145,hspace=.62,wspace=.2)
    fig.text(.06,.012,'Both arms use ground-aware association, identical sensor budgets and the same numerical surface metric.\n'
        'Saved paths and headings; no smoothing. Open circles mark turns. † Original unqualified status remains retained.\n'
        'Missing/running/failed conditions are labelled explicitly; no path or score is inferred for them.',fontsize=8,color=INK)
    save(fig,output,'exposure_paired_routes')


def quality(slots,output):
    fig,axes=plt.subplots(3,3,figsize=(10.2,8.6),sharex=True,sharey=True)
    metrics=(('C_nav','Measured navigable coverage'),('F1','All-instance surface F1'),('J_nav','Joint documentation score'))
    for i,family in enumerate(FAMILIES):
        for j,(key,title) in enumerate(metrics):
            ax=axes[i,j]
            for x,method in enumerate(METHODS):
                pair={r['arm']:r for r in slots if r['scene_id']=='ART1_'+family+'_DEV' and r['method']==method};points=[]
                for delta,arm in [(-.075,'GroundV2'),(.075,'ExposureV3')]:
                    row=pair[arm]
                    if base.has_metric(row):
                        y=float(row[key]);points.append((x+delta,y))
                        ax.scatter(x+delta,y,s=42,facecolors='white' if arm=='GroundV2' else COLORS[method],edgecolors=COLORS[method],lw=1.2,zorder=3)
                        if row['original_end_to_end_qualified']=='False':ax.annotate('†',(x+delta,y),xytext=(-1,7),textcoords='offset points',ha='center',fontsize=10)
                    else:ax.text(x+delta,.025,'—',ha='center',color=COLORS[method],fontsize=10)
                if len(points)==2:ax.plot(*np.asarray(points).T,color=COLORS[method],lw=1.,zorder=2)
            if i==0:ax.set_title(title,fontsize=10,color=INK)
            if j==0:ax.set_ylabel(family,fontsize=10,color=INK)
            ax.set_xticks(range(4),METHODS,fontsize=9);ax.set_ylim(0,1.04);ax.set_xlim(-.5,3.5)
            ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_color('#B9C0C5')
            ax.grid(axis='y',color='#E4E8EB',lw=.6);ax.tick_params(labelsize=8,colors=INK)
    handles=[base.Line2D([0],[0],ls='',marker='o',mfc='white',mec=INK,label='Exposure Off (GroundV2)'),
        base.Line2D([0],[0],ls='',marker='o',color=INK,label='Exposure On (ExposureV3)')]
    fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.005),frameon=False,fontsize=9)
    fig.subplots_adjust(top=.92,bottom=.13,hspace=.23,wspace=.17)
    fig.text(.08,.014,'One point per saved episode; three development layouts. No inferential error bars.\n'
        'Dashes mean unavailable completed-return outcomes, not zero scores; detailed statuses remain in the path figure and slots.csv.\n'
        '† Original unqualified status is retained separately. Both arms use the same canonical surface measurement.',fontsize=8,color=INK)
    save(fig,output,'exposure_paired_quality')


def render(analysis,output):
    root=analysis.resolve();output=output.resolve()
    require(not output.exists() and output.is_relative_to(ROOT/'docs/thesis/figures'),'fresh thesis figure snapshot required')
    manifest_pin=sha(root/'manifest.json');sources={str(Path(p).relative_to(ROOT)):sha(p) for p in (__file__,base.__file__,scenes.__file__)}
    slots,traces=verified_snapshot(root);records,_=scenes.load_scenes()
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    output.mkdir(parents=True);routes(slots,traces,records,output);quality(slots,output)
    (output/'generator_source.py').write_bytes(Path(__file__).read_bytes())
    captions=dict(routes_en='All twelve matched development conditions, including missing attempts. Dashed grey is Exposure Off (GroundV2); solid colored is Exposure On (ExposureV3). Both arms use ground-aware association and full measured-depth fusion. Routes, headings, turns and observation positions come from saved sensor receipts. Endpoint markers do not imply successful return; original qualification is retained separately. Equipment outlines are offline truth context only.',
        routes_zh='全部十二组固定开发条件，保留缺失尝试。灰虚线为曝光抵扣关闭的GroundV2，彩实线为开启的ExposureV3；两臂均使用共同地面关联前端和完整实测深度融合。路径、朝向、转向和观测位置来自保存传感记录；终点标记不代表成功返航，流程资格单列。设备真值轮廓仅作离线解释。',
        quality_en='Measured navigable coverage, unweighted four-instance macro surface F1, and their product under one canonical numerical evaluation. Each point is one saved episode; the three development layouts are descriptive units. Missing values are not zeros. No quality is inferred from routes and no main-experiment mechanism approval follows from these figures.',
        quality_zh='共同数值评价下的实测可导航覆盖、四实例等权宏平均表面F1及其乘积。每点为一条保存任务，三个开发布局作为描述单元；缺失值不是零。路径不用于推算质量，这些图也不代表主实验机制门已通过。')
    (output/'captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2)+'\n')
    for name,pin in sources.items():require(sha(ROOT/name)==pin,'plot source changed during rendering')
    require(sha(root/'manifest.json')==manifest_pin,'analysis seal changed during rendering')
    manifest=dict(schema='article.exposure_comparison_figures.v1',analysis=str(root.relative_to(ROOT)),analysis_manifest_sha256=manifest_pin,
        source_sha256=sources,scene_sources=scenes.SOURCES,arm_labels=LABELS,declared_arm_records=24,
        saved_reviewed_paths=len(traces),motion_complete_paths=sum(r['motion_completion_verified']=='True' and r['run_id'] in traces for r in slots),
        status_counts={a:dict(Counter(r['status'] for r in slots if r['arm']==a)) for a in ARMS},
        new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    require(sum(p.stat().st_size for p in output.iterdir())<=MAX_OUTPUT_BYTES,'figure output exceeds 12MiB')
    return dict(output=str(output),saved_reviewed_paths=len(traces),declared_arm_records=24)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render',action='store_true',help='Explicit release after reviewed analysis snapshot')
    args=parser.parse_args()
    if not args.render:parser.error('require explicit --render; no automatic real-data rendering')
    print(json.dumps(render(args.analysis,args.output)))


if __name__=='__main__':main()
