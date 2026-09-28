#!/usr/bin/env python3
"""Fixed-method G, all-layout print strip from the reviewed Ground snapshot.

This is a display-only selection fixed in the task before making this figure.
All three layouts and both signed directions remain. No World/raw-episode
access, policy replay, fusion, or quality evaluation is performed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/nso-ground-g-print-strip-mpl')
from scripts import plot_article_ground_comparisons_20260928 as base

plt,np,scenes=base.plt,base.np,base.scenes
METHOD='G'
FAMILIES=('AISLE','CELL','LOOP')
ARMS=('GroundOff','GroundOn')
STEM='ground_fixed_g_routes'
MAX_OUTPUT_BYTES=3*1024**2
DEFAULT_ANALYSIS=ROOT/'audit_results/article_stage_20260928/analysis_v1/ground12_complete'
DEFAULT_OUTPUT=ROOT/'docs/thesis/figures/article_ground_20260928/fixed_g_print_v1'


def require(value,message):
    if not value:raise ValueError(message)


def validate_paths(slots,traces):
    selected=[r for r in slots if r['method']==METHOD]
    require(len(selected)==6 and {(r['arm'],r['scene_id']) for r in selected}
        =={(a,'ART1_'+f+'_DEV') for a in ARMS for f in FAMILIES},'fixed G, all three layouts, both arms')
    records=[]
    for row in selected:
        require(row['review_passed']=='True' and row['motion_completion_verified']=='True'
            and row['quality_measurement_available']=='True' and row['run_id'] in traces,'reviewed full-return route and canonical quality')
        require(row['metric_version']=='article.common_numeric_face_evaluation.v1'
            and row['budget']=='160' and row['noise_seed']=='92801','common fixed development conditions')
        values=traces[row['run_id']];xy=np.asarray([[float(v['x_m']),float(v['y_m'])] for v in values])
        distance=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
        turns=sum(v['actual_sensor_action'] in ('turn_left','turn_right') for v in values)
        observations=sum(v['actual_sensor_action']=='observe' and int(v['paid_step'])>0 for v in values)
        require(abs(distance-float(row['path_length_m']))<=1e-9 and turns==int(row['turns'])
            and observations==int(row['explicit_observe_actions']),'actual saved route and action counts match reviewed CSV')
        require(np.allclose(xy[0],xy[-1],rtol=0,atol=1e-9)
            and abs((float(values[-1]['yaw_rad'])-float(values[0]['yaw_rad'])+math.pi)%(2*math.pi)-math.pi)<1e-9,
            'actual complete-pose return')
        if row['scene_id']=='ART1_CELL_DEV' and row['arm']=='GroundOff':
            require(row['original_end_to_end_qualified']=='False' and row['online_status']=='attempt_failed'
                and row['quality_mode']=='new_derived_measurement','CELL original evaluation failure retained')
        records.append({k:row[k] for k in ('run_id','scene_id','arm','method','online_status','original_end_to_end_qualified',
            'motion_completion_verified','quality_mode','metric_version','C_nav','P','R','F1','J_nav',
            'paid_actions','path_length_m','turns','explicit_observe_actions')})
    return selected,records


def route(ax,values,arm):
    xy=np.asarray([[float(r['x_m']),float(r['y_m'])] for r in values]);yaw=np.asarray([float(r['yaw_rad']) for r in values])
    on=arm=='GroundOn';color=base.COLORS[METHOD] if on else '#89969F';zorder=12 if on else 10
    ax.plot(*xy.T,color=color,lw=1.75 if on else 2.7,ls='-' if on else (0,(3.5,2.0)),alpha=1. if on else .8,zorder=zorder)
    turns=[i for i,r in enumerate(values) if r['actual_sensor_action'] in ('turn_left','turn_right')]
    observes=[i for i,r in enumerate(values) if r['actual_sensor_action']=='observe' and int(r['paid_step'])>0]
    ax.scatter(*xy[turns].T,s=24,facecolors='none',edgecolors=color,lw=.85,zorder=zorder+2)
    ax.scatter(*xy[observes].T,s=24,marker='s',facecolors=color if on else 'white',edgecolors=color,lw=.85,zorder=zorder+3)
    indices=np.arange(0,len(values),20)
    ax.quiver(*xy[indices].T,np.cos(yaw[indices]),np.sin(yaw[indices]),color=color,
        angles='xy',scale_units='xy',scale=3.4,width=.005,zorder=zorder+4,alpha=1. if on else .65)
    ax.scatter(*xy[0],s=70,marker='D',facecolors='white',edgecolors=base.INK,lw=1.4,zorder=30)


def signed(value):
    return '0' if value==0 else f'{value:+g}'.replace('-','−')


def render(analysis,output):
    analysis=analysis.resolve();output=output.resolve()
    require(not output.exists() and output.is_relative_to(ROOT/'docs/thesis/figures'),'fresh figure output only')
    snapshot_pin=base.sha(analysis/'manifest.json');slots,traces=base.verified_snapshot(analysis)
    require(base.load(analysis/'summary.json')['findings']==0 and base.load(analysis/'findings.json')==[], 'fully reviewed released comparison')
    selected,panel_records=validate_paths(slots,traces);scene_records,_=scenes.load_scenes()
    source_pins={str(Path(p).relative_to(ROOT)):base.sha(p) for p in (__file__,base.__file__,scenes.__file__)}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(10.4,5.45))
    differences=[]
    for index,(ax,family) in enumerate(zip(axes,FAMILIES)):
        scene='ART1_'+family+'_DEV';scenes.plan_map(ax,scene_records[scene],show_axes=False)
        for label in ax.texts:
            if label.get_text()=='H':label.set_visible(False)
            else:label.set_fontsize(12)
        pair={r['arm']:r for r in selected if r['scene_id']==scene}
        for arm in ARMS:route(ax,traces[pair[arm]['run_id']],arm)
        off,on=pair['GroundOff'],pair['GroundOn'];dagger='†' if off['original_end_to_end_qualified']=='False' else ''
        ax.set_title(f'({chr(97+index)}) {family}',fontsize=14,color=base.INK,pad=10)
        ax.tick_params(labelsize=12,colors='#657581')
        ax.set_xlabel('$x$ (m)',fontsize=12,labelpad=2)
        if index==0:ax.set_ylabel('$y$ (m)',fontsize=12,labelpad=2)
        ax.text(.5,-.27,f"$J$: {float(off['J_nav']):.3f}{dagger} → {float(on['J_nav']):.3f}",
            transform=ax.transAxes,ha='center',va='center',fontsize=13,color=base.INK)
        delta={k:float(on[k])-float(off[k]) for k in ('J_nav','path_length_m','turns','explicit_observe_actions')}
        ax.text(.5,-.385,f"Δ path {signed(delta['path_length_m'])} m · turns {signed(delta['turns'])}\npaid observe {signed(delta['explicit_observe_actions'])}",
            transform=ax.transAxes,ha='center',va='center',fontsize=12,color=base.INK,linespacing=1.4)
        differences.append(dict(scene_id=scene,direction='GroundOn minus GroundOff',**delta))
    handles=[base.Line2D([0],[0],color='#89969F',lw=2.7,ls='--',label='Ground Off'),
        base.Line2D([0],[0],color=base.COLORS[METHOD],lw=1.75,label='Ground On'),
        base.Line2D([0],[0],ls='',marker='D',mfc='white',mec=base.INK,label='Start / return'),
        base.Line2D([0],[0],ls='',marker='o',mfc='none',mec=base.INK,label='Turn'),
        base.Line2D([0],[0],ls='',marker='s',color=base.INK,label='Paid observe')]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.51,.985),frameon=False,
        fontsize=12,columnspacing=1.1,handlelength=1.8,handletextpad=.55)
    fig.subplots_adjust(left=.065,right=.992,top=.80,bottom=.30,wspace=.20)
    fig.text(.065,.025,'Fixed G; all three layouts. Arrows: recorded headings. Δ: On − Off.',fontsize=12,color=base.INK)
    output.mkdir(parents=True)
    for ext in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(output/(STEM+'.'+ext),dpi=240,facecolor='white',metadata=metadata)
    plt.close(fig)
    captions=dict(en='Fixed G comparison of the common ground-association component across all three development layouts, chosen by method and layout inventory rather than outcome. Grey dashed paths are Ground Off; blue solid paths are Ground On. The six paths use identical world-coordinate scales and saved sensor poses, with headings, actual turns and extra paid observations. Every episode has independently verified complete-pose return. J is measured navigable coverage times the unweighted four-instance macro surface F1 under the common numerical evaluation. Differences below each panel are On minus Off, including negative outcomes. † CELL Ground Off retains its original evaluation failure; its displayed score is the separately declared common-version measurement after verified motion completion. Equipment outlines are offline explanatory truth only. Results for all four methods and the complete twelve-condition route grid remain available in the full experimental material.',
        zh='固定G方法下共同地面关联模块在全部三个开发布局中的比较，按预定方法与布局清单选取而非按效果选择。灰虚线为Ground Off，蓝实线为Ground On。六条路径统一世界坐标尺度，来自保存传感位姿，并显示朝向、实际转向与额外付费观察；各条路径均经独立核验完成全位姿返航。J为实测可导航覆盖与四实例等权宏平均表面F1的乘积，统一采用共同数值评价。各面板差值为On减Off，保留负向结果。† CELL Ground Off原评价失败仍保留，所示分数来自运动完成后另行声明的共同版本补测。设备轮廓仅作离线真值解释。四种方法与完整十二组路线图保留在完整实验材料中。')
    (output/'captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2)+'\n')
    (output/'panel_records.json').write_text(json.dumps(dict(method=METHOD,layouts=list(FAMILIES),records=panel_records,
        differences=differences,selection_rule='fixed G and all three declared development layouts; no outcome threshold'),indent=2)+'\n')
    archive=output/'source_archive';archive.mkdir()
    for name,pin in source_pins.items():
        require(base.sha(ROOT/name)==pin,'source unchanged during render')
        (archive/Path(name).name).write_bytes((ROOT/name).read_bytes())
    require(base.sha(analysis/'manifest.json')==snapshot_pin,'released snapshot seal unchanged')
    manifest=dict(schema='article.ground_fixed_g_print_strip.v1',analysis=str(analysis.relative_to(ROOT)),
        analysis_manifest_sha256=snapshot_pin,source_sha256=source_pins,scene_sources=scenes.SOURCES,
        fixed_method=METHOD,all_development_layouts=list(FAMILIES),saved_actual_paths=6,
        figure_width_inches=10.4,minimum_annotation_points=12,minimum_points_at_7in=12*7/10.4,
        new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0,
        files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=base.sha(p)) for p in output.rglob('*') if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file());require(size<=MAX_OUTPUT_BYTES,'output exceeds 3MiB')
    return dict(output=str(output),bytes=size,saved_actual_paths=6)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=DEFAULT_ANALYSIS);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();print(json.dumps(render(args.analysis,args.output)))


if __name__=='__main__':main()
