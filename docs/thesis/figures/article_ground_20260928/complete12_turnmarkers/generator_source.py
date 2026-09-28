#!/usr/bin/env python3
"""Plot saved, reviewed paths and the common metric; no simulator or evaluator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_article_scene_overview_20260928 as scenes

METHODS = ('NBV', 'G', 'B', 'S')
FAMILIES = ('AISLE', 'CELL', 'LOOP')
COLORS = dict(NBV='#7B8794', G='#4477AA', B='#CC9933', S='#228833')
INK = '#253641'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def verified_snapshot(root):
    manifest = load(root/'manifest.json')
    if manifest['schema'] != 'article.ground_comparison_manifest.v1':
        raise ValueError('reviewed common-metric snapshot required')
    for name, pin in manifest['files'].items():
        path = root/name
        if path.is_symlink() or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe snapshot member')
        if path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
            raise ValueError('altered snapshot: '+name)
    def rows(name):
        with (root/name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    slots = rows('slots.csv')
    expected = {(arm, 'ART1_'+family+'_DEV', method) for arm in ('GroundOff','GroundOn')
        for family in FAMILIES for method in METHODS}
    keys = [(r['arm'],r['scene_id'],r['method']) for r in slots]
    if len(keys) != 24 or set(keys) != expected:
        raise ValueError('all twenty-four arm records required, including missing outcomes')
    traces = {}
    for row in rows('trajectories.csv'):
        if row['motion_completion_verified'] != 'True':
            raise ValueError('unverified trajectory')
        traces.setdefault(row['run_id'], []).append(row)
    for run, values in traces.items():
        slot = next(r for r in slots if r['run_id'] == run)
        if [int(r['paid_step']) for r in values] != list(range(int(slot['paid_actions'])+1)):
            raise ValueError('incomplete ordered actual path')
        if any(r['original_end_to_end_qualified'] != slot['original_end_to_end_qualified'] for r in values):
            raise ValueError('trajectory status disagrees with metric row')
    return slots, traces


def number(row, name):
    return float(row[name]) if row[name] else None


def has_metric(row):
    return row['quality_measurement_available'] == 'True' and row['motion_completion_verified'] == 'True'


def save(fig, output, stem):
    for ext in ('pdf','svg','png'):
        metadata = {'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(output/(stem+'.'+ext), bbox_inches='tight', dpi=220, facecolor='white', metadata=metadata)
    plt.close(fig)


def plot_path(ax, values, method, arm):
    xy = np.array([[float(r['x_m']),float(r['y_m'])] for r in values])
    yaw = np.array([float(r['yaw_rad']) for r in values])
    if arm == 'GroundOff':
        ax.plot(*xy.T, color='#88959E', ls=(0,(3,2)), lw=2.4, alpha=.75, zorder=10)
    else:
        color = COLORS[method]
        ax.plot(*xy.T, color=color, lw=1.45, zorder=11)
        turns = [i for i,r in enumerate(values) if r['actual_sensor_action'] in ('turn_left','turn_right')]
        observes = [i for i,r in enumerate(values) if r['actual_sensor_action']=='observe']
        if turns:
            ax.scatter(*xy[turns].T, s=15, facecolors='none', edgecolors=color, lw=.65, zorder=12)
        if observes:
            ax.scatter(*xy[observes].T, s=15, marker='s', color=color, edgecolors='white', lw=.35, zorder=13)
        indices = np.arange(0,len(values),16)
        ax.quiver(*xy[indices].T, np.cos(yaw[indices]),np.sin(yaw[indices]),color=color,
            angles='xy', scale_units='xy',scale=3.5,width=.004,zorder=14)
    ax.scatter(*xy[0],s=42,marker='D',facecolors='white',edgecolors=INK,lw=1.3,zorder=20)


def routes(slots, traces, records, output):
    fig,axes = plt.subplots(3,4,figsize=(13.2,13.1))
    for i,family in enumerate(FAMILIES):
        for j,method in enumerate(METHODS):
            ax = axes[i,j];scene='ART1_'+family+'_DEV'
            scenes.plan_map(ax,records[scene],show_axes=False)
            pair = {r['arm']:r for r in slots if r['scene_id']==scene and r['method']==method}
            for arm,row in pair.items():
                if row['run_id'] in traces:
                    plot_path(ax,traces[row['run_id']],method,arm)
            off,on = pair['GroundOff'],pair['GroundOn']
            old = f"{number(off,'J_nav'):.3f}" if has_metric(off) else 'unavailable'
            new = f"{number(on,'J_nav'):.3f}" if has_metric(on) else 'pending'
            mark = '†' if off['original_end_to_end_qualified']=='False' and has_metric(off) else ''
            line = f'J: {old}{mark} → {new}'
            if on['run_id'] in traces:
                line += f"\nOn: {number(on,'path_length_m'):.1f} m · {int(on['turns'])} turns\n{int(on['explicit_observe_actions'])} extra observations"
            else:
                line += '\nOn: '+on['status'].replace('_',' ')
            ax.text(.5,-.23 if i==2 else -.16,line,transform=ax.transAxes,ha='center',va='top',fontsize=8,color=INK)
            if j==0: ax.set_ylabel(family+'\ny (m)',fontsize=10,color=INK)
            if i==2: ax.set_xlabel('x (m)',fontsize=10,color=INK)
            if i==0: ax.set_title(method,fontsize=13,color=COLORS[method],pad=10,weight='bold')
            ax.set_xticks([0,2,4,6,8]);ax.set_yticks([0,2,4,6,8]);ax.tick_params(labelsize=8,colors='#657581')
    handles=[Line2D([0],[0],color='#88959E',lw=2.4,ls='--',label='Original association'),
        Line2D([0],[0],color=INK,lw=1.5,label='Ground-aware association'),
        Line2D([0],[0],color=INK,ls='',marker='D',mfc='white',label='Start / complete-pose return'),
        Line2D([0],[0],color=INK,ls='',marker='s',label='Extra paid observation (On)')]
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,.998),frameon=False,fontsize=9)
    fig.subplots_adjust(top=.935,bottom=.145,hspace=.59,wspace=.20)
    fig.text(.06,.005,'Budget: 160 paid actions. Saved paths and headings; no trajectory smoothing.\n'
        '† Original evaluation failed after successful motion; J is a separately derived common-version measurement.',fontsize=8,color=INK)
    save(fig,output,'ground_paired_routes')


def quality(slots, output):
    fig,axes = plt.subplots(3,3,figsize=(10.2,8.4),sharex=True,sharey=True)
    metrics=(('C_nav','Measured navigable coverage'),('F1','All-instance surface F1'),('J_nav','Joint documentation score'))
    for i,family in enumerate(FAMILIES):
        for j,(key,title) in enumerate(metrics):
            ax=axes[i,j]
            for x,method in enumerate(METHODS):
                pair={r['arm']:r for r in slots if r['scene_id']=='ART1_'+family+'_DEV' and r['method']==method}
                ys=[]
                for delta,arm in [(-.075,'GroundOff'),(.075,'GroundOn')]:
                    row=pair[arm]
                    if has_metric(row):
                        y=number(row,key);ys.append((x+delta,y))
                        ax.scatter(x+delta,y,s=42,facecolors='white' if arm=='GroundOff' else COLORS[method],
                            edgecolors=COLORS[method],lw=1.2,zorder=3)
                        if row['original_end_to_end_qualified']=='False':
                            ax.annotate('†',(x+delta,y),xytext=(-1,7),textcoords='offset points',ha='center',fontsize=10)
                    else:
                        ax.text(x+delta,.03,'—',ha='center',color=COLORS[method],fontsize=10)
                if len(ys)==2:ax.plot(*np.array(ys).T,color=COLORS[method],lw=1.,zorder=2)
            if i==0:ax.set_title(title,fontsize=10,color=INK)
            if j==0:ax.set_ylabel(family,fontsize=10,color=INK)
            ax.set_xticks(range(4),METHODS,fontsize=9);ax.set_ylim(0,1.04);ax.set_xlim(-.5,3.5)
            ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_color('#B9C0C5')
            ax.grid(axis='y',color='#E4E8EB',lw=.6);ax.tick_params(labelsize=8,colors=INK)
    handles=[Line2D([0],[0],ls='',marker='o',mfc='white',mec=INK,label='Original association'),
        Line2D([0],[0],ls='',marker='o',color=INK,label='Ground-aware association')]
    fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.005),frameon=False,fontsize=9)
    fig.subplots_adjust(top=.92,bottom=.11,hspace=.23,wspace=.17)
    fig.text(.08,.015,'Each point is one episode; three development layouts. No inferential error bars.\n'
        'Dashes denote missing outcomes, not zero scores. † Original evaluation failure retained; derived score shown.',fontsize=8,color=INK)
    save(fig,output,'ground_paired_quality')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.analysis.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('new immutable figure snapshot required')
    slots,traces=verified_snapshot(root)
    records,_=scenes.load_scenes()
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    output.mkdir(parents=True)
    routes(slots,traces,records,output);quality(slots,output)
    (output/'generator_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'captions.json').write_text(json.dumps(dict(
        routes_en='All declared matched development conditions under one numerical evaluation version. Dashed grey paths use original association; colored paths use measured-ground exclusion only in association. Both use full measured depth for reconstruction. Raw original evaluation failure is retained separately from motion completion and the derived score. Equipment outlines are offline ground-truth context, never controller inputs.',
        routes_zh='统一数值评价下全部声明开发条件的实际路径。灰虚线为原对象关联，彩实线为仅在关联中排除实测地面的版本，两者均融合完整测量深度。原评价失败、运动完成和派生分数分别保留。设备真值轮廓仅用于离线解释。',
        quality_en='Coverage, unweighted four-instance macro surface F1 and their product. Each point represents one saved episode; layouts are the independent descriptive units. Original evaluation failures and missing conditions remain visible. No quality is inferred from plotted paths.',
        quality_zh='实测覆盖、四实例等权宏平均表面F1及其乘积。每点为一条保存任务，布局作为描述性比较单元；原评价失败和缺失条件明确标示，不根据路径推算质量。'),ensure_ascii=False,indent=2)+'\n')
    manifest=dict(schema='article.ground_comparison_figures.v1',analysis=str(root.relative_to(ROOT)),
        analysis_manifest_sha256=sha(root/'manifest.json'),generator_sha256=sha(Path(__file__)),
        scene_figure_source_sha256=sha(Path(scenes.__file__)),scene_sources=scenes.SOURCES,
        declared_arm_records=len(slots),motion_complete_paths=len(traces),
        new_worlds=0,new_tsdf_integrations=0,new_surface_evaluations=0,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(output),files=len(manifest['files']),motion_complete_paths=len(traces))))


if __name__=='__main__':main()
