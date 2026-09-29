#!/usr/bin/env python3
"""Submission-size checkpoint figure from all forty sealed measurements.

Only reads saved scalar measurements. No World, mapper, sensor or evaluator
module is imported; the original five-column companion figure remains intact.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/nso-submission-checkpoints-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'audit_results/article_stage_20260928/reconstruction_checkpoints'
COMPANION=ROOT/'docs/thesis/figures/article_reconstruction_20260928'
STEPS=(18,24,30,36,42)
PAIRS=(('P00',0),('P00',1),('P01',0),('P01',1))
FIELDS=('C_map','f1','joint')
COLORS={'G':'#0072B2','S':'#D55E00'}
PINS={}


def sha(payload):return hashlib.sha256(payload).hexdigest()


def read(path,expected=None):
    payload=path.read_bytes();actual=sha(payload)
    if expected is not None and actual!=expected:raise ValueError('Input SHA changed: '+str(path))
    PINS[str(path.relative_to(ROOT))]=dict(sha256=actual,bytes=len(payload))
    return payload


def obj(path,expected=None):return json.loads(read(path,expected))


def load():
    previous=obj(COMPANION/'manifest.json')
    seal=obj(BASE/'seal.json',previous['source_seal_sha256'])
    result=obj(BASE/'result.json',previous['source_result_sha256'])
    assert result['status']=='completed' and result['checkpoint_count']==40 and result['original_independent_trajectories']==8
    raw=read(BASE/'checkpoint_metrics.csv',seal['checkpoint_metrics.csv'])
    rows=list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows)==40
    for row in rows:
        for key in ('case_index','hypothesis','paid_actions','saved_frames'):row[key]=int(row[key])
        for key in ('translation_m','C_map','precision','recall','f1','joint'):row[key]=float(row[key])
        path=ROOT/row['measurement_path'];measured=obj(path,seal[str(path.relative_to(BASE))])
        assert measured['contract']=='v34-area-surface-1' and measured['main_threshold_m']==.05
        assert measured['paid_actions']==row['paid_actions']
        assert measured['C_map']==row['C_map']
        for field in ('precision','recall','f1','joint'):assert measured['05cm'][field]==row[field]
        assert measured['main_joint']==row['joint'] and abs(row['joint']-row['C_map']*row['f1'])<1e-12
        assert all(math.isfinite(row[f]) and 0<=row[f]<=1 for f in FIELDS)
    expected={(p,h,m,t) for p,h in PAIRS for m in ('G','S') for t in STEPS}
    assert {(r['parent'],r['hypothesis'],r['mode'],r['paid_actions']) for r in rows}==expected
    def series(parent,h,mode):return sorted((r for r in rows if (r['parent'],r['hypothesis'],r['mode'])==(parent,h,mode)),key=lambda r:r['paid_actions'])
    for p,h in PAIRS:
        if h==0:
            assert all(g[f]==s[f] for g,s in zip(series(p,h,'G'),series(p,h,'S')) for f in FIELDS)
    p00g=series('P00',1,'G')[1];p00s=series('P00',1,'S')[1]
    assert p00s['f1']<p00g['f1'] and p00s['joint']<p00g['joint']
    return rows,raw,series


def main(output):
    if output.exists():raise ValueError('Exclusive output directory required')
    rows,original_csv,series=load()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.linewidth':.6,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.labelsize':8.5,'axes.titlesize':9,
        'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'article-submission-checkpoints-20260928'})
    fig,axes=plt.subplots(4,3,figsize=(7,6.55))
    fig.subplots_adjust(left=.107,right=.980,top=.887,bottom=.155,wspace=.27,hspace=.42)
    titles=(r'$C_{\mathrm{map}}$',r'$F_1$ @ 5 cm',r'$J_5=C_{\mathrm{map}}F_1$')
    plotted=[]
    for row_index,(parent,hypothesis) in enumerate(PAIRS):
        for column,field in enumerate(FIELDS):
            ax=axes[row_index,column]
            # Larger open S circles surround smaller G squares at exact ties:
            # both methods remain visible without perturbing any coordinates.
            for method in ('S','G'):
                values=series(parent,hypothesis,method)
                ax.plot(STEPS,[v[field] for v in values],color=COLORS[method],
                    linestyle='--' if method=='G' else '-',linewidth=1.15 if method=='G' else 1.5,
                    marker='s' if method=='G' else 'o',markersize=3.2 if method=='G' else 5.2,
                    markerfacecolor='white',markeredgewidth=.9,label=method,zorder=4 if method=='G' else 3)
                plotted.append(dict(parent=parent,hypothesis=hypothesis,method=method,field=field,
                    paid_actions=list(STEPS),values=[v[field] for v in values]))
            ax.set(xlim=(16.5,43.5),ylim=(0,1),xticks=STEPS,yticks=(0,.5,1))
            ax.set_yticklabels(('0','0.5','1'),fontsize=8)
            ax.tick_params(axis='both',labelsize=8,length=2.7,pad=2.8)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',color='#E5EBEF',linewidth=.6)
            ax.set_axisbelow(True)
            if row_index==0:ax.set_title(titles[column],fontsize=9.5,pad=7)
            if column==0:ax.set_ylabel(f'{parent} / h{hypothesis}',fontsize=9,weight='bold',labelpad=8)
            if row_index==3:ax.set_xlabel('Paid actions',fontsize=8.5,labelpad=4)
            if hypothesis==0 and column==0:
                ax.text(.06,.15,'G = S at all five stages',transform=ax.transAxes,fontsize=8,color='#52606A')
            if (parent,hypothesis)==('P00',1) and field in ('f1','joint'):
                delta=series(parent,hypothesis,'S')[1][field]-series(parent,hypothesis,'G')[1][field]
                ax.text(.045,.12,f't=24: S − G = {delta:.5f}',transform=ax.transAxes,fontsize=8,color='#52606A')
    fig.text(.107,.962,'Measured reconstruction checkpoints',fontsize=11,weight='bold',color='#22333D')
    fig.text(.107,.934,'Four paired conditions · eight saved V36 trajectories · 40 measured checkpoints',fontsize=8.5,color='#52606A')
    handles=[Line2D([],[],color=COLORS[m],ls='--' if m=='G' else '-',marker='s' if m=='G' else 'o',
        markerfacecolor='white',markersize=4,label=m) for m in ('G','S')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.535,.078),ncol=2,frameon=False,fontsize=8.5,
        handlelength=2.6,columnspacing=2)
    fig.text(.107,.054,'Markers: saved measurements. Connecting lines: visual guides only.',fontsize=8,color='#52606A')
    fig.text(.107,.032,'All stages are retained; intermediate checkpoints are not new independent tasks.',fontsize=8,color='#52606A')
    fig.canvas.draw()
    visible_text=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    minimum=min(t.get_fontsize() for t in visible_text)
    assert minimum>=8 and float(fig.get_size_inches()[0])==7
    assert len(plotted)==24 and sum(len(r['values']) for r in plotted)==120
    output.mkdir(parents=True)
    for ext in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(output/f'checkpoint_quality_submission.{ext}',dpi=350,facecolor='white',metadata=metadata)
    plt.close(fig)
    (output/'checkpoint_metrics_original.csv').write_bytes(original_csv)
    (output/'plotted_values.json').write_text(json.dumps(plotted,indent=2)+'\n')
    captions=dict(
        en='Coverage and reconstruction quality at all five measured checkpoints (18, 24, 30, 36, 42 paid actions) of each of the eight saved V36 trajectories. Four rows retain both layouts and both configurations; columns show C_map, surface F1 at 5 cm and J5=C_map×F1. G/S colours match the companion evidence figure. Both coincident h0 pairs and the early negative P00/h1 result at action24 are retained. Open markers show actual offline measurements; segments are visual guides, not measurements at additional times. All panels share the full 0–1 vertical range. Forty checkpoints are repeated measurements of eight trajectories, not forty independent trials; intermediate checkpoints do not imply terminal task qualification. The original five-column precision/recall figure and full scalar data remain available in the companion evidence package. Metrics retain their original v34-area-surface-1/public-ROI scope; this display does not substitute the later C_nav metric.',
        zh='对原V36全部八条保存轨迹，完整展示18、24、30、36、42付费动作处的40个既有离线实测检查点。四行保留两个布局及两种构型，三列为C_map、5cm表面F1和J5=C_map×F1，G/S颜色沿用原证据稿。两个h0持平条件及P00/h1第24步早期负差完整保留；圆形与方形空心标记显示真实测量，连接线仅为阅读引导，不产生额外时刻的测量。所有面板纵轴统一0–1，7英寸输出下全部可见文字标称字号至少8磅。40点是八条轨迹的重复测量，不能当作40个独立试验；中间状态不等于终态任务合格。含精确率和召回率的原五列完整版及全标量数据保留于伴随证据包；指标仍使用原v34-area-surface-1和公共ROI评价范围，未替换成后续C_nav。')
    (output/'captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2)+'\n')
    source=Path(__file__).resolve();read(source);(output/'source.py').write_bytes(source.read_bytes())
    files={p.name:dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in sorted(output.iterdir())}
    manifest=dict(schema='article.submission_checkpoint_figure.v1',figure_width_inches=7,figure_height_inches=6.55,
        minimum_visible_text_points=minimum,axes_tick_points=8,uniform_ylim=[0,1],columns=list(FIELDS),
        rows=[dict(parent=p,hypothesis=h) for p,h in PAIRS],paid_action_checkpoints=list(STEPS),
        source_trajectories=8,source_measurements=40,plotted_scalar_points=120,all_conditions_retained=True,
        coincident_h0_pairs_retained=True,early_negative_P00_h1_t24_retained=True,
        line_interpretation='visual guides only; no extra measurement or interpolation',colors=COLORS,
        companion_figure=str((COMPANION/'checkpoint_quality_components.pdf').relative_to(ROOT)),
        new_worlds=0,new_policy_runs=0,new_fusions=0,new_measurements=0,source_inputs=PINS,files=files)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    size=sum(p.stat().st_size for p in output.iterdir());assert size<2*1024**2
    print(json.dumps(dict(output=str(output),bytes=size,minimum_text_pt=minimum,measured_checkpoints=40)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
