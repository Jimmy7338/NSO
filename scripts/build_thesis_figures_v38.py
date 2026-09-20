#!/usr/bin/env python3
"""Publication figures from saved evidence only; never import a simulator/planner.

Run with .venv-3d/bin/python. Refuse altered numerical sources. Figures are
editable PDF/SVG plus 300 dpi PNG. CSV rows preserve the complete source values.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso_v38_mpl')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'docs/thesis/V38_EVIDENCE_MAP_20260920.json'
COL = {'G': '#737984', 'S': '#0072B2', 'swapped': '#E69F00',
       'swapped_no_feedback': '#D55E00'}
LABEL = {'G': 'G: active geometry', 'S': 'S: semantic',
         'swapped': 'X: swapped class', 'swapped_no_feedback': 'Xnf: no correction'}
MARK = {'G': 'o', 'S': 's', 'swapped': '^', 'swapped_no_feedback': 'D'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_refs(value, checked):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            p = ROOT / value['path']
            assert digest(p) == value['sha256'], f'Changed source: {p}'
            checked[value['path']] = value['sha256']
        for v in value.values():
            verify_refs(v, checked)
    elif isinstance(value, list):
        for v in value:
            verify_refs(v, checked)


def style():
    plt.rcParams.update({
        'font.family': 'Liberation Sans', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': .65, 'lines.linewidth': 1.25,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'nso-publication-v38', 'savefig.facecolor': 'white',
        'mathtext.fontset': 'dejavusans',
    })


def axis(ax, letter, title, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color='#E6E8EB', linewidth=.5)
    ax.set_title(title, loc='left', pad=10)
    ax.text(-.14, 1.06, letter, transform=ax.transAxes, fontweight='bold', fontsize=11)


def save(fig, out, name):
    for ext in ('pdf', 'svg', 'png'):
        meta = {'Creator': 'NSO saved-evidence figure builder V38'} if ext == 'pdf' else None
        fig.savefig(out / f'{name}.{ext}', dpi=300, metadata=meta)
    plt.close(fig)


def architecture(out):
    fig, ax = plt.subplots(figsize=(7.05, 3.45))
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.02)
    ax.set(xlim=(0, 10), ylim=(0, 5)); ax.axis('off')
    def box(x, y, w, h, title, detail, edge='#737984', face='#F5F6F8'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.07',
                                   facecolor=face, edgecolor=edge, linewidth=.8))
        ax.text(x+w/2, y+h*.70, title, ha='center', va='center', weight='bold', fontsize=8.5)
        ax.text(x+w/2, y+h*.32, detail, ha='center', va='center', fontsize=7.5, linespacing=1.4)
    def arrow(a, b, color='#4C535C', **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9,
                                   linewidth=.8, color=color, **kw))
    box(.1, 3.05, 2.05, 1.05, 'Paid observations', 'RGB-D / scan / exact pose')
    box(2.65, 3.05, 2.55, 1.05, 'OV-SDF interface', 'Class prior + observed state', '#0072B2', '#F0F7FB')
    box(5.70, 3.05, 2.5, 1.05, 'STGHP interface', 'Posterior-weighted view choice', '#0072B2', '#F0F7FB')
    box(5.70, 1.35, 2.5, 1.05, 'RPN-UQ interface', 'Graph edge + return budget')
    box(2.65, 1.35, 2.55, 1.05, 'IGCR interface', 'Measured residual feedback', '#E69F00', '#FFFAF0')
    box(.1, 1.35, 2.05, 1.05, 'Shared CPU TSDF', 'Measured depth only')
    box(8.60, 1.35, 1.25, 1.05, 'Execute', '1 m / 90 deg')
    for a,b in [((2.15,3.58),(2.65,3.58)),((5.20,3.58),(5.70,3.58)),
                ((6.95,3.05),(6.95,2.40)),((8.20,1.88),(8.60,1.88)),
                ((3.92,2.40),(3.92,3.05)),((1.12,3.05),(1.12,2.40))]: arrow(a,b)
    arrow((2.0,3.05),(2.95,2.40), color='#E69F00')
    ax.plot([9.22,9.22,1.12],[2.40,4.42,4.42],color='#4C535C',lw=.8)
    arrow((1.12,4.42),(1.12,4.10))
    ax.text(5.15,4.65,'Next paid observation',ha='center',fontsize=8)
    ax.text(8.42,3.55,'Global\nlayer',va='center',color='#4C535C',fontsize=8)
    ax.text(7.55,1.05,'Local guard and execution',ha='center',fontsize=8)
    ax.plot([.1,9.85],[.76,.76],color='#ADB2B9',lw=.7,ls='--')
    ax.text(.12,.46,'After sealing:',fontsize=8,weight='bold')
    ax.text(2.05,.46,'stored mesh + reference → coverage, precision, recall, F1, joint score',fontsize=8)
    ax.text(.12,.10,'Shared public templates and safe graph enter planning; evaluation scores never enter the controller.',fontsize=7.3)
    save(fig,out,'fig01_architecture')


def confirmation(cohort, out):
    rows={(m['parent'],m['hypothesis'],m['method_id']):m for m in cohort['methods']}
    pairs=[('P00',0),('P00',1),('P01',0),('P01',1)]
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.95),gridspec_kw={'width_ratios':[1.15,1]})
    fig.subplots_adjust(left=.075,right=.98,bottom=.21,top=.80,wspace=.36)
    x=np.arange(4)
    for j,m in enumerate(('G','S')):
        vals=[rows[p,h,m]['primary'] for p,h in pairs]
        axs[0].plot(x+(j-.5)*.13,vals,ls='none',marker=MARK[m],ms=5,color=COL[m],label=LABEL[m])
    for i,(p,h) in enumerate(pairs):
        a,b=(rows[p,h,m]['primary'] for m in ('G','S'))
        axs[0].plot([i-.065,i+.065],[a,b],color='#B3B8BE',lw=.8,zorder=0)
        axs[0].text(i,max(a,b)+.052,f'{b-a:+.4f}',ha='center',fontsize=7.3)
    axs[0].set_xticks(x,['P00\nh0','P00\nh1','P01\nh0','P01\nh1'])
    axs[0].set(ylim=(0,1),ylabel=r'Terminal $J_5=C_{map}F_{1,5cm}$')
    axis(axs[0],'a','All paired configurations')
    axs[0].legend(loc='upper left',bbox_to_anchor=(-.01,1.30),ncol=2,frameon=False,handletextpad=.4,columnspacing=1)
    for j,(p,mk,co) in enumerate([('P00','o','#0072B2'),('P01','^','#D55E00')]):
        effect=[]
        for t in ('02cm','05cm','10cm'):
            effect.append(np.mean([rows[p,h,'S']['thresholds'][t]['joint']-rows[p,h,'G']['thresholds'][t]['joint'] for h in (0,1)]))
        axs[1].plot(np.arange(3)+(j-.5)*.04,effect,marker=mk,color=co,label=p,ls='-' if j==0 else '--',ms=4.5)
    axs[1].axhline(0,color='#626971',lw=.75)
    axs[1].set_xticks(range(3),['2 cm','5 cm\n(primary)','10 cm'])
    axs[1].set(ylim=(-.005,.06),yticks=[0,.02,.04,.06],ylabel=r'Paired mean $J_S-J_G$')
    axis(axs[1],'b','Distance-threshold sensitivity')
    axs[1].legend(frameon=False,loc='lower right')
    save(fig,out,'fig02_confirmation')


def ablation(cohort,out):
    rows={(m['hypothesis'],m['method_id']):m for m in cohort['methods']}
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.95))
    fig.subplots_adjust(left=.075,right=.98,bottom=.2,top=.81,wspace=.37)
    for j,m in enumerate(COL):
        axs[0].plot(np.arange(2)+(j-1.5)*.09,[rows[h,m]['primary'] for h in (0,1)],
                    color=COL[m],marker=MARK[m],ls='none',ms=4.8,label=LABEL[m])
    axs[0].set_xticks([0,1],['h0','h1']); axs[0].set(xlim=(-.45,1.45),ylim=(0,1),ylabel=r'Terminal $J_5$')
    axis(axs[0],'a','Four controlled conditions (P00)')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,1.02),ncol=2,frameon=False,columnspacing=1.5)
    for h,mk,co,ls in [(0,'o','#0072B2','-'),(1,'^','#D55E00','--')]:
        y=[rows[h,'swapped']['thresholds'][t]['joint']-rows[h,'swapped_no_feedback']['thresholds'][t]['joint'] for t in ('02cm','05cm','10cm')]
        axs[1].plot(range(3),y,marker=mk,ms=4.5,color=co,ls=ls,label=f'h{h}')
        if h==1:
            axs[1].annotate(f'{y[0]:+.4f}',(0,y[0]),xytext=(.25,-.009),fontsize=7.3,
                            arrowprops={'arrowstyle':'-','lw':.6})
    axs[1].axhline(0,color='#626971',lw=.75)
    axs[1].set_xticks(range(3),['2 cm','5 cm\n(primary)','10 cm'])
    axs[1].set(ylim=(-.012,.032),ylabel=r'Correction effect $J_X-J_{Xnf}$')
    axis(axs[1],'b','Wrong-prior correction policy')
    axs[1].legend(frameon=False,loc='upper left')
    save(fig,out,'fig03_ablation')


def learned(cohort,stress,out):
    keys=['learned_geometry','fixed_semantic','learned_swapped']
    names=['Learned geometry','Fixed semantic','Permuted semantic']
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.05),gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.185,right=.98,bottom=.24,top=.82,wspace=.6)
    for i,(key,col) in enumerate(zip(keys,['#737984','#0072B2','#D55E00'])):
        c=cohort['reported_context_inference'][key]
        vals=[v['joint_gain'] for v in c['context_differences'].values()]
        # Deterministic offsets display all eight parent contexts; no artificial noise.
        axs[0].scatter(vals,i+np.linspace(-.17,.17,len(vals)),color=col,s=14,alpha=.75,zorder=2)
        lo,hi=c['bootstrap_95_percentile_interval']; mean=c['mean_difference']
        axs[0].errorbar(mean,i+.27,xerr=[[mean-lo],[hi-mean]],fmt='D',color=col,ms=4,capsize=2,elinewidth=1.2)
    axs[0].axvline(0,color='#626971',lw=.7)
    axs[0].set_yticks(range(3),names); axs[0].set(xlim=(-.04,.8),ylim=(-.5,2.65),xlabel=r'Semantic minus comparator in $\Delta(AF_1)$ (m$^2$)')
    axs[0].invert_yaxis(); axis(axs[0],'a','Learned-head confirmation',grid='x')
    axs[0].text(0,-.36,'Dots: 8 parent contexts; diamonds: mean + saved 95% CI',transform=axs[0].transAxes,fontsize=7.1)
    for i,(c,col) in enumerate(zip([cohort,stress],['#0072B2','#737984'])):
        row=next(x for x in c['contrasts'] if x['baseline']=='learned_geometry')
        axs[1].bar(i,row['absolute_effect'],width=.45,color=col)
        axs[1].text(i,row['absolute_effect']+(.018 if i==0 else -.025),f"{row['absolute_effect']:+.4f}",ha='center',va='bottom' if i==0 else 'top',fontsize=8)
    axs[1].axhline(0,color='#626971',lw=.7)
    axs[1].set_xticks([0,1],['Near-complete\ncoverage','Coverage\npressure'])
    axs[1].set(ylim=(-.10,.45),ylabel=r'Mean $\Delta(AF_1)_S-\Delta(AF_1)_G$ (m$^2$)')
    axis(axs[1],'b','Benefit depends on task phase')
    save(fig,out,'fig04_learned_evidence')


def timeline(out,source):
    # Added only if the raw-log audit is available; no invented trajectory.
    if not source.exists(): return {}
    d=json.loads(source.read_text())
    # Contract supplied by the separately audited timeline extractor.
    checked={}
    for relative,record in d['sources'].items():
        assert digest(ROOT/relative)==record['sha256'], relative
        checked[relative]=record['sha256']
    rows=[]
    for series in d['series']:
        if series['phase']!='V35' or series['parent']!='P00' or series['hypothesis']!=1:
            continue
        for r in series['rows']:
            rows.append({'mode':series['mode_in_source'],
                         'observation_step':r['observation_after_executed_action'],
                         'probability_h1':r['posterior_h1_after_observation']})
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.8),sharey=True)
    fig.subplots_adjust(left=.075,right=.98,bottom=.21,top=.80,wspace=.20)
    for ax,modes,letter,title in [(axs[0],['G','S'],'a','Early semantic choice'),(axs[1],['swapped','swapped_no_feedback'],'b','Correction of a wrong class')]:
        for m in modes:
            seq=sorted([r for r in rows if r['mode']==m],key=lambda r:r['observation_step'])
            ax.step([r['observation_step'] for r in seq],[r['probability_h1'] for r in seq],where='post',
                    color=COL[m],label=LABEL[m],ls='-' if m in ('S','swapped') else '--',lw=1.4)
        ax.set(xlim=(0,42),ylim=(-.04,1.04),xlabel='Observation index (before next action)')
        axis(ax,letter,title)
    handles=[]; labels=[]
    for ax in axs:
        h,l=ax.get_legend_handles_labels(); handles.extend(h); labels.extend(l)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,.995),
               ncol=4,frameon=False,fontsize=7.1,handlelength=1.8,columnspacing=1.1)
    axs[0].set_ylabel(r'Posterior weight $p(h=1)$')
    for ax,t,txt in [(axs[0],18,'Decision 18\naction 19'),(axs[1],28,'Decision 28\naction 29')]:
        ax.axvline(t,color='#80868D',lw=.7,ls=':')
        ax.text(t+.7,.43,txt,fontsize=7.3)
    save(fig,out,'fig05_causal_timeline')
    return checked


def tables(cohorts,out):
    v36=cohorts['v36_online']; v35=cohorts['v35_online']
    all_rows=[]
    for c in (v35,v36):
        for m in c['methods']:
            for t,d in m['thresholds'].items():
                all_rows.append(dict(cohort=c['cohort_id'],parent=m['parent'],hypothesis=m['hypothesis'],
                                     method=m['method_id'],threshold=t,C_map=m['C_map'],**d))
    with (out/'online_measurements.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    headers={
        'table_confirmation_rows.tex': ('lllrrrrr', r'布局 & 构型 & 方法 & $C_{\mathrm{map}}$ (\%) & $P_5$ & $R_5$ & $F_{1,5}$ & $J_5$'),
        'table_ablation_rows.tex': ('lrrrr', r'条件 & $J_5(h_0)$ & $J_5(h_1)$ & 平均$F_{1,5}$ & 平均$J_5$'),
        'table_learned_rows.tex': ('lrrrr', r'对照 & 对照均值 & 绝对差 & 相对差 & 差的95\%区间'),
    }
    def write(name,lines):
        spec,header=headers[name]
        content=[r'\begin{tabular}{'+spec+'}',r'\toprule',header+r'\\\midrule',*lines,r'\bottomrule',r'\end{tabular}']
        (out/name).write_text('\n'.join(content)+'\n')
    lines=[]
    for m in v36['methods']:
        q=m['thresholds']['05cm']
        lines.append(f"{m['parent']} & $h_{m['hypothesis']}$ & {m['method_id']} & {100*m['C_map']:.2f} & {q['precision']:.4f} & {q['recall']:.4f} & {q['f1']:.4f} & {q['joint']:.4f} \\\\")
    write('table_confirmation_rows.tex',lines)
    lines=[]
    for mode in COL:
        ms=[m for m in v35['methods'] if m['method_id']==mode]
        q=[m['thresholds']['05cm'] for m in ms]
        name={'swapped':'X','swapped_no_feedback':'Xnf'}.get(mode,mode)
        lines.append(f"{name} & {q[0]['joint']:.4f} & {q[1]['joint']:.4f} & {np.mean([r['f1'] for r in q]):.4f} & {np.mean([r['joint'] for r in q]):.4f} \\\\")
    write('table_ablation_rows.tex',lines)
    c=cohorts['v11_1_confirmation2_efficacy']; lines=[]
    for key,name in [('learned_geometry','同容量几何'),('fixed_semantic','固定语义'),('learned_swapped','类别置换')]:
        r=c['reported_context_inference'][key]; lo,hi=r['bootstrap_95_percentile_interval']
        lines.append(f"{name} & {r['baseline_mean']:.4f} & {r['mean_difference']:.4f} & {100*r['relative_to_baseline']:.2f}\\% & [{lo:.4f}, {hi:.4f}] \\\\")
    write('table_learned_rows.tex',lines)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'docs/thesis/figures/v38')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    d=json.loads(MAP.read_text()); checked={}; verify_refs(d,checked)
    cohorts={c['cohort_id']:c for c in d['cohorts']}; style()
    architecture(a.output); confirmation(cohorts['v36_online'],a.output)
    ablation(cohorts['v35_online'],a.output)
    learned(cohorts['v11_1_confirmation2_efficacy'],cohorts['v11_1_confirmation2_stress'],a.output)
    source=ROOT/'docs/thesis/V38_CAUSAL_TIMELINE_20260920.json'
    checked.update(timeline(a.output,source)); tables(cohorts,a.output)
    if source.exists(): checked[str(source.relative_to(ROOT))]=digest(source)
    checked[str(MAP.relative_to(ROOT))]=digest(MAP)
    checked[str(Path(__file__).resolve().relative_to(ROOT))]=digest(Path(__file__))
    manifest={'source_sha256':checked,'outputs':{p.name:digest(p) for p in sorted(a.output.iterdir()) if p.name!='manifest.json'},
              'scope':'Saved scores and recorded beliefs only; zero new World, planner, fusion, evaluation, inference or significance test',
              'style':{'width_mm':179.07,'fonts':'Liberation Sans; PDF TrueType embedded; SVG text editable','raster_dpi':300,
                       'color_vision':'Okabe-Ito hues with distinct markers/linestyles; all zero and negative effects retained',
                       'uncertainty':'Only V11.1 original eight-parent bootstrap intervals; no V35/V36 population CI'},
              'execution_accounting':d['execution_accounting']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figures':len(list(a.output.glob('*.pdf'))),'verified_source_files':len(checked),
                      'output_bytes':sum(p.stat().st_size for p in a.output.iterdir()),'output':str(a.output)},indent=2))


if __name__=='__main__': main()
