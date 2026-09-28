#!/usr/bin/env python3
"""Static V35 scientific figures from sealed measured results only."""
import argparse
import io
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/nso_v35_matplotlib')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import (ROOT,read,sha,write,write_bytes,freeze,seal,verify_sources)


def run():
    import importlib.util
    spec=importlib.util.spec_from_file_location('v35_collector',ROOT/'scripts/run_online_routes_v35.py')
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    batch=collector.OUTPUT
    collector.frozen();collector.verify_seal(batch,'final_seal.json')
    config=read(batch/'config.json')
    result=read(batch/'result.json')
    out=ROOT/'audit_results/v35_online_semantic_figures_20260918'
    if out.exists():raise FileExistsError('no overwrite')
    out.mkdir()
    inputs={str(p.relative_to(ROOT)):sha(p) for p in batch.rglob('*') if p.is_file()}
    freeze(out,[Path(__file__)],input_sha256=inputs,
        scope='saved measured scores only; no new World, controller, TSDF or Q')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'nso-v35'})
    modes=('G','S','swapped','swapped_no_feedback')
    labels=('Geometry','Semantic','Wrong class\n+ feedback','Wrong class\nno feedback')
    rows=[]
    for folder in collector.case_directories():
        if not (folder/'result.json').exists():continue
        d=read(folder/'result.json');case=d['physical_case']
        rows.append(dict(index=case['index'],hypothesis=case['hypothesis'],mode=case['mode'],
            final=d['stages']['final']['measurement']))
    scores={(r['hypothesis'],r['mode']):r['final'] for r in rows}
    write(out,out/'plot_data.json',dict(rows=rows,main_result_sha256=sha(batch/'result.json')))
    figures={}
    def save(fig,name):
        for ext in ('png','svg'):
            path=ROOT/'docs/research/figures'/f'v35_{name}_20260918.{ext}'
            buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=160)
            write_bytes(out,path,buf.getvalue());figures[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.8),constrained_layout=True)
    x=np.arange(4);width=.36
    for ax,key,title in zip(axes,('C_map','f1','joint'),('Actual 2D coverage','Surface F1 at 5 cm','Joint score at 5 cm')):
        for h,color in ((0,'#3274a1'),(1,'#df8a32')):
            values=[(scores[h,m][key] if key=='C_map' else scores[h,m]['05cm'][key])
                if (h,m) in scores else np.nan for m in modes]
            bars=ax.bar(x+(h-.5)*width,values,width,label=f'h{h}',color=color)
            for bar,value in zip(bars,values):
                if np.isfinite(value):ax.text(bar.get_x()+width/2,value+.012,f'{value:.3f}',ha='center',fontsize=8)
        if key=='C_map':ax.axhline(.8,color='black',linestyle='--',linewidth=1,label='C80 requirement')
        ax.set_xticks(x,labels,fontsize=8);ax.set_ylim(0,1.08);ax.set_title(title)
        ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle('V35 online CPU four-module development | paired P00 configurations',fontsize=14)
    fig.supxlabel('Same public templates, action budget, noisy depth and TSDF; artificial RGB classes, exact pose. One parent, not independent scene confirmation.',fontsize=9)
    save(fig,'online_endpoints')
    if len(scores)==8:
        fig,ax=plt.subplots(figsize=(9.8,4.8),constrained_layout=True)
        thresholds=('02cm','05cm','10cm');xx=np.arange(3);width=.2
        for i,(mode,label,color) in enumerate(zip(modes,labels,('#3274a1','#2d9755','#df8a32','#a255a7'))):
            values=[sum(scores[h,mode][t]['joint'] for h in (0,1))/2 for t in thresholds]
            ax.bar(xx+(i-1.5)*width,values,width,label=label.replace('\n',' '),color=color)
        ax.set_xticks(xx,['2 cm (secondary)','5 cm (primary)','10 cm (secondary)'])
        ax.set_ylabel('Equal-weight mean C_map × F1');ax.set_ylim(0,1.02)
        ax.set_title('V35 prescribed threshold comparison');ax.legend(fontsize=8,ncols=2)
        ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
        fig.supxlabel('Both paired configurations retained; deterministic replays do not increase the sample size.',fontsize=9)
        save(fig,'online_thresholds')
    collector.verify_seal(batch,'final_seal.json');verify_sources(out)
    for rel,want in inputs.items():
        if sha(ROOT/rel)!=want:raise ValueError('source evidence changed')
    write(out,out/'result.json',dict(status='complete',figures=figures,measured_cells=len(rows),
        new_worlds=0,new_controller_calls=0,new_sensor_packets=0,new_TSDF_integrations=0,
        new_quality_evaluations=0,new_main_tasks=0))
    seal(out);print(read(out/'result.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.run:run()
    else:p.print_help()
