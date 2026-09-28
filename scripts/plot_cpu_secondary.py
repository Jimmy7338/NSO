#!/usr/bin/env python3
"""Descriptive secondary figures; no method selection or significance testing."""
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    folder=Path(folder)
    summary=json.loads((folder/'formal_summary.json').read_text())
    config=json.loads((folder/'config.json').read_text())
    episodes=[json.loads(s) for s in (folder/'episodes.jsonl').read_text().splitlines()]
    layouts=list(dict.fromkeys(e['layout'] for e in episodes))
    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    for layout,ax in zip(layouts,axes.flat):
        first=next(e for e in episodes if e['layout']==layout)
        for method in config['methods']:
            e=next(e for e in episodes if e['map_id']==first['map_id'] and e['start_seed']==first['start_seed'] and e['method']==method)
            with (folder/e['artifact_dir']/'steps.csv').open() as f:rows=list(csv.DictReader(f))
            ax.plot([float(r['path_length_m']) for r in rows],[100*float(r['coverage_ratio']) for r in rows],label=method,linewidth=1)
        ax.set(title=first['map_id'],xlabel='Travelled distance (m)',ylabel='Coverage (%)',ylim=(0,100));ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    fig.suptitle('First prespecified map/start of each layout | actual paths, no extrapolation')
    fig.tight_layout(rect=(0,.16,1,.96));fig.savefig(folder/'coverage_path_examples.png',dpi=150);plt.close(fig)
    methods=[m for m in config['methods'] if any('controller_attempts' in e and e['method']==m for e in episodes)]
    fractions={}
    for method in methods:
        values=[];counts={'reached':0,'failed':0,'censored':0}
        for map_id in {e['map_id'] for e in episodes}:
            starts=[]
            for e in episodes:
                if e['map_id']!=map_id or e['method']!=method:continue
                rows=[json.loads(s) for s in (folder/e['artifact_dir']/'goal_attempts.jsonl').read_text().splitlines()]
                n=len(rows)
                observed=[sum(r['controller_success']==1 for r in rows),sum(r['controller_success']==0 for r in rows),sum(r['controller_success'] is None for r in rows)]
                for k,v in zip(counts,observed):counts[k]+=v
                if n:starts.append(np.asarray(observed)/n)
            if starts:values.append(np.mean(starts,axis=0))
        fractions[method]={'map_mean_fractions':np.mean(values,axis=0).tolist(),'pooled_counts_descriptive_only':counts}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.5))
    left=np.zeros(len(methods))
    for i,label in enumerate(('Reached','Failed (collision/timeout)','Censored')):
        values=np.array([fractions[m]['map_mean_fractions'][i]*100 for m in methods])
        ax.barh(methods,values,left=left,label=label);left+=values
    ax.set(xlabel='Share of all selected goals (%)',xlim=(0,100));ax.tick_params(axis='y',labelsize=8);ax.legend(fontsize=8,loc='lower right')
    bx.barh(config['methods'],[100*summary['means'][m]['revisit_ratio'] for m in config['methods']])
    bx.set(xlabel='Repeated successful moves (%)');bx.tick_params(axis='y',labelsize=8)
    fig.suptitle('Secondary diagnostics | mean over starts, then maps; censored goals are not failures')
    fig.tight_layout();fig.savefig(folder/'goal_outcomes_revisits.png',dpi=150);plt.close(fig)
    (folder/'goal_outcome_summary.json').write_text(json.dumps(fractions,indent=2)+'\n')
    (folder/'secondary_figures_metadata.json').write_text(json.dumps(dict(
        scope='descriptive secondary plots; not an additional confirmatory test',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        formal_summary_sha256=hashlib.sha256((folder/'formal_summary.json').read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);plot(p.parse_args().folder)
