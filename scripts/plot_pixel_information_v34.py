#!/usr/bin/env python3
"""Display existing V34 pixel arrays; no new sensing or quality evaluation."""
import io
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/nso_v34_mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import ROOT,read,sha,write,write_bytes,freeze,verify_sources,verify_inventory,seal


def main():
    source=ROOT/'audit_results/v34_pixel_information_20260918'
    review=ROOT/'audit_results/v34_pixel_information_review_20260918'
    verify_sources(source);verify_inventory(source);verify_inventory(review)
    verified=read(review/'result.json')
    if not verified['saved_evidence_verified'] or verified['source_result_sha256']!=sha(source/'result.json'):
        raise ValueError('independent review is not bound to this source')
    out=ROOT/'audit_results/v34_pixel_information_figures_20260918'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    inputs={str(p.relative_to(ROOT)):sha(p) for folder in (source,review) for p in folder.iterdir() if p.is_file()}
    freeze(out,[Path(__file__)],input_sha256=inputs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','svg.hashsalt':'v34-pixels'})
    fig,axes=plt.subplots(2,4,figsize=(12.5,5.6),constrained_layout=True)
    rows=[]
    for row,parent in enumerate(('P00','P01')):
        meta=read(source/(parent+'_observations.json'))
        old=read(ROOT/f'audit_results/v33_direction_information_r1_20260917/{parent}_geometry.json')
        anchor=old['tables'][0]['anchor'];first=meta['pair']['first_information_witnesses'][0]['node']
        arrays=[]
        for h in (0,1):
            with np.load(source/f'{parent}_h{h}_pixels.npz',allow_pickle=False) as z:
                arrays.append(dict(rgb=z['rgb'][anchor],depth=z['depth'][first],initial_depth=z['depth'][anchor]))
        assert np.array_equal(arrays[0]['initial_depth'],arrays[1]['initial_depth'])
        count=int(np.count_nonzero(arrays[0]['depth']!=arrays[1]['depth']))
        for h in (0,1):
            axes[row,h].imshow(arrays[h]['rgb'])
            axes[row,h].set_title(f'{parent} initial RGB | type {"AB"[h]}')
            im=axes[row,h+2].imshow(arrays[h]['depth'],vmin=0,vmax=4,cmap='viridis')
            axes[row,h+2].set_title(f'{parent} first geometry view | h{h}')
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        axes[row,0].set_ylabel(parent)
        layer=meta['pair']['first_information_action_layer']
        axes[row,2].set_xlabel(f'First reachable difference: {layer} actions')
        axes[row,3].set_xlabel(f'{count} differing depth pixels at this view')
        rows.append(dict(parent=parent,initial_node=anchor,first_node=first,first_layer=layer,different_depth_pixels=count))
    fig.colorbar(im,ax=axes[:,2:],label='Axial depth (m; 0 means invalid)',shrink=.75)
    fig.suptitle('V34 saved sensor evidence: class colour precedes geometric distinction',fontsize=13)
    fig.supxlabel('Procedural clean RGB-D, 96 x 72; camera Euclidean range 4 m. No new rendering, TSDF or quality score.',fontsize=9)
    figures={}
    for ext in ('png','svg'):
        file=ROOT/f'docs/research/figures/v34_pixel_information_20260918.{ext}'
        buffer=io.BytesIO();fig.savefig(buffer,format=ext,dpi=160)
        write_bytes(out,file,buffer.getvalue());figures[str(file.relative_to(ROOT))]=sha(file)
    plt.close(fig)
    for rel,want in inputs.items():
        if sha(ROOT/rel)!=want:raise ValueError('source changed')
    verify_sources(out)
    write(out,out/'result.json',dict(status='complete',figures=figures,rows=rows,
        new_worlds=0,new_sensor_queries=0,new_DP=0,new_metric_evaluations=0,new_main_tasks=0))
    seal(out);print(read(out/'result.json'))


if __name__=='__main__':main()
