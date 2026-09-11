#!/usr/bin/env python3
"""Descriptive analysis of the two prespecified scene scales."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_targeted_scenes import interval


def analyze(folder):
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text())
    if summary['audit']['episodes_reconstructed']!=summary['episodes']:raise ValueError('unaudited suite')
    records=[json.loads(s) for s in (folder/'episodes.jsonl').read_text().splitlines()]
    report={}
    for size in sorted({r['size'] for r in records}):
        rows=[r for r in records if r['size']==size]
        methods={case:{r['map_id']:r for r in rows if r['case']==case} for case in dict.fromkeys(r['case'] for r in rows)}
        values={case:{k:float(np.mean([r[k] for r in bymap.values()])) for k in ('coverage_auc','coverage_ratio','productive_area_coverage')} for case,bymap in methods.items()}
        diffs=[methods['combined_aligned'][m]['coverage_auc']-r['coverage_auc'] for m,r in methods['geometry'].items()]
        checkpoints=[100,200,400,600];times={}
        for case in ('geometry','semantic_aligned','combined_aligned'):
            observed=[]
            for r in methods[case].values():
                with (folder/r['artifact_dir']/'steps.csv').open() as f:steps=list(csv.DictReader(f))
                observed.append(np.interp(checkpoints,[int(s['steps']) for s in steps],[float(s['coverage_ratio']) for s in steps]))
            times[case]=np.mean(observed,axis=0).tolist()
        report[str(size)]=dict(means=values,combined_minus_geometry_auc=interval(diffs),
            paired_map_differences=dict(zip(methods['geometry'],diffs)),checkpoints=checkpoints,
            coverage_at_checkpoints=times,relative_mean_auc_gain=values['combined_aligned']['coverage_auc']/values['geometry']['coverage_auc']-1)
    (folder/'scale_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);analyze(p.parse_args().folder)
