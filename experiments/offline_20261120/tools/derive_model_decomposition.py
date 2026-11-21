"""Read-only derived attribution; weighted predictions are not physical area."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOTS={'v1':BASE/'dual_template_geometry_20261120','v2':BASE/'dual_template_instance_20261120'}
rows=[];sources=[]
for study,root in ROOTS.items():
    for directory in sorted((root/'replay').glob('*_attempt_001')):
        for p in sorted(directory.glob('*_predictions.json')):
            d=json.loads(p.read_text());a=np.load(p.with_suffix('.npz'))
            sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'npz_sha256':hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()})
            for mode in ('G','S'):
                obj=d['model_audits'][mode]['objects'];gid=a[mode+'_hypothesis_ids']//2
                chosen=max(d['routes'],key=lambda r:r['scores'][mode]);cid=chosen['candidate_id']
                gain=a[f'{mode}_gain_{cid}'];total=float(gain.sum())
                unknown=[i for i,o in enumerate(obj)if o['marked_points']==0]
                track0=[i for i,o in enumerate(obj)if o['support_track']==0]
                marked=[o for o in obj if o['marked_points']>0]
                unmarked=[o for o in obj if o['marked_points']==0]
                z=lambda os:[o.get('template_valid_probability_mass',1.) for o in os]
                rz,rk=z(unmarked),z(marked)
                rows.append(dict(study=study,state=directory.name.removesuffix('_attempt_001'),variant=d['variant'],model=mode,candidate=cid,
                    predicted_weighted_gain=total,unmarked_gain=float(gain[np.isin(gid,unknown)].sum()),track0_gain=float(gain[np.isin(gid,track0)].sum()),
                    unmarked_fraction=float(gain[np.isin(gid,unknown)].sum()/total)if total else None,
                    object_groups=len(obj),unmarked_groups=len(unknown),marked_groups=len(marked),
                    unmarked_Z_min=min(rz)if rz else None,unmarked_Z_max=max(rz)if rz else None,
                    marked_Z_min=min(rk)if rk else None,marked_Z_max=max(rk)if rk else None,
                    skipped_pooled_groups=len(d['model_audits'][mode].get('skipped_pooled_groups',[]))))
with (HERE/'model_decomposition.csv').open('w',newline='')as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(HERE/'model_decomposition.json').write_text(json.dumps({'scope':'all six fixed prefixes; per-variant own selected candidate; prediction-weight accounting only, not ground-truth area or independent samples','model_Z_meaning':'heuristic absolute consistency mass; not calibrated probability','rows':rows,'input_hashes':sources},indent=2)+'\n')
print(json.dumps({'rows':len(rows),'states':len(set(r['state']for r in rows)),'prediction_files':len(sources)}))
