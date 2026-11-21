#!/usr/bin/env python3
"""Reproduce the declared live unknown-coverage calibration from grid masks.

These derived masks contain no RGB pixels or depth images. Camera realization
retains the existing ExplorationObservedGainCalibration unknown-before gate;
using the bare vendor camera equation would describe a different variant.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

def reproduce(directory):
    directory=Path(directory)
    records=json.loads((directory/'records.json').read_text())
    alpha={'radar':1,'camera':1};beta={'radar':1,'camera':1}
    rows=[];files=[];new_known_union=None;new_camera_unknown_union=None;new_known_transition_sum=0
    for record in records:
        aid=record['action_id']
        if aid==0:continue
        before_path=directory/'masks'/Path(record['frozen_prediction_inputs']['path']).name
        after_path=directory/'masks'/Path(record['frozen_actual_feedback_inputs']['path']).name
        for path,key in [(before_path,'frozen_prediction_inputs'),(after_path,'frozen_actual_feedback_inputs')]:
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            assert sha==record[key]['sha256'],str(path)
            files.append(dict(path=str(path.relative_to(directory)),bytes=path.stat().st_size,sha256=sha))
        with np.load(before_path,allow_pickle=False) as z:before={k:z[k] for k in z.files}
        with np.load(after_path,allow_pickle=False) as z:after={k:z[k] for k in z.files}
        assert set(before)=={'radar_prediction','camera_prediction','known_before','camera_before'}
        assert set(after)=={'belief_after','camera_seen_after'}
        shape=before['known_before'].shape
        assert len(shape)==2
        assert all(v.shape==shape and v.dtype==np.bool_ for v in before.values())
        assert after['camera_seen_after'].shape==shape and after['camera_seen_after'].dtype==np.bool_
        assert after['belief_after'].shape==shape and after['belief_after'].dtype==np.int8
        assert np.isin(after['belief_after'],[-1,0,1]).all()
        actual={'radar':(after['belief_after']!=-1)&~before['known_before'],
                'camera':after['camera_seen_after']&~before['camera_before']&~before['known_before']}
        if new_known_union is None:
            new_known_union=np.zeros(shape,bool);new_camera_unknown_union=np.zeros(shape,bool)
        new_known_union|=actual['radar'];new_camera_unknown_union|=actual['camera']
        new_known_transition_sum+=int(actual['radar'].sum())
        outcome=record['original_prediction_and_actual_feedback'];event=outcome['actual_gain_event']
        for channel in ['radar','camera']:
            prediction=before[channel+'_prediction'];realized=actual[channel]
            counts=dict(predicted_cells=int(prediction.sum()),
                realized_predicted_cells=int(np.count_nonzero(prediction&realized)),
                unrealized_predicted_cells=int(np.count_nonzero(prediction&~realized)),
                unpredicted_realized_cells=int(np.count_nonzero(realized&~prediction)))
            assert counts==event[channel],(aid,channel,counts,event[channel])
            if event['feedback_consumed_for_planning']:
                alpha[channel]+=counts['realized_predicted_cells']
                beta[channel]+=counts['unrealized_predicted_cells']
            posterior=alpha[channel]/(alpha[channel]+beta[channel])
            assert posterior==event['posterior_after'][channel],(aid,channel)
            assert alpha[channel]==outcome['gain_snapshot']['alpha'][channel]
            assert beta[channel]==outcome['gain_snapshot']['beta'][channel]
            rows.append(dict(action_id=aid,channel=channel,**counts,alpha=alpha[channel],beta=beta[channel],
                posterior=posterior,ros_status=record['ros_status'],primitive_arrival_verified=outcome['primitive_arrival_verified']))
    result=dict(schema='actual_pilot_grid_mask_reproduction/1',all_logged_counts_and_posteriors_exact=True,
        scope='Reproduction of derived actual-controller grid feedback; not an additional robot trial or algorithm efficacy comparison',
        camera_eligibility='camera_seen_after AND NOT camera_before AND NOT known_before (declared unknown-coverage variant)',
        radar_eligibility='belief_after != -1 AND NOT known_before',
        raw_RGBD_included=False,independent_samples_added=0,files=files,rows=rows,
        unique_internal_unknown_to_known_grid_cells=0 if new_known_union is None else int(new_known_union.sum()),
        per_step_unknown_to_known_transition_sum=new_known_transition_sum,
        unique_internal_camera_unknown_grid_cells=0 if new_camera_unknown_union is None else int(new_camera_unknown_union.sum()),
        coverage_scope='Internal frozen-epoch grid support; no surveyed physical area or complete-room coverage is inferred',
        final_alpha=alpha,final_beta=beta,final_posterior={k:alpha[k]/(alpha[k]+beta[k]) for k in alpha})
    (directory/'mask_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    if rows:
        with (directory/'mask_reproduction.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    result=reproduce(parser.parse_args().directory)
    print(json.dumps({k:result[k] for k in ['all_logged_counts_and_posteriors_exact','final_alpha','final_beta','final_posterior']},indent=2))
