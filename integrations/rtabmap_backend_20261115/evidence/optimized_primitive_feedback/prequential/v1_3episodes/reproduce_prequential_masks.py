#!/usr/bin/env python3
"""Portable NumPy-only frozen-mask verification, no raw sensors/models/ROS.

Local full-artifact verification includes raw originals that are deliberately
not distributed. This verifier checks the distributed derived masks, method
source hashes, original pre-action chronology and all saved calibration events
and metrics. It does not independently verify absent raw archives.
"""
import argparse,ast,hashlib,json,math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import evaluate_frozen_gain_original as definitions


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reproduce(directory):
    root=Path(directory);index=json.loads((root/'portable_manifest.json').read_text())
    expected=json.loads((root/'results.json').read_text())
    namespace=dict(np=np,deepcopy=deepcopy,hashlib=hashlib)
    for source in index['method_sources']:
        path=root/source['portable_path'];assert sha(path)==source['sha256']
        node=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.ClassDef)and n.name==source['class_name'])
        code=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
        exec(compile(code,str(path),'exec'),namespace)
    cls=namespace['ExplorationObservedGainCalibration'];results=[];checks=[]
    for entry,reference in zip(index['episodes'],expected['episodes']):
        assert entry['label']==reference['label']
        records=json.loads((root/entry['records_portable_path']).read_text())
        assert [r['action_id']for r in records]==list(range(len(records)))
        gain=cls(tuple(entry['shape']),feedback_enabled=True,consume_predicted_attempts=False)
        rows=[]
        for record in records[1:]:
            aid=record['action_id'];outcome=record['original_prediction_and_actual_feedback']
            pred=record['frozen_prediction_inputs'];actual=record['frozen_actual_feedback_inputs']
            arrays=[]
            for role,item in [('prediction',pred),('actual',actual)]:
                path=root/record['portable_mask_paths'][role]
                assert sha(path)==item['sha256']
                with np.load(path,allow_pickle=False)as z:arrays.append({k:z[k].copy()for k in z.files})
                checks.append(dict(episode=entry['label'],action_id=aid,role=role,
                                   portable_path=str(path.relative_to(root)),sha256=item['sha256']))
            before,after=arrays
            assert set(before)=={'radar_prediction','camera_prediction','known_before','camera_before'}
            assert set(after)=={'belief_after','camera_seen_after'}
            assert all(v.shape==tuple(entry['shape'])for v in [*before.values(),*after.values()])
            assert all(v.dtype==bool for v in before.values())
            assert after['camera_seen_after'].dtype==bool
            p_before=gain.snapshot()['posterior_mean']
            assert outcome['predicted_gain']['posterior_before']==p_before==pred['posterior_before']
            assert pred['original_source_before_sha256']==records[aid-1]['original_sha256']
            assert outcome['frozen_prediction_inputs']==pred
            assert outcome['frozen_actual_feedback_inputs']==actual
            gain._pending[aid]=dict(radar=before['radar_prediction'],camera=before['camera_prediction'],
                known_before=before['known_before'],camera_before=before['camera_before'])
            measured=gain.observe(SimpleNamespace(belief=after['belief_after'],
                camera_seen=after['camera_seen_after']),action_id=aid)
            assert measured==outcome['actual_gain_event']
            assert gain.snapshot()==outcome['gain_snapshot']
            row=dict(action_id=aid,action=outcome['action'],ros_status=outcome['ros_status'],
                primitive_arrival_verified=outcome['primitive_arrival_verified'],
                reason=outcome['reason'],channels={})
            for sensor in ('radar','camera'):
                event=measured[sensor];n=event['predicted_cells'];s=event['realized_predicted_cells'];f=event['unrealized_predicted_cells']
                assert n==s+f==outcome['predicted_gain']['predicted_'+sensor+'_cells']
                assert not np.any(before[sensor+'_prediction']&before['known_before'])
                value=dict(**event,posterior_before=p_before[sensor],posterior_after=measured['posterior_after'][sensor],realized_predicted_fraction=s/n if n else None)
                for name,p in [('online',p_before[sensor]),('fixed_half',.5)]:
                    assert 0<p<1
                    error=p*n-s
                    value[name]=dict(probability=p,predicted_gain=p*n,
                        brier_sum=s*(1-p)**2+f*p*p,
                        logloss_sum=-s*math.log(p)-f*math.log(1-p),
                        gain_abs_error=abs(error),gain_square_error=error*error)
                row['channels'][sensor]=value
            rows.append(row)
        assert rows==reference['actions']
        metrics={sensor:definitions.metric(rows,sensor)for sensor in ('radar','camera')}
        assert metrics==reference['metrics']
        assert gain.snapshot()==reference['final_posterior']
        chronology=reference.get('held_out_interpretation')
        if chronology:
            assert chronology==entry['held_out_interpretation']
            times=chronology['per_action_actual_RGBD_acquisition_unix_ns']
            for tag,cutoff,ids_key in [
                ('after_result_freeze_metrics','original_result_freeze_unix_ns','action_ids_acquired_after_result_freeze'),
                ('after_whole_bundle_freeze_metrics','original_bundle_freeze_unix_ns','action_ids_acquired_after_whole_bundle_freeze')]:
                ids=[r['action_id']for r in rows if times[str(r['action_id'])]>chronology[cutoff]]
                assert ids==chronology[ids_key]
                suffix=[r for r in rows if r['action_id']in ids]
                assert {sensor:definitions.metric(suffix,sensor)for sensor in ('radar','camera')}==chronology[tag]
        results.append(dict(label=entry['label'],completed_attempts=len(rows),
            true_arrivals=sum(r['primitive_arrival_verified']for r in rows),
            ROS_failed_action_ids=[r['action_id']for r in rows if r['ros_status']!=3],
            metrics=metrics,held_out_interpretation=chronology,
            count_reconciliation=entry.get('count_reconciliation'),
            raw_timestamp_headers_not_distributed=True,
            all_frozen_events_posteriors_and_metrics_exact=True))
    assert len(checks)==index['distributed_mask_count']
    result=dict(schema='portable_prequential_mask_reproduction/1',success=True,
        distributed_masks_verified=len(checks),completed_dependent_attempts=sum(r['completed_attempts']for r in results),
        local_full_verification_artifact_count=index['local_verification_artifact_count'],
        absent_raw_originals_reverified=False,new_real_measurements=False,new_robot_actions=0,
        same_event_probability_baseline_only=True,physical_GS_efficiency_claim=False,
        unit_of_analysis=expected['unit_of_analysis'],limitations=expected['limitations'],
        episodes=results,verified_mask_artifacts=checks)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=reproduce(args.directory)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('episodes','verified_mask_artifacts')},indent=2))
