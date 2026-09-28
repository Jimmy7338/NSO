#!/usr/bin/env python3
"""Diagnose a saved forecast provenance alias using a fixed fast-only prefix.

This reuses the already slow/fast-verified 0..83 prefix. It reads saved paid
packets only and does not run a simulator, generate actions or inspect scores.
The failed verifier artifacts and original replay are left unchanged.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_local_exact_acceleration import (
    canonical_sha,file_sha,first_differences,historical_forecast_reference,write_json,
)
from scripts.diagnose_mechanism_marker_chain import load_packet
from nso.observed_instances_local import ObservedInstancesLocal
from nso.observed_residual_v41 import ObservedResidualV41
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_v42 import ViewQualityPredictorV42


def main():
    began=time.perf_counter()
    base=ROOT/'audit_results/mechanism_development_20260923'
    episode=base/'episodes/pilot_A_S'
    failed=base/'diagnostics/exact_acceleration_validation'
    output=base/'diagnostics/forecast_reference_alias_diagnosis.json'
    if output.exists():
        raise FileExistsError('diagnostic output already exists; do not silently repeat')
    old_path=base/'diagnostics/local_frontend_saved_history_replay.json'
    old=json.loads(old_path.read_text())['first_reliable_plane_forecasts'][0]
    reference,repair=historical_forecast_reference(old)
    prior=json.loads((episode/'public_spec.json').read_text())['structure_prior']
    workspace=json.loads((episode/'public_workspace.json').read_text())
    ledger=ObservedInstancesLocal(palette=workspace['marker_palette'],
        structure_names=prior['abstract_structures'],class_structure_prior=prior['probability_by_category'],
        mode='S',maximum_instances=8,exact_distance_acceleration=True)
    residual=ObservedResidualV41()
    predictor=ViewQualityPredictorV42(maximum_instances=8)
    prefix={row['paid_step']:row for row in map(json.loads,(failed/'frames.jsonl').read_text().splitlines())}
    checked=[]
    actual=None
    for path in sorted((episode/'packets').glob('*_rgbd.npz')):
        if time.perf_counter()-began>60.:
            raise TimeoutError('60 second diagnostic cap exceeded; do not automatically rerun')
        obs=load_packet(path)
        if obs.paid_step>old['paid_step']:
            break
        associated=ledger.observe(obs)
        if obs.paid_step in prefix:
            assert canonical_sha(associated)==prefix[obs.paid_step]['association_sha256']
            assert file_sha(path)==prefix[obs.paid_step]['packet_file_sha256']
            checked.append(obs.paid_step)
        scores=residual.observe(obs,associated['accepted'])
        for row in scores['results']:
            if row['accepted']:
                ledger.apply_geometry_feedback(row['instance_id'],frame_id=obs.frame_id,
                    observation_sha256=obs.sha256(),log_likelihoods=row['log_likelihoods'])
        predictor.observe(obs,associated['accepted'])
        if obs.paid_step==old['paid_step']:
            instance=next(row for row in ledger.snapshot()['instances'] if row['instance_id']==old['instance_id'])
            transform=np.array(obs.world_from_camera,copy=True)
            yaw=np.arctan2(transform[1,2],transform[0,2])+np.pi/6
            transform[:3,:3]=[[np.sin(yaw),0.,np.cos(yaw)],[-np.cos(yaw),0.,np.sin(yaw)],[0.,-1.,0.]]
            candidate=CandidateViewV40(obs.intrinsic,transform,obs.depth_m.shape[1],obs.depth_m.shape[0],
                                      view_id='public_turn_plus_30')
            actual=deepcopy(predictor.forecast(instance,[candidate]))
    assert actual is not None
    report=dict(schema='forecast_reference_alias_diagnosis.v1',world_calls=0,new_actions=0,
        tsdf_integrations=0,evaluation_truth_read=False,slow_frontend_calls=0,
        fast_paid_packets=old['paid_step']+1,previously_verified_prefix_associations_rechecked=checked,
        source_sha256={name:file_sha(ROOT/name) for name in (
            'scripts/diagnose_forecast_reference_alias.py','scripts/verify_local_exact_acceleration.py',
            'nso/observed_instances_local.py','nso/exact_support_distance.py','nso/view_quality_v42.py')},
        original_replay_sha256=file_sha(old_path),original_reference_repair=repair,
        original_forecast_differences=first_differences(actual,old),
        restored_reference_differences=first_differences(actual,reference),
        exact_equal_after_verified_metadata_repair=(actual==reference),
        actual_forecast_sha256=canonical_sha(actual),actual_forecast=actual,
        elapsed_s=time.perf_counter()-began,
        cause='V42 forecast observed_geometry.source_frames aliases the mutable predictor list; the historical replay did not deepcopy before later observations appended rows.',
        scope='Only historical forecast provenance metadata changed; complete numerical forecast and original geometry digest must agree exactly.')
    write_json(output,report)
    print(json.dumps({key:report[key] for key in ('fast_paid_packets','elapsed_s',
        'original_forecast_differences','restored_reference_differences','exact_equal_after_verified_metadata_repair')}))
    return 0 if actual==reference else 1


if __name__=='__main__':
    raise SystemExit(main())
