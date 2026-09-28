#!/usr/bin/env python3
"""Replay a fixed paid packet history through common local observation modules.

No new actions, World, TSDF integrations or evaluation truth are used. This is
an observation-chain repair check, not a counterfactual trajectory experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_mechanism_marker_chain import load_packet
from nso.observed_instances_local import ObservedInstancesLocal
from nso.observed_residual_v41 import ObservedResidualV41
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_v42 import ViewQualityPredictorV42


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(episode, maximum_seconds):
    began = time.monotonic()
    spec = json.loads((episode/'public_spec.json').read_text())
    workspace = json.loads((episode/'public_workspace.json').read_text())
    prior = spec['structure_prior']
    ledger = ObservedInstancesLocal(palette=workspace['marker_palette'],
        structure_names=prior['abstract_structures'],
        class_structure_prior=prior['probability_by_category'], mode='S', maximum_instances=8)
    residual = ObservedResidualV41()
    predictor = ViewQualityPredictorV42(maximum_instances=8)
    packets = sorted((episode/'packets').glob('*_rgbd.npz'))
    rows, forecasts, forecasted = [], [], set()
    inputs = {}
    timings = Counter()
    for path in packets:
        if time.monotonic()-began > maximum_seconds:
            break
        inputs[str(path.relative_to(episode))] = file_sha(path)
        obs = load_packet(path)
        start = time.monotonic()
        associated = ledger.observe(obs)
        timings['association_s'] += time.monotonic()-start
        start = time.monotonic()
        scores = residual.observe(obs, associated['accepted'])
        feedback = [ledger.apply_geometry_feedback(row['instance_id'], frame_id=obs.frame_id,
            observation_sha256=obs.sha256(), log_likelihoods=row['log_likelihoods'])
            for row in scores['results'] if row['accepted']]
        timings['residual_s'] += time.monotonic()-start
        start = time.monotonic()
        views = predictor.observe(obs, associated['accepted'])
        timings['view_evidence_s'] += time.monotonic()-start
        # Probe the first recovered reliable plane at an ordinary public turn
        # candidate. This does not render that viewpoint or choose a new action.
        reliable = {row['instance_id'] for row in views['results'] if row['reliable_cached_plane']}
        for instance in ledger.snapshot()['instances']:
            key = instance['instance_id']
            if key not in reliable or key in forecasted:
                continue
            transform = np.array(obs.world_from_camera, copy=True)
            yaw = np.arctan2(transform[1,2],transform[0,2])+np.pi/6
            transform[:3,:3] = [[np.sin(yaw),0.,np.cos(yaw)],[-np.cos(yaw),0.,np.sin(yaw)],[0.,-1.,0.]]
            candidate = CandidateViewV40(obs.intrinsic,transform,obs.depth_m.shape[1],obs.depth_m.shape[0],
                                        view_id='public_turn_plus_30')
            start = time.monotonic()
            forecast = predictor.forecast(instance,[candidate])
            timings['forecast_probe_s'] += time.monotonic()-start
            forecasts.append(forecast)
            forecasted.add(key)
        rows.append(dict(paid_step=obs.paid_step, observation_sha256=obs.sha256(),
            accepted_associations=[dict(instance_id=r['instance_id'], pixels=len(r['pixel_indices']),
                marker_pixels=len(r['marker_pixel_indices']), novel_support_voxels=r['novel_support_voxels'],
                geometry_feedback_eligible=r['geometry_feedback_eligible'], support_sha256=r['support_sha256'])
                for r in associated['accepted']],
            association_rejection_counts=dict(Counter(r['reason'] for r in associated['rejected'])),
            local_proposal_audit=associated['local_proposal_audit'],
            residual_results=scores['results'], geometry_feedback_applied=[r['instance_id'] for r in feedback if r['applied']],
            view_evidence=views,
            instances=[{k:r[k] for k in ('instance_id','observed_class','semantic_conditioning_used',
                'association_uncertain','geometric_feedback_frames','novel_support_frames','support_sha256')}
                for r in ledger.snapshot()['instances']]))
    count = len(rows)
    return dict(schema='local_marker_frontend.fixed_paid_history_replay.v1',
        episode_path=str(episode), status='complete' if count == len(packets) else 'time_limit_partial',
        maximum_seconds=maximum_seconds, elapsed_s=time.monotonic()-began, timings=dict(timings),
        input_files_sha256=inputs,
        source_sha256={name:file_sha(ROOT/name) for name in (
            'scripts/replay_marker_local_frontend.py','scripts/diagnose_mechanism_marker_chain.py',
            'nso/observed_instances_local.py','nso/observed_instances_v41.py',
            'nso/observed_residual_v41.py','nso/view_quality_v42.py')},
        world_calls=0, tsdf_integrations=0, task_scores_read=False, evaluation_truth_read=False,
        new_physical_actions=0, original_plane_thresholds_unchanged=True,
        summary=dict(paid_packets_processed=count, expected_paid_packets=len(packets),
            accepted_associations=sum(len(row['accepted_associations']) for row in rows),
            accepted_residuals=sum(sum(r['accepted'] for r in row['residual_results']) for row in rows),
            feedback_applications=sum(len(row['geometry_feedback_applied']) for row in rows),
            accepted_current_plane_fits=[dict(step=row['paid_step'],instance_id=r['instance_id'])
                for row in rows for r in row['residual_results'] if r.get('current_plane_fit',{}).get('accepted')],
            reliable_cached_plane_observations=sum(sum(r['reliable_cached_plane'] for r in row['view_evidence']['results']) for row in rows),
            first_reliable_plane_forecast_probes=len(forecasts),
            nonfallback_forecast_probes=sum(not f['candidates'][0]['fallback'] for f in forecasts)),
        frames=rows, first_reliable_plane_forecasts=forecasts,
        causal_limits=['Original observation history is fixed; a repaired online controller may choose different actions.',
            'Accepted planes and residuals demonstrate recovered interfaces, not improved reconstruction quality.',
            'Common local proposal repair must be used by all compared methods.',
            'No plane acceptance thresholds, geometry evidence scales, or class prior weights were tuned.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=ROOT/'audit_results/mechanism_development_20260923/episodes/pilot_A_S')
    parser.add_argument('--output', type=Path, default=ROOT/'audit_results/mechanism_development_20260923/diagnostics/local_frontend_saved_history_replay.json')
    parser.add_argument('--maximum-seconds', type=float, default=300.)
    args = parser.parse_args()
    if not 0 < args.maximum_seconds <= 600:
        raise ValueError('bounded replay time in (0,600] required')
    result = replay(args.episode.resolve(),args.maximum_seconds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),status=result['status'],elapsed_s=result['elapsed_s'],summary=result['summary']),sort_keys=True))


if __name__=='__main__':
    main()
