#!/usr/bin/env python3
"""Summarize the two completed cost pilots; never execute a sensor or policy.

This is a measurement-chain diagnostic, not a new-mechanism efficacy test.
Episode manifests and the separately verified saved-policy replays are pinned.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


PINS = {
    'pilot_A_G': '90b1f67bae4796cb33b95d2b7894ae0bab2befb974dee017c640432d12673168',
    'pilot_A_S': '44fa95d84306f6749e571424373260c657f224fdaa256fb897b6934355d6ac73',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    rows = []
    for run, manifest_sha in PINS.items():
        episode = root / 'episodes' / run
        if sha(episode / 'artifact_manifest.json') != manifest_sha:
            raise ValueError('unexpected pilot manifest: ' + run)
        result = json.loads((episode / 'result.json').read_text())
        review_path = root / 'reviews' / (run + '.json')
        review = json.loads(review_path.read_text())
        if (review['episode_manifest_sha256'] != manifest_sha or
                review['replay']['status'] != 'verified' or
                not review['replay']['prediction_verified']):
            raise ValueError('verified saved-policy and mapping replay required')
        steps = sorted((episode / 'steps').glob('*.json.gz'))
        if len(steps) != result['acquired_and_saved_packets']:
            raise ValueError('incomplete saved decisions')
        counts, reasons, inputs = Counter(), Counter(), {}
        for step_path in steps:
            step = json.loads(gzip.decompress(step_path.read_bytes()))
            inputs[step_path.name] = sha(step_path)
            decision, evidence = step['decision'], step['controller_evidence']
            counts['decisions'] += 1
            counts['decisions_with_positive_inspection_score'] += int(any(
                x['inspection_expected_new_area_m2'] > 0
                for x in decision.get('candidate_utilities', [])))
            for forecast in decision.get('forecasts', []):
                counts['instance_forecasts'] += 1
                counts['semantic_conditioned_instance_forecasts'] += int(
                    forecast['semantic_conditioning_used'])
                counts['instance_forecasts_with_valid_candidates'] += int(any(
                    not x['fallback'] for x in forecast['candidates']))
            for item in evidence['view_evidence']['results']:
                fit = item['current_plane_fit']
                if fit is not None:
                    counts['plane_fit_attempts'] += 1
                    counts['accepted_plane_fits'] += int(fit['accepted'])
                    reasons[fit['reason']] += 1
            counts['accepted_residuals'] += sum(
                int(x['accepted']) for x in evidence['observed_residual']['results'])
            counts['applied_geometry_feedback'] += sum(
                int(x['applied']) for x in evidence['geometry_feedback'])
        actions = [x['controller_action'] for x in result['actions']]
        rows.append(dict(run_id=run, episode_manifest_sha256=manifest_sha,
            result_sha256=sha(episode / 'result.json'), review_sha256=sha(review_path),
            status=result['status'], paid_actions=result['executed_paid_actions'],
            saved_packets=result['acquired_and_saved_packets'], collisions=result['collisions'],
            returned_xy_and_yaw=result['sensor_status']['returned_xy_and_yaw'],
            execution_elapsed_s=result['elapsed_s'], timings=result['timings'],
            saved_episode_bytes=sum(p.stat().st_size for p in episode.rglob('*') if p.is_file()),
            action_sequence_sha256=hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
            mesh_file_sha256=sha(episode / 'prediction/mesh.npz'),
            occupancy_file_sha256=sha(episode / 'prediction/occupancy.npz'),
            saved_policy_replay_verified_frames=review['replay']['frames_verified'],
            original_review_status=review['status'], original_review_error=review.get('error'),
            counts=dict(counts), plane_fit_reasons=dict(reasons), step_file_sha256=inputs))
    return dict(schema='mechanism.cost_pilot_pair_diagnostic.v1',
        scope='DEV_A_00 legacy greedy G/S; cost and observation-chain diagnosis only',
        source_sha256=sha(Path(__file__)), original_executed_worlds=2,
        new_worlds_in_this_analysis=0, new_paid_actions_in_this_analysis=0,
        candidate_mechanism_trajectories=0, strong_geometry_comparison=False,
        semantic_information_value_established=False, algorithm_increment_established=False,
        identical_action_sequences=rows[0]['action_sequence_sha256']==rows[1]['action_sequence_sha256'],
        identical_endpoint_mesh_bytes=rows[0]['mesh_file_sha256']==rows[1]['mesh_file_sha256'],
        identical_endpoint_occupancy_bytes=rows[0]['occupancy_file_sha256']==rows[1]['occupancy_file_sha256'],
        interpretation='No valid observed label planes reached inspection forecasting. '
            'Equal trajectories do not evaluate the new reliability mechanism or refute semantic value.',
        pilots=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=Path('audit_results/mechanism_development_20260923'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'pilots'}, sort_keys=True))
