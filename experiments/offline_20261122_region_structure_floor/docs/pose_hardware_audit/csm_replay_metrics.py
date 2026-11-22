"""Score archived CSM output against a separate simulator trajectory and retain missing scans."""

from pathlib import Path
import argparse
import json

import numpy as np

from audit_archive import sha256


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def relative(start, end):
    delta = end - start
    c, s = np.cos(start[..., 2]), np.sin(start[..., 2])
    return np.stack((c * delta[..., 0] + s * delta[..., 1],
                     -s * delta[..., 0] + c * delta[..., 1], wrap(delta[..., 2])), axis=-1)


def errors(residual):
    if not len(residual):
        return None
    translation = np.linalg.norm(residual[:, :2], axis=1)
    yaw = np.abs(np.degrees(wrap(residual[:, 2])))
    return {'xy_rmse_m': float(np.sqrt(np.mean(translation ** 2))),
            'xy_p95_m': float(np.percentile(translation, 95)),
            'xy_max_m': float(np.max(translation)),
            'yaw_rmse_deg': float(np.sqrt(np.mean(yaw ** 2))),
            'yaw_p95_deg': float(np.percentile(yaw, 95)), 'yaw_max_deg': float(np.max(yaw))}


def score(out):
    source = out / 'independent_simulator_reference.npz'
    pose_path = out / 'csm_stamped_poses.json'
    poses = json.loads(pose_path.read_text())
    with np.load(source, allow_pickle=False) as reference:
        stamps, raw_truth, paid = (reference[key] for key in ('stamp_ns', 'reference_xyyaw', 'paid_action_id'))
    truth = relative(raw_truth[:1], raw_truth)
    paired = np.full_like(truth, np.nan)
    by_stamp = {int(stamp): index for index, stamp in enumerate(stamps)}
    for pose in poses['ros_poses']:
        index = by_stamp[pose['stamp_ns']]
        if np.isfinite(paired[index]).any():
            raise ValueError('Same scan stamp yielded multiple estimator poses')
        assert pose['frame_id'] == 'odom' and pose['callerid'] == '/nso_csm/matcher'
        paired[index] = pose['xyyaw']
    observed = np.isfinite(paired).all(axis=1)
    charge = np.flatnonzero(paid >= 0)
    # CSM starts in a fresh odom frame. Align only by the simulator's fixed
    # initial reference, with no fit over the resulting CSM trajectory.
    sampled = charge[observed[charge]]
    adjacent = observed[charge[:-1]] & observed[charge[1:]]
    first, last = charge[:-1][adjacent], charge[1:][adjacent]
    seen_ids = np.flatnonzero(observed)
    paired_residuals = paired - truth
    rpe = relative(paired[first], paired[last]) - relative(truth[first], truth[last])
    score = {'schema': 'nso.offline_csm_replay_metrics/1', 'closed_loop_estimated_pose': False,
             'absolute_coordinates_fit_to_reference': False, 'estimator_input_had_reference': False,
             'estimator_start_odom_origin_identity': True,
             'scan_keys_requested': len(truth), 'scan_keys_estimated': int(observed.sum()),
             'keyframe_ends_requested_including_bootstrap': len(charge),
             'keyframe_ends_estimated_including_bootstrap': len(sampled),
             'missing_scan_indices_zero_based': list(map(int, np.flatnonzero(~observed))),
             'missing_action_end_ids': list(map(int, paid[charge[~observed[charge]]])),
             'matched_adjacent_paid_action_rpe_pairs': len(first),
             'native_action_end_ate': errors(paired_residuals[sampled]),
             'all_matched_scan_ate': errors(paired_residuals[seen_ids]),
             'one_paid_action_rpe': errors(rpe),
             'reference_path_distance_m': float(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1).sum()),
             'reference_sha256': sha256(source), 'estimator_pose_sha256': sha256(pose_path),
             'rtab_pose_graph_applied': False}
    (out / 'csm_replay_metrics.json').write_text(json.dumps(score, indent=2) + '\n')
    print(json.dumps(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    score(parser.parse_args().out)
