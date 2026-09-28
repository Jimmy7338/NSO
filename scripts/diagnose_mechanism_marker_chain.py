#!/usr/bin/env python3
"""Diagnose paid marker packets only; no World, TSDF or task-score access."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.instance_belief_v40 import PaidRGBDObservationV40, _components
from nso.observed_instances_v41 import _depth_components
from nso.observed_residual_v41 import fit_marker_plane_v41


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_packet(path):
    with np.load(path, allow_pickle=False) as arrays:
        fields = {key: np.array(arrays[key], copy=True) for key in arrays.files}
    fields['frame_id'] = str(fields['frame_id'].item())
    fields['paid_step'] = int(fields['paid_step'].item())
    return PaidRGBDObservationV40(**fields)


def geometry_diagnostics(observation, pixels, relative_sigma):
    pixels = np.sort(np.asarray(pixels, dtype=np.int64))
    y, x = np.divmod(pixels, observation.depth_m.shape[1])
    depth = observation.depth_m.ravel()[pixels].astype(float)
    valid = (depth >= .1) & (depth <= 4.)
    x, y, depth = x[valid], y[valid], depth[valid]
    rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(observation.intrinsic).T
    rays /= rays[:, 2, None]
    directions = rays @ observation.world_from_camera[:3, :3].T
    origin = observation.world_from_camera[:3, 3]
    points = origin + depth[:, None]*directions
    center = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points-center, full_matrices=False)
    normal = vt[-1]
    toward_camera = origin-center
    if normal @ toward_camera < 0:
        normal = -normal
    incidence = float(normal @ toward_camera / np.linalg.norm(toward_camera))
    rms = float(np.sqrt(np.mean(((points-center) @ normal)**2)))
    expected_sigma = float(np.sqrt(np.mean((relative_sigma*depth*(directions @ normal))**2)))
    horizontal_normal = normal.copy()
    horizontal_normal[2] = 0.
    horizontal_normal /= np.linalg.norm(horizontal_normal)
    horizontal = np.cross([0., 0., 1.], horizontal_normal)
    extent = np.ptp(np.column_stack(((points-center) @ horizontal, points[:, 2])), axis=0)
    # Diagnostic alternative only: use the existing upright mounting assumption
    # and known relative axial noise to fit inverse depth. No acceptance gate
    # or planning state is changed by this statistical calculation.
    design = directions[:, :2]
    inverse_sigma = relative_sigma/depth
    weights = 1./inverse_sigma**2
    information = design.T @ (weights[:, None]*design)
    coefficient = np.linalg.solve(information, design.T @ (weights/depth))
    covariance = np.linalg.inv(information)
    upright = -np.r_[coefficient, 0.]/np.linalg.norm(coefficient)
    if upright @ toward_camera < 0:
        upright = -upright
    gradient = np.array([-coefficient[1], coefficient[0]])/np.dot(coefficient, coefficient)
    yaw_std = float(np.degrees(np.sqrt(gradient @ covariance @ gradient)))
    fitted_depth = 1./(design @ coefficient)
    fitted_points = origin + fitted_depth[:, None]*directions
    fitted_horizontal = np.cross([0., 0., 1.], upright)
    fitted_extent = np.ptp(np.column_stack((fitted_points @ fitted_horizontal,
                                            fitted_points[:, 2])), axis=0)
    normalized_residual = (1./depth-design @ coefficient)/inverse_sigma
    height, width = observation.depth_m.shape
    return dict(marker_pixels=len(pixels), valid_depth_pixels=len(depth),
        pixel_bounds_xy=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
        touches_image_border=bool(np.any((x == 0) | (x == width-1) | (y == 0) | (y == height-1))),
        depth_min_m=float(depth.min()), depth_median_m=float(np.median(depth)), depth_max_m=float(depth.max()),
        svd_singular_values=singular.tolist(), svd_small_to_middle_ratio=float(singular[2]/singular[1]),
        svd_normal_world=normal.tolist(), plane_rms_m=rms,
        expected_perpendicular_noise_rms_m=expected_sigma,
        rms_over_expected_noise=rms/expected_sigma,
        incidence=incidence, raw_pixel_center_extent_m=extent.tolist(),
        original_fitter=fit_marker_plane_v41(observation, pixels),
        diagnostic_upright_inverse_depth=dict(normal_world=upright.tolist(), yaw_std_deg=yaw_std,
            normalized_residual_rms=float(np.sqrt(np.mean(normalized_residual**2))),
            plane_projected_pixel_center_extent_m=fitted_extent.tolist(),
            fitting_assumption='upright plane plus public relative axial depth noise',
            applied_to_controller=False, completeness_not_established=True))


def diagnose(episode):
    workspace_path, spec_path = episode/'public_workspace.json', episode/'public_spec.json'
    workspace, spec = json.loads(workspace_path.read_text()), json.loads(spec_path.read_text())
    palette = workspace['marker_palette']
    relative_sigma = spec['sensor']['depth_noise_relative_std']
    if not 0 < relative_sigma < 1:
        raise ValueError('diagnostic noise normalization requires public positive relative sigma')
    source_hashes = {}
    for name in ('nso/instance_belief_v40.py', 'nso/observed_instances_v41.py',
                 'nso/observed_residual_v41.py'):
        current, saved = ROOT/name, episode/'source'/name
        if sha(current) != sha(saved):
            raise ValueError('original pilot source differs: '+name)
        source_hashes[name] = sha(saved)
    inputs = {str(p.relative_to(episode)): sha(p) for p in (workspace_path, spec_path)}
    markers, attempts = [], []
    frames = []
    for path in sorted((episode/'packets').glob('*_rgbd.npz')):
        observation = load_packet(path)
        step = observation.paid_step
        step_path = episode/f'steps/{step:03d}.json.gz'
        saved = json.loads(gzip.decompress(step_path.read_bytes()))
        evidence = saved['controller_evidence']
        if evidence['observation_sha256'] != observation.sha256():
            raise ValueError('saved paid observation SHA differs at step '+str(step))
        for item in (path, step_path):
            inputs[str(item.relative_to(episode))] = sha(item)
        associations = evidence['association']
        rejection_counts = dict(Counter(row['reason'] for row in associations['rejected']))
        frames.append(dict(step=step, association_rejection_counts=rejection_counts,
                           accepted_instances=len(associations['accepted'])))
        for row in evidence['observed_residual']['results']:
            fit = row.get('current_plane_fit', {})
            if 'plane_rms_m' not in fit:
                continue
            association = next(a for a in associations['accepted'] if a['instance_id'] == row['instance_id'])
            diagnostic = geometry_diagnostics(observation, association['marker_pixel_indices'], relative_sigma)
            attempts.append(dict(step=step, instance_id=row['instance_id'],
                diagnostic=diagnostic, recorded_fit=fit,
                recorded_fit_equals_recomputed=fit == diagnostic['original_fitter']))
        depth = observation.depth_m
        valid = (depth >= .1) & (depth <= 4.)
        visible = []
        for label, color in palette.items():
            mask = np.all(observation.rgb == np.asarray(color, np.uint8), axis=-1) & valid
            for component in _components(mask):
                if len(component) >= 16:
                    visible.append((label, np.sort(component[:, 0]*depth.shape[1]+component[:, 1])))
        if not visible:
            continue
        y, x = np.indices(depth.shape)
        rays = np.column_stack((x.ravel(), y.ravel(), np.ones(depth.size))) @ np.linalg.inv(observation.intrinsic).T
        world = ((rays*depth.ravel()[:, None]) @ observation.world_from_camera[:3, :3].T
                 + observation.world_from_camera[:3, 3])
        components = list(_depth_components(valid, world.reshape(*depth.shape, 3), .15))
        for label, pixels in visible:
            diagnostic = geometry_diagnostics(observation, pixels, relative_sigma)
            accepted = [dict(instance_id=a['instance_id'], marker_pixels=len(a['marker_pixel_indices']),
                             marker_mask_equals_full_component=set(a['marker_pixel_indices']) == set(pixels.tolist()))
                        for a in associations['accepted'] if np.isin(a['marker_pixel_indices'], pixels).any()]
            connected = []
            anchor = np.median(world[pixels], axis=0)
            for component in components:
                hits = int(np.isin(component, pixels).sum())
                if not hits:
                    continue
                points = world[component]
                bounded = points[np.linalg.norm(points-anchor, axis=1) <= .6]
                extent = np.ptp(points, axis=0)
                connected.append(dict(component_points=len(component), marker_points=hits,
                    measured_extent_m=extent.tolist(), original_oversize_rejection=bool(extent.max() > 2.5),
                    original_bootstrap_radius_m=.6, points_within_bootstrap_radius=len(bounded),
                    local_extent_m=np.ptp(bounded, axis=0).tolist() if len(bounded) else None))
            markers.append(dict(step=step, observed_class=label, diagnostic=diagnostic,
                accepted_associations=accepted, original_rejection_counts=rejection_counts,
                measured_depth_components=connected))
    return dict(schema='marker_observation_chain.paid_packet_diagnosis.v1', episode_path=str(episode),
        input_files_sha256=inputs, original_source_sha256=source_hashes,
        diagnostic_script_sha256=sha(Path(__file__)), depth_noise_relative_std=relative_sigma,
        world_calls=0, tsdf_integrations=0, evaluation_truth_read=False, task_scores_read=False,
        geometry_only_saved_packet_analysis=True, counterfactual_trajectory_claim=False,
        original_plane_attempts=attempts, complete_marker_components=markers, per_frame=frames,
        summary=dict(paid_packets=len(frames), original_fitted_marker_attempts=len(attempts),
            all_original_fits_reproduced=all(a['recorded_fit_equals_recomputed'] for a in attempts),
            marker_components_at_least_16_pixels=len(markers),
            nonborder_marker_components=sum(not r['diagnostic']['touches_image_border'] for r in markers),
            unassociated_marker_components=sum(not r['accepted_associations'] for r in markers),
            full_marker_fits_accepted_by_original_fitter=[r['step'] for r in markers if r['diagnostic']['original_fitter']['accepted']],
            full_marker_fit_rejection_reasons=dict(Counter(r['diagnostic']['original_fitter']['reason'] for r in markers)),
            unassociated_markers_in_oversize_components=sum(not r['accepted_associations'] and
                any(c['original_oversize_rejection'] for c in r['measured_depth_components']) for r in markers)),
        causal_limits=['Full marker fits are recomputations from existing paid pixels, not a new autonomous trajectory.',
            'Border-clipped marker fragments do not establish a complete label pose.',
            'The inverse-depth fit is a diagnostic calculation, not an approved replacement acceptance rule.',
            'This identifies observation-chain failures; it does not establish a semantic mechanism advantage.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=ROOT/'audit_results/mechanism_development_20260923/episodes/pilot_A_S')
    parser.add_argument('--output', type=Path, default=ROOT/'audit_results/mechanism_development_20260923/diagnostics/marker_observation_chain.json')
    args = parser.parse_args()
    result = diagnose(args.episode.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), summary=result['summary']), sort_keys=True))


if __name__ == '__main__':
    main()
