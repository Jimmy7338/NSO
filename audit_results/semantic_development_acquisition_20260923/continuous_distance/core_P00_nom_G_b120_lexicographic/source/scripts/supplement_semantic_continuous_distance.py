#!/usr/bin/env python3
"""Describe saved endpoint distances without changing the frozen primary score.

Requires an explicitly pinned complete original review and its independently
verified saved-policy/TSDF replay. Only fixed reference samples and the saved
complete mesh are read; no World, replay, mapping or primary scoring is run.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.complete_surface_evaluation import _positive_area_mesh, _sample, _closest
from nso.episode_driver_v43 import canonical_bytes, file_sha256
from nso.offline_evaluation_v44 import _array_sha
from nso.semantic_experiment import EVALUATION, _repo_path
from nso.surface_evaluation_v40 import ReferenceSurfaceV40
from scripts import reuse_semantic_scene_endpoint_evaluation as proof


SUPPLEMENTAL_SOURCES = ('scripts/supplement_semantic_continuous_distance.py',
    'scripts/reuse_semantic_scene_endpoint_evaluation.py')
REFERENCE_DOMAIN = ('All fixed observable task-facility reference points to the complete saved prediction mesh; '
    'every reference facility remains in the denominator, regardless of reconstruction or identification.')
PREDICTION_DOMAIN = ('Area-weighted complete prediction samples to the complete scene GT mesh, including background; '
    'not target precision, not target-only accuracy and not symmetric Chamfer distance.')


def weighted_summary(distances, weights):
    """Inverse empirical CDF quantiles, with +infinity mass never discarded."""
    d, w = np.asarray(distances, float), np.asarray(weights, float)
    if (d.ndim != 1 or w.shape != d.shape or np.isnan(d).any() or np.any(d < 0)
            or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError('nonnegative distances (possibly +inf) and positive finite weights required')
    total = float(w.sum()); finite = np.isfinite(d)
    unmatched = float(w[~finite].sum())
    def quantile(q):
        if not len(d):
            return None
        order = np.argsort(d, kind='stable')
        index = min(int(np.searchsorted(np.cumsum(w[order]), q * total, side='left')), len(d)-1)
        value = float(d[order[index]])
        return value if np.isfinite(value) else None
    return dict(samples=len(d), total_weight=total, finite_weight=float(w[finite].sum()),
        unmatched_weight=unmatched, unmatched_fraction=unmatched/total if total else None,
        mean_m=float(np.dot(d, w)/total) if len(d) and finite.all() else None,
        p50_m=quantile(.5), p95_m=quantile(.95), empty_domain=not len(d),
        mean_is_infinite=bool(len(d) and not finite.all()),
        null_convention='null means an infinite quantile/mean or an empty sampling domain; never a zero error')


def weighted_ecdf(distances, weights):
    weighted_summary(distances, weights)
    d, w = np.asarray(distances, float), np.asarray(weights, float)
    if not len(d):
        return np.empty(0), np.empty(0)
    order = np.argsort(d, kind='stable')
    support, first = np.unique(d[order], return_index=True)
    mass = np.add.reduceat(w[order], first)
    cdf = np.cumsum(mass)/float(w.sum())
    cdf[-1] = 1.
    return support, cdf


def continuous_distances(reference, vertices, triangles, *, sample_spacing_m, seed, max_samples):
    """Pure finite quadrature; use the main evaluator's exact mesh/distance rules."""
    if type(reference) is not ReferenceSurfaceV40:
        raise TypeError('frozen ReferenceSurfaceV40 required')
    manifest = reference.manifest()
    ids = np.unique(reference.point_instance_id)
    target_ids = np.unique(reference.triangle_instance_id[reference.triangle_instance_id >= 0])
    if (not len(ids) or not np.array_equal(ids, target_ids)
            or not np.isfinite(reference.area_weights).all() or np.any(reference.area_weights <= 0)):
        raise ValueError('every task facility must have its complete fixed positive reference quadrature')
    pv, pt, _, pa, _ = _positive_area_mesh(vertices, triangles, allow_empty=True)
    submitted_count, submitted_area = len(pt), float(pa.sum())
    # Identical to the frozen main evaluator: exact geometric duplicates carry
    # no additional surface, while every distinct strictly positive face stays.
    seen, keep = set(), []
    for index, triangle in enumerate(pv[pt]):
        key = tuple(sorted(tuple(float(x) for x in p) for p in triangle))
        if key not in seen:
            seen.add(key); keep.append(index)
    retained = np.asarray(keep, dtype=np.int64)
    pt, pa = pt[retained], pa[retained]
    pp, pw, sample_triangle = _sample(pv, pt, pa, sample_spacing_m, seed, max_samples)
    began = time.monotonic()
    rd, rq, rt = _closest(reference.points, pv, pt)
    recall_s = time.monotonic()-began
    began = time.monotonic()
    pd, pq, pgt = _closest(pp, reference.vertices, reference.triangles)
    prediction_s = time.monotonic()-began
    # Empty prediction has no nearest point/triangle; make the point sentinel
    # explicit instead of exposing _closest's internal initial zero coordinates.
    rq[rt < 0] = np.nan
    macro_weights = np.empty(len(rd), float)
    per_instance = []
    for instance in ids:
        select = reference.point_instance_id == instance
        weights = reference.area_weights[select]
        stats = weighted_summary(rd[select], weights)
        macro_weights[select] = weights/(len(ids)*float(weights.sum()))
        per_instance.append(dict(instance_id=int(instance), reference_area_m2=float(weights.sum()),
            unmatched_area_m2=stats['unmatched_weight'], **stats))
    macro = weighted_summary(rd, macro_weights)
    ecdf_d, ecdf_p = weighted_ecdf(rd, macro_weights)
    pred_stats = weighted_summary(pd, pw)
    nearest_owner = reference.triangle_instance_id[pgt]
    all_stats = weighted_summary(rd, reference.area_weights)
    result = dict(reference_domain=REFERENCE_DOMAIN, prediction_domain=PREDICTION_DOMAIN,
        reference_fingerprint=manifest['fingerprint'], reference_samples=len(rd),
        all_task_instance_ids=[int(i) for i in ids], all_task_instances_in_macro_denominator=True,
        prediction_roi_cropped=False, reference_points_resampled=False,
        reference_to_prediction=dict(per_instance=per_instance, area_weighted_all_reference=all_stats,
            equal_facility_macro=dict(**macro, facility_count=len(ids), weight_per_facility=1./len(ids),
                weight_rule='w_ij = area_weight_ij / (N_facilities * reference_area_i)',
                ecdf=dict(npz_distance_key='reference_macro_ecdf_distance_m',
                    npz_cumulative_probability_key='reference_macro_ecdf_probability',
                    support_points=len(ecdf_d), infinite_support_mass=macro['unmatched_weight'],
                    exact_weighted_empirical_cdf=True, finite_values_not_renormalized=True))),
        prediction_to_full_scene=dict(**pred_stats, prediction_area_m2=pred_stats['total_weight'],
            unmatched_area_m2=pred_stats['unmatched_weight'], includes_background=True,
            nearest_gt_background_area_m2=float(pw[nearest_owner < 0].sum()),
            nearest_gt_task_area_m2=float(pw[nearest_owner >= 0].sum()),
            owner_partition_note='Nearest GT owner only, at any distance; these areas are not true positives.'),
        sampling=dict(sample_spacing_m=sample_spacing_m, seed=seed, max_samples=max_samples,
            prediction_samples=len(pp), submitted_prediction_triangles=submitted_count,
            canonical_prediction_triangles=len(pt), submitted_prediction_area_m2=submitted_area,
            canonical_prediction_area_m2=float(pa.sum()),
            duplicate_prediction_triangles_removed=submitted_count-len(pt),
            duplicate_prediction_area_removed_m2=submitted_area-float(pa.sum()),
            distinct_positive_micro_faces_preserved=int(np.count_nonzero(pa <= 5e-13)),
            prediction_sampling_matches_frozen_main_evaluator=True,
            quantile_definition='smallest distance whose cumulative area weight reaches q; no interpolation',
            distance_kernel='frozen _closest exhaustive Euclidean point-to-triangle distance'),
        distance_elapsed_s=dict(reference_to_prediction=recall_s, prediction_to_full_scene=prediction_s))
    arrays = dict(reference_points_m=reference.points, reference_instance_id=reference.point_instance_id,
        reference_distance_m=rd, reference_area_weight_m2=reference.area_weights,
        reference_equal_facility_macro_weight=macro_weights, reference_nearest_prediction_point_m=rq,
        reference_nearest_prediction_canonical_triangle=rt,
        reference_macro_ecdf_distance_m=ecdf_d, reference_macro_ecdf_probability=ecdf_p,
        prediction_points_m=pp, prediction_area_weight_m2=pw, prediction_distance_to_full_scene_m=pd,
        prediction_sample_canonical_triangle=sample_triangle,
        prediction_sample_original_triangle=retained[sample_triangle],
        canonical_prediction_original_triangle=retained,
        prediction_nearest_full_scene_point_m=pq, prediction_nearest_full_scene_triangle=pgt,
        prediction_nearest_full_scene_instance_id=nearest_owner)
    return result, arrays


def _checked_inputs(episode_root, manifest_sha256, review_path, review_sha256):
    episode = proof._checked_episode(episode_root, manifest_sha256)
    review = proof._checked_review(review_path, review_sha256, episode, complete_numerical=True)
    surface, _, record = proof._load_reference(episode)
    evaluation = review['evaluation']; metrics = evaluation.get('metrics', {})
    if (evaluation.get('schema') != 'semantic_scene.complete_saved_prediction_evaluation.v1'
            or evaluation.get('episode_manifest_sha256') != manifest_sha256
            or evaluation.get('asset_id') != episode['slot']['asset_id']
            or evaluation.get('asset_manifest_sha256') != episode['protocol']['asset_manifest_sha256']
            or evaluation.get('reference_manifest_sha256') != episode['reference']['manifest_sha256']
            or evaluation.get('input_prediction_sha256') != proof._prediction_pins(episode)
            or evaluation.get('evaluator_source_sha256') != episode['manifest']['source_sha256']['nso/complete_surface_evaluation.py']
            or evaluation.get('all_task_instances_in_macro_denominator') is not True
            or evaluation.get('prediction_roi_cropped') is not False
            or evaluation.get('prediction_reintegrated') is not False
            or metrics.get('reference_fingerprint') != surface.fingerprint
            or metrics.get('prediction_mesh_validation') != 'strict_positive_area_without_absolute_area_floor'
            or metrics.get('prediction_sample_spacing_m') != EVALUATION['sample_spacing_m']
            or metrics.get('prediction_seed') != EVALUATION['seed']
            or metrics.get('threshold_m') != EVALUATION['threshold_m']
            or evaluation.get('coverage', {}).get('denominator') != record['coverage']):
        raise ValueError('original complete review does not bind the fixed reference, whole prediction and evaluator')
    return episode, review, surface


def supplement(*, episode, manifest_sha256, review, review_sha256, output_dir):
    began, cpu_began = time.monotonic(), time.process_time()
    before = runtime_counts_v41()
    episode_root, review_path, output = (Path(x).resolve() for x in (episode, review, output_dir))
    if output.exists():
        raise ValueError('new supplemental directory required; existing failures/results are never overwritten')
    inspected, original, reference = _checked_inputs(episode_root, manifest_sha256, review_path, review_sha256)
    immutable = [episode_root, _repo_path(inspected['reference']['root'])]
    immutable += [_repo_path(inspected['protocol'][key]) for key in ('asset_root', 'navigation_root')]
    if any(output.is_relative_to(root.resolve()) for root in immutable):
        raise ValueError('supplement must be outside immutable episode, asset, navigation and reference trees')
    sources = {name: file_sha256(ROOT/name) for name in SUPPLEMENTAL_SOURCES}
    output.mkdir(parents=True)
    for name, pin in sources.items():
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        if file_sha256(target) != pin:
            raise ValueError('supplemental source changed while copying')
    declaration = dict(schema='semantic.continuous_distance.declaration.v1',
        episode_root=str(episode_root), episode_manifest_sha256=manifest_sha256,
        review_path=str(review_path), review_sha256=review_sha256,
        run_id=inspected['started']['run_id'], reference=deepcopy(inspected['reference']),
        source_sha256=deepcopy(inspected['manifest']['source_sha256']), supplemental_source_sha256=sources,
        input_sha256=deepcopy(inspected['manifest']['input_sha256']), evaluation_configuration=deepcopy(EVALUATION),
        main_metrics_modified=False, automatic_retry=False, new_worlds=0)
    (output/'started.json').write_bytes(canonical_bytes(declaration))
    try:
        with np.load(episode_root/'prediction/mesh.npz', allow_pickle=False) as mesh:
            if set(mesh.files) != {'vertices', 'triangles', 'vertex_colors'}:
                raise ValueError('complete original prediction array inventory required')
            summary, arrays = continuous_distances(reference, mesh['vertices'], mesh['triangles'],
                **{key: EVALUATION[key] for key in ('sample_spacing_m', 'seed', 'max_samples')})
        metrics = original['evaluation']['metrics']
        if (summary['reference_samples'] != metrics['reference_samples']
                or summary['sampling']['prediction_samples'] != metrics['prediction_samples']
                or summary['sampling']['canonical_prediction_triangles'] != metrics['canonical_prediction_triangles']
                or summary['all_task_instance_ids'] != [row['instance_id'] for row in metrics['per_instance']]):
            raise ValueError('supplemental quadrature inventory differs from original complete evaluation')
        npz = output/'distances.npz'
        np.savez_compressed(npz, **arrays)
        final, _, _ = _checked_inputs(episode_root, manifest_sha256, review_path, review_sha256)
        if (final['manifest'] != inspected['manifest']
                or any(file_sha256(ROOT/name) != pin for name, pin in sources.items())):
            raise ValueError('input/source changed during supplemental distance computation')
        after = runtime_counts_v41()
        if before != after:
            raise ValueError('unexpected live sensor activity during offline computation')
        result = dict(declaration, schema='semantic.continuous_distance.supplement.v1',
            status='continuous_distance_completed', method=inspected['slot']['method'],
            asset_id=inspected['slot']['asset_id'], descriptive_supplement_only=True,
            primary_numerical_evaluation_recomputed=False, independent_replay_performed_here=False,
            original_complete_review_status=original['status'],
            unchanged_primary_metrics_from_review={key:metrics[key] for key in ('C_nav','Q','J_nav','threshold_m')},
            no_facility_selection_from_results=True, no_semantic_importance_weighting=True,
            new_sensor_packets=0, new_tsdf_integrations=0, physical_actions=0,
            summary=summary, distance_archive=dict(path='distances.npz', sha256=file_sha256(npz),
                bytes=npz.stat().st_size, arrays={key:dict(shape=list(value.shape), dtype=value.dtype.str,
                    sha256=_array_sha(value)) for key, value in arrays.items()},
                infinite_distance_representation='IEEE +inf; absent nearest point is NaN and triangle index is -1'),
            runtime=dict(before=before, after=after, no_new_world_or_sensor_action=True,
                wall_elapsed_s=time.monotonic()-began, process_cpu_s=time.process_time()-cpu_began))
        (output/'receipt.json').write_bytes(canonical_bytes(result))
        return result
    except Exception as error:
        (output/'error.json').write_bytes(canonical_bytes(dict(declaration,
            status='continuous_distance_failed', error_type=type(error).__name__, error=str(error),
            elapsed_s=time.monotonic()-began, automatic_retry=False)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    result = supplement(**vars(parser.parse_args()))
    print(canonical_bytes(dict(status=result['status'], run_id=result['run_id'],
        runtime=result['runtime'], summary=result['summary'])).decode(), end='')


if __name__ == '__main__':
    main()
