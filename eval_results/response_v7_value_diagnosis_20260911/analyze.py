#!/usr/bin/env python3
"""Read-only decomposition of frozen V7 observations, never an online scorer."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'eval_results/response_v7_training_20260911'
FAMILIES = ('storage_shelves', 'ventilation_baffles')
consumed = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked(path, expected=None):
    digest = sha(path)
    assert expected is None or digest == expected, str(path)
    assert str(path) not in consumed or consumed[str(path)] == digest
    consumed[str(path)] = digest
    return path


def read(path, expected=None):
    return json.loads(tracked(path, expected).read_text())


def oracle(a, b, candidate_ids):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ids = np.asarray(candidate_ids, int)
    if not len(ids):
        return {'candidate_ids': [], 'unavailable': True}
    marginal = (a + b) / 2
    conditional = (a.max() + b.max()) / 2
    value = float(conditional - marginal.max())
    aa = ids[a == a.max()].tolist(); bb = ids[b == b.max()].tolist()
    tolerance = 1e-12 * max(1., float(abs(a).max()), float(abs(b).max()))
    assert value >= -tolerance
    return {'candidate_ids': ids.tolist(), 'storage_values': a.tolist(), 'ventilation_values': b.tolist(),
        'storage_optimal_ids': aa, 'ventilation_optimal_ids': bb, 'common_oracle_ids': ids[marginal == marginal.max()].tolist(),
        'maximizers_intersect_exactly': bool(set(aa) & set(bb)),
        'conditional_oracle': float(conditional), 'marginal_oracle': float(marginal.max()),
        'information_value_upper_bound': value, 'zero_within_roundoff': abs(value) <= tolerance,
        'roundoff_tolerance': tolerance}


def analyze():
    branch_rows, reference_rows, context_rows = [], [], []
    for index in range(8):
        context = f'T{index}'; run = RUNS / context
        manifest = read(run / 'artifact_hashes.json')
        verification = read(run / 'verification.json')
        metadata = read(run / 'metadata.json', manifest['metadata.json'])
        assert metadata['status'] == 'complete' and metadata['physical_branches'] == 12
        assert verification['status'] == 'passed_full' and verification['passed_full'] and not verification['partial']
        assert verification['branches_checked'] == verification['branches_total'] == 12
        assert verification['artifact_manifest_sha256'] == sha(run / 'artifact_hashes.json')
        assert verification['source_archive_sha256'] == sha(run / 'sources.zip')
        assert verification['paired_route_pools_exact'] and verification['paired_nonsemantic_features_exact']
        families = {}
        for family, object_class in zip(FAMILIES, (2, 3)):
            folder = run / family
            def asset(relative):
                path = folder / relative
                return tracked(path, manifest[str(path.relative_to(run))])
            with np.load(asset('reference.npz'), allow_pickle=False) as reference:
                classes = reference['classes'].copy(); weights = reference['weights'].copy()
                vertices, triangles = reference['vertices'], reference['triangles']
                points = reference['points']
                triangle_points = vertices[triangles]
                full_union_area = float(np.linalg.norm(np.cross(triangle_points[:, 1] - triangle_points[:, 0],
                                                               triangle_points[:, 2] - triangle_points[:, 0]), axis=1).sum() / 2)
                assert weights.shape == classes.shape == (len(points),)
                np.testing.assert_allclose(weights, full_union_area / 32000, rtol=1e-13, atol=1e-14)
                assert set(np.unique(classes)) == {1, object_class}
            routes = read(asset('candidates.json'))
            outcomes = {r['candidate_id']: r for r in read(asset('outcomes.json'))}
            assert [r['candidate_id'] for r in routes] == list(range(6)) and set(outcomes) == set(range(6))
            rows = []; first_prefix = None
            for route in routes:
                cid = route['candidate_id']; outcome = outcomes[cid]
                with np.load(asset(f'candidate_{cid:03d}/visibility.npz'), allow_pickle=False) as visibility:
                    masks = {k: visibility[k].copy() for k in visibility.files}
                assert all(x.dtype == bool and x.shape == classes.shape for x in masks.values())
                prefix, union = masks['prefix'], masks['union']
                if first_prefix is None:
                    first_prefix = prefix
                else:
                    np.testing.assert_array_equal(prefix, first_prefix)
                np.testing.assert_array_equal(union, masks['outbound'] | masks['endpoint'] | masks['return'])
                new = union & ~prefix
                parts = {name: float(weights[new & subset].sum()) for name, subset in (
                    ('all', np.ones(len(classes), bool)), ('object', classes == object_class), ('background', classes == 1))}
                np.testing.assert_allclose(parts['all'], parts['object'] + parts['background'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(parts['all'], outcome['new_area_m2'], atol=1e-10, rtol=0)
                actions = read(asset(f'candidate_{cid:03d}/actions.json'))
                paid = len(actions)
                assert paid == outcome['paid_actions'] and route['cost'] == outcome['planned_actions']
                returned = list(actions[-1]['position']) + [actions[-1]['heading']] == route['states'][0]
                assert outcome['failure'] is not None or returned
                seen = prefix.copy(); stage_parts = {}
                for stage in ('outbound', 'endpoint', 'return'):
                    increment = masks[stage] & ~seen
                    stage_parts[stage] = {name: float(weights[increment & subset].sum()) for name, subset in
                        (('all', np.ones(len(classes), bool)), ('object', classes == object_class), ('background', classes == 1))}
                    np.testing.assert_allclose(stage_parts[stage]['all'], outcome['stage_new_area_m2'][stage], atol=1e-10, rtol=0)
                    seen |= masks[stage]
                np.testing.assert_allclose(parts['all'] / paid, outcome['area_per_action'], atol=1e-12, rtol=0)
                row = {'context_id': context, 'family': family, 'candidate_id': cid, 'role': route['group'],
                    'planned_cost': route['cost'], 'actual_paid_cost': paid, 'failure': outcome['failure'],
                    'returned_to_origin': returned, 'successful_full_route': outcome['failure'] is None and returned and paid == route['cost'],
                    **{f'new_area_{k}_m2': value for k, value in parts.items()},
                    **{f'area_rate_{k}': value / paid for k, value in parts.items()},
                    'background_fraction_of_new_area': parts['background'] / parts['all'] if parts['all'] else None,
                    'object_fraction_of_new_area': parts['object'] / parts['all'] if parts['all'] else None,
                    'f1_gain_05cm': outcome['f1_gain_05cm'], 'f1_gain_per_action': outcome['f1_gain_per_action'],
                    'branch_joint_auc_05cm': outcome['branch_joint_auc_05cm'], 'f1_after_05cm': outcome['after']['f1_05cm'],
                    'coverage_gain_m2': outcome['coverage_gain_m2'], 'stage_parts': stage_parts}
                rows.append(row); branch_rows.append(row)
            reference_rows.append({'context_id': context, 'family': family, 'object_class': object_class,
                'full_union_area_m2': full_union_area, 'filtered_reference_points': len(weights), 'weight_per_initial_sample': float(weights[0]),
                'reference_area_all_m2': float(weights.sum()),
                **{f'reference_area_{name}_m2': float(weights[classes == label].sum()) for name, label in (('background', 1), ('object', object_class))},
                **{f'prefix_observed_fraction_{name}': float(weights[first_prefix & (classes == label)].sum() / weights[classes == label].sum())
                   for name, label in (('background', 1), ('object', object_class))}})
            families[family] = {'rows': rows, 'routes': routes}
        a, b = [families[f]['rows'] for f in FAMILIES]
        assert families[FAMILIES[0]]['routes'] == families[FAMILIES[1]]['routes']
        success_ids = [i for i in range(6) if a[i]['successful_full_route'] and b[i]['successful_full_route']]
        metrics = ('area_rate_all', 'area_rate_object', 'area_rate_background', 'f1_gain_05cm', 'f1_gain_per_action', 'branch_joint_auc_05cm')
        primary = {key: oracle([r[key] for r in a], [r[key] for r in b], list(range(6))) for key in metrics}
        sensitivity = {key: oracle([a[i][key] for i in success_ids], [b[i][key] for i in success_ids], success_ids) for key in metrics}
        for name, table in (('all', primary), ('common_success_only_sensitivity', sensitivity)):
            for metric, result in table.items():
                if 'information_value_upper_bound' not in result:
                    continue
                result['background_fraction_at_storage_optima'] = [a[i]['background_fraction_of_new_area'] for i in result['storage_optimal_ids']]
                result['background_fraction_at_ventilation_optima'] = [b[i]['background_fraction_of_new_area'] for i in result['ventilation_optimal_ids']]
                result['failed_storage_optimal_ids'] = [i for i in result['storage_optimal_ids'] if not a[i]['successful_full_route']]
                result['failed_ventilation_optimal_ids'] = [i for i in result['ventilation_optimal_ids'] if not b[i]['successful_full_route']]
        context_rows.append({'context_id': context, 'primary_all_candidates': primary,
            'common_success_candidate_ids': success_ids, 'common_success_only_sensitivity': sensitivity,
            'excluded_only_in_sensitivity': sorted(set(range(6)) - set(success_ids))})
    old_oracle = read(ROOT / 'eval_results/response_v7_T_fit_20260911/paired_information_oracle.json')
    for old, new in zip(old_oracle['contexts'], context_rows):
        np.testing.assert_allclose(old['value_of_family_information_upper_bound'],
                                   new['primary_all_candidates']['area_rate_all']['information_value_upper_bound'], atol=1e-12, rtol=0)
    summary = {'schema_version': 'response_v7_value_diagnosis/1', 'scope': 'posthoc evaluator-only diagnosis; no online changes, new scenes, fitting, or held-out data',
        'all_original_candidates_retained': True, 'new_physical_branches': 0, 'contexts': context_rows,
        'reference_diagnostics': reference_rows, 'physical_failures': [r for r in branch_rows if r['failure'] is not None],
        'nonzero_oracle_contexts_by_metric': {key: [row['context_id'] for row in context_rows if not row['primary_all_candidates'][key]['zero_within_roundoff']]
            for key in metrics},
        'nonzero_success_sensitivity_oracle_contexts_by_metric': {key: [row['context_id'] for row in context_rows if not row['common_success_only_sensitivity'][key]['zero_within_roundoff']]
            for key in metrics},
        'all_branch_pooled_background_fraction': sum(r['new_area_background_m2'] for r in branch_rows) / sum(r['new_area_all_m2'] for r in branch_rows),
        'all_branch_mean_background_fraction_excluding_zero_gain': float(np.mean([r['background_fraction_of_new_area'] for r in branch_rows if r['new_area_all_m2']])),
        'counterfactual_sums_are_not_one_trajectory_area': True,
        'reference_weights_renormalized': False, 'GT_classes_used_only_for_diagnosis': True,
        'fresh_ray_casts_or_mesh_reconstruction_performed': False,
        'oracle_can_replace_primary_gate': False, 'classification_accuracy_or_TSDF_guarantee': False}
    for filename, value in [('summary.json', summary), ('branches.json', branch_rows), ('input_hashes.json', consumed)]:
        (OUT / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (OUT / 'branches.csv').open('w') as stream:
        keys = [k for k in branch_rows[0] if k != 'stage_parts']
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in branch_rows)
    for filename, expected in consumed.items():
        assert sha(Path(filename)) == expected, filename
    print('nonzero', summary['nonzero_oracle_contexts_by_metric'])
    print('success nonzero', summary['nonzero_success_sensitivity_oracle_contexts_by_metric'])
    print('background pooled', summary['all_branch_pooled_background_fraction'])
    for row in context_rows:
        print(row['context_id'], {key: (round(value['information_value_upper_bound'], 9), value['storage_optimal_ids'], value['ventilation_optimal_ids'])
              for key, value in row['primary_all_candidates'].items()})


if __name__ == '__main__':
    analyze()
