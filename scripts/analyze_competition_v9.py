#!/usr/bin/env python3
"""Analyze every prospectively fixed V9 route with the original frozen gates.

This process reads evidence only. Both score families and all six methods are
reported, including failed routes. No sensor, mapper or planner is imported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_competition_v8_1 import (
    Inputs, require, near, finite, gate, averages, comparison, load_reference,
    audit_outcome, audit_saved_evidence, paired_oracle, replay_status, sha,
)

METHODS = ('G', 'O', 'S', 'N', 'X', 'M')
PROTOCOL = 'configs/virtual3d/competition_v9_validation_protocol.json'
PROTOCOL_SHA = '0cd8cdd4795ca7faaf5707e77fd266914d7858701c43eeccaa49dff781c3020a'
PRIMARY = ('new_area_m2', 'final_f1_05cm', 'final_joint_05cm')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def analyze(run, output, review):
    run, output, review = run.resolve(), output.resolve(), review.resolve()
    require(not output.exists() and not output.is_relative_to(run), 'separate new output required')
    own = sha(__file__); helper = sha(ROOT / 'scripts/analyze_competition_v8_1.py')
    inputs = Inputs(); manifest = inputs.manifest(run)
    metadata = inputs.declared_read(run, 'metadata.json')
    seal = inputs.declared_read(run, 'pre_execution_seal.json')
    require(seal['source_sha256'] == metadata['source_sha256'], 'source seal mismatch')
    require(metadata['source_sha256'].get(str(Path(__file__).relative_to(ROOT))) == own,
            'analyzer must be frozen before execution')
    require(metadata['source_sha256'].get('scripts/analyze_competition_v8_1.py') == helper,
            'shared analysis helpers must be frozen before execution')
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        require(set(archive.namelist()) == set(metadata['source_sha256']), 'archive inventory')
        for name, digest in metadata['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, 'archive changed: ' + name)
        require(metadata['source_sha256'][PROTOCOL] == PROTOCOL_SHA, 'prospective protocol changed')
        protocol = json.loads(archive.read(PROTOCOL))
    require(seal['protocol'] == protocol, 'executor protocol differs')
    require(protocol['physical_branch_limit'] == seal['physical_branch_limit'] == 48, 'branch limit')
    for name, digest in protocol['provenance']['unchanged_dependencies'].items():
        require(metadata['source_sha256'][name] == digest, 'old component changed')
    prepared = Path(metadata['prepared'])
    pmanifest = inputs.manifest(prepared)
    require(inputs.note(prepared / 'artifact_hashes.json') == seal['prepared_manifest_sha256'], 'preparation binding')
    pm = inputs.declared_read(prepared, 'metadata.json')
    require(pm['status'] == 'complete_structural_pass' and pm['new_candidate_branches'] == 0, 'preparation gate')
    structure = inputs.declared_read(prepared, 'structure_summary.json')
    exact_histories = {(p, a) for p in protocol['parent_order'] for a in protocol['arrangement_order']}
    require(structure['passed'] and len(structure['histories']) == 8
            and {(h['context'], h['arrangement']) for h in structure['histories']} == exact_histories
            and all(h['passed'] and all(h['checks'].values()) for h in structure['histories']), 'complete structural inventory')
    for name, digest in pm['source_sha256'].items():
        require(metadata['source_sha256'].get(name) == digest, 'prepared execution source changed')
    expected_guards = {f'eval_results/execution_guard_v8_{suffix}_20260911/verification.json'
                       for suffix in ('failures', 'success_impacts')}
    require(set(seal['guard_verification_sha256']) == expected_guards, 'guard evidence inventory')
    for name, digest in seal['guard_verification_sha256'].items():
        require(inputs.note(ROOT / name) == digest and inputs.read(ROOT / name)['passed_full'] is True,
                'guard evidence missing or changed')
    prechoices = inputs.declared_read(prepared, 'pre_evaluator_choices_seal.json')
    for name, digest in prechoices.items():
        require(seal['decision_assets'].get(name) == digest, 'pre-evaluator choice changed')
    for name, digest in seal['decision_assets'].items():
        require(pmanifest.get(name) == manifest.get(name) == digest, 'decision asset binding')
    release = inputs.read(review)
    require(inputs.note(review) == seal['structural_review_sha256'], 'release receipt changed')
    require(release['release_allowed'] and release['prepared_manifest_sha256'] == seal['prepared_manifest_sha256'], 'release not granted')
    verification = replay_status(inputs, run, metadata, 48)
    if (run / 'verification.json').exists():
        replay_receipt = inputs.read(run / 'verification.json')
        require(replay_receipt['verifier_sha256'] == metadata['source_sha256']['scripts/replay_competition_v9.py'], 'replay implementation not frozen')
        require(replay_receipt['structural_review_sha256'] == seal['structural_review_sha256'], 'replay structural receipt differs')
    parents, arrangements, ids = [protocol[k] for k in ('parent_order', 'arrangement_order', 'candidate_order')]
    selected = {family: {} for family in ('total', 'rate')}
    histories, rows, tables, evidence = [], [], {}, []
    expected = {f'{p}/{a}/candidate_{i:03d}/outcome.json' for p in parents for a in arrangements for i in ids}
    require({n for n in manifest if n.endswith('/outcome.json')} == expected, 'missing or additional physical branch')
    for parent in parents:
        for arrangement in arrangements:
            rel = f'{parent}/{arrangement}'; folder = run / rel
            routes = inputs.declared_read(run, rel + '/candidates.json')
            prediction = inputs.declared_read(run, rel + '/predictions.json')
            require([r['candidate_id'] for r in routes] == ids, 'candidate IDs/order')
            require([r['group'] for r in routes] == protocol['pre_outcome_structural_gates']['candidate_roles_in_order'], 'candidate roles')
            ref = load_reference(inputs, folder); table = {}
            for route in routes:
                branch = f'{rel}/candidate_{route["candidate_id"]:03d}'
                row = audit_outcome(inputs.declared_read(run, branch + '/outcome.json'), route, parent, arrangement, ref, protocol)
                evidence.append({'branch': branch, **audit_saved_evidence(inputs, run, branch, row, ref, protocol)})
                for state in (row['before'], row['after']):
                    near(state['covered_area_m2'], state['coverage_2d'] * int(ref['reachable'].sum())
                         * protocol['sensors']['resolution_m'] ** 2, 'covered area formula')
                    for tag in ('02cm', '05cm'):
                        near(state[f'area_times_f1_{tag}'], state['covered_area_m2'] * state[f'f1_{tag}'], 'area times F1 formula')
                for field in ('covered_area_m2', 'area_times_f1_02cm', 'area_times_f1_05cm',
                              'f1_02cm', 'precision_02cm', 'precision_05cm', 'recall_02cm', 'recall_05cm'):
                    row['final_' + field] = row['after'][field]
                table[route['candidate_id']] = row; rows.append(row)
            require(all(r['before'] == table[0]['before'] for r in table.values()), 'different prefix metrics')
            tables[(parent, arrangement)] = table
            history = {'context': parent, 'arrangement': arrangement, 'families': {}}
            for family, sf, cf in (('total', 'scores', 'selected_candidate_ids'), ('rate', 'rate_scores', 'rate_selected_candidate_ids')):
                scores, choices = prediction[sf], prediction[cf]
                require(set(scores) == set(choices) == set(METHODS), 'method inventory')
                for mode in METHODS:
                    require(len(scores[mode]) == 6 and all(finite(v) for v in scores[mode]), 'invalid score vector')
                    pick = min(ids, key=lambda i: (-scores[mode][i], routes[i]['cost'], i))
                    require(choices[mode] == pick, 'frozen choice/tie disagreement')
                    if family == 'total':
                        require(scores[mode] == [v * r['cost'] for v, r in zip(prediction['rate_scores'][mode], routes)], 'total objective changed')
                selected[family][(parent, arrangement)] = {mode: table[choices[mode]] for mode in METHODS}
                history['families'][family] = {'scores': scores, 'choices': choices,
                    'M_equals_G': scores['M'] == scores['G'] and choices['M'] == choices['G']}
            histories.append(history)
    for parent in parents:
        a, b = [h for h in histories if h['context'] == parent]
        require(inputs.read(run / parent / arrangements[0] / 'candidates.json') == inputs.read(run / parent / arrangements[1] / 'candidates.json'), 'paired routes changed')
        for family in selected:
            first, second = a['families'][family]['scores'], b['families'][family]['scores']
            require(all(first[m] == second[m] for m in ('G', 'O', 'N', 'M')), 'paired nonsemantic scores')
            require(first['S'] == second['X'] and first['X'] == second['S'], 'paired class swap')
    methods = {family: {m: averages([row[m] for row in selected[family].values()], 8) for m in METHODS} for family in selected}
    for family in selected:
        for mode in METHODS:
            for field in ('covered_area_m2', 'area_times_f1_02cm', 'area_times_f1_05cm',
                          'f1_02cm', 'precision_02cm', 'precision_05cm', 'recall_02cm', 'recall_05cm'):
                key = 'final_' + field
                methods[family][mode]['means'][key] = float(np.mean([row[mode][key] for row in selected[family].values()]))
    comparisons = {m: comparison(selected['total'], m, parents, arrangements) for m in ('G', 'O', 'N', 'X', 'M')}
    cross = {key: {'S': selected['total'][key]['S'], 'rate': selected['rate'][key]['S']} for key in selected['total']}
    rate_comparison = comparison(cross, 'rate', parents, arrangements)
    spec = protocol['progression_necessary_conditions']
    gates = {'all_48_complete': gate(len(rows) == metadata['new_physical_branches'] == 48),
             'independent_full_replay': verification,
             'source_structure_and_release_integrity': gate(True),
             'all_routes_collision_free_target_and_return': gate(all(r['collision_count'] == 0 and r['failure'] is None
                 and r['original_target_reached'] and r['returned_to_anchor'] and r['full_original_route_completed'] for r in rows))}
    for other in spec['primary_total_comparators']:
        delta = comparisons[other]['means']; cm = methods['total'][other]['means']['new_area_m2']
        area_pass = (delta['new_area_m2'] >= spec['near_zero_minimum_absolute_area_gain_m2'] if cm < spec['near_zero_comparator_area_m2']
                     else delta['new_area_m2'] / cm >= spec['min_S_relative_mean_new_area_gain_against_each_total_comparator'])
        gates[f'S_total_area_over_{other}'] = gate(area_pass, delta_m2=delta['new_area_m2'], relative_gain=delta['new_area_m2']/cm if cm else None)
        for metric, bound in (('final_f1_05cm', 'min_S_minus_each_total_comparator_mean_global_F1_05m'),
                              ('final_joint_05cm', 'min_S_minus_each_total_comparator_mean_C_times_F1_05m'),
                              ('branch_joint_auc_05cm', 'minimum_mean_S_minus_each_total_comparator_sparse_joint_auc')):
            gates[f'S_total_{metric}_over_{other}'] = gate(delta[metric] >= spec[bound], delta=delta[metric], threshold=spec[bound])
        for metric in PRIMARY:
            values = {p: comparisons[other]['per_parent'][p][metric] for p in parents}
            gates[f'all_parents_S_{metric}_over_{other}'] = gate(all(v > 0 for v in values.values()), deltas=values)
    for metric, bound in (('new_area_m2', 'minimum_mean_S_total_minus_S_rate_new_area_m2'),
                          ('final_f1_05cm', 'minimum_mean_S_total_minus_S_rate_global_F1'),
                          ('final_joint_05cm', 'minimum_mean_S_total_minus_S_rate_C_times_F1'),
                          ('branch_joint_auc_05cm', 'minimum_mean_S_total_minus_S_rate_sparse_joint_auc')):
        value = rate_comparison['means'][metric]
        gates[f'total_vs_rate_{metric}'] = gate(value >= spec[bound], delta=value, threshold=spec[bound])
    gates['correct_S_over_X_all_primary'] = gate(all(comparisons['X']['means'][k] > 0 for k in PRIMARY),
                                                deltas={k: comparisons['X']['means'][k] for k in PRIMARY})
    gates['missing_M_exact_G_both_families'] = gate(all(h['families'][f]['M_equals_G'] for h in histories for f in selected))
    gates['coverage_noninferiority_to_N'] = gate(comparisons['N']['means']['final_coverage_2d'] >= spec['minimum_mean_S_minus_N_coverage_2d'])
    for p in parents:
        choices = [h['families']['total']['choices']['S'] for h in histories if h['context'] == p]
        gates[f'{p}_semantic_choice_exchange'] = gate(len(set(choices)) == 2, choices=choices)
    oracles = {p: {k: paired_oracle(tables[(p, arrangements[0])], tables[(p, arrangements[1])], ids, k,
                                  protocol['information_value_diagnostic']['tie_absolute_tolerance'])
                   for k in (*PRIMARY, 'branch_joint_auc_05cm')} for p in parents}
    failed = [k for k, g in gates.items() if g['passed'] is False]
    pending = [k for k, g in gates.items() if g['passed'] is None]
    positive = not failed and not pending
    result = {'schema_version': 'competition_v9_analysis/1',
        'status': 'passed_prospective_finite_mechanism_gates' if positive else 'failed_prospective_gates' if failed else 'pending_replay',
        'run': str(run), 'protocol_sha256': PROTOCOL_SHA, 'histories': histories,
        'method_averages': methods, 'S_total_comparisons': comparisons, 'S_total_vs_S_rate': rate_comparison,
        'paired_oracles': oracles, 'gates': gates, 'failed_gates': failed, 'pending_gates': pending,
        'available_branches': len(rows), 'paid_branch_actions': sum(r['paid_actions'] for r in rows),
        'unique_prefix_paid_actions': 1200, 'independent_geometry_blocks': 4,
        'sparse_auc_rule': protocol['metrics']['sparse_branch_joint_auc_rule'],
        'physical_failures': [r for r in rows if not r['common_success_sensitivity_eligible']],
        'all_prefix_coverage_equals_one': all(r['before']['coverage_2d'] == 1 for r in rows),
        'positive_finite_mechanism_confirmation_allowed': positive,
        'whole_four_module_system_advantage_proven': False, 'mainstream_superiority_proven': False,
        'learned_or_calibrated_semantic_response': False, 'natural_or_open_vocabulary_semantics': False,
        'broad_population_generalization_proven': False, 'prior_V8_1_failed_gates_preserved': True,
        'new_rule_was_prospective_for_this_batch': True, 'minimum_one_sided_sign_p': 0.0625,
        'no_extra_physical_negative_control_branches': True,
        'saved_branch_evidence_audits': evidence}
    chosen = [{'family': f, 'method': m, **row} for f in selected for modes in selected[f].values() for m, row in modes.items()]
    lines = ['# V9 新几何场景预注册验证', '', f"状态：`{result['status']}`。48 条路线，{result['paid_branch_actions']} 个后续付费动作，另含 1200 个独立前置采集动作。", '',
             '|评分|方法|新增表面 m²|终点 F1@5cm|C×F1|稀疏48动作联合AUC|付费动作|', '|---|---|---:|---:|---:|---:|---:|']
    for f in selected:
        for m in METHODS:
            v = methods[f][m]['means']
            lines.append(f"|{f}|{m}|{v['new_area_m2']:.6f}|{v['final_f1_05cm']:.6f}|{v['final_joint_05cm']:.6f}|{v['branch_joint_auc_05cm']:.6f}|{v['paid_actions']:.2f}|")
    lines += ['', '所有门槛、负结果、跨设备可见面积和失败路线均保留在 summary.json / all_branches.json。', '',
              '未通过门槛：' + (', '.join(failed) or '无'), '待核验门槛：' + (', '.join(pending) or '无'), '',
              '本批为 4 个预先固定尺寸/噪声组合的人工类别标记试验。先验未学习或校准；未证明自然语义、完整四模块系统或主流方法优势。每条短路线停止后保持地图至48动作，不包含剩余预算再规划。',
              '四组全正时单侧符号检验最小 p=0.0625；不作 p<0.05 或总体泛化结论。',
              '若前置二维覆盖已经为1，C×F1与F1相同，只能支持后期三维巡检机制。']
    for path, digest in inputs.hashes.items(): require(sha(path) == digest, 'input changed during analysis')
    require(sha(__file__) == own and sha(ROOT / 'scripts/analyze_competition_v8_1.py') == helper, 'analysis code changed')
    output.mkdir(parents=True)
    for name, value in (('summary.json', result), ('all_branches.json', rows), ('selected_branches.json', chosen), ('input_hashes.json', inputs.hashes)):
        write(output / name, value)
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    shutil.copyfile(__file__, output / 'analysis_source.py')
    shutil.copyfile(ROOT / 'scripts/analyze_competition_v8_1.py', output / 'shared_analysis_helpers.py')
    write(output / 'artifact_hashes.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'status': result['status'], 'failed_gates': failed, 'pending_gates': pending,
                      'mean_total_S': methods['total']['S']['means']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--structural-review', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.output, args.structural_review)
