#!/usr/bin/env python3
"""One posthoc objective probe on the already observed, fixed 24 V8.1 branches.

For all six methods, multiply each sealed score by that candidate's planned
cost exactly once; keep the original cost/ID tie rule. No fitting, new scenes,
new branches, coefficient tuning, or protocol-gate replacement is performed.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import zipfile


METHODS = ('G', 'O', 'S', 'N', 'X', 'M')
RULES = ('original_rate', 'posthoc_absolute_nominal_potential')
PARENTS = ('P0', 'P1')
ARRANGEMENTS = ('shelf_west', 'shelf_east')
METRICS = ('new_area_m2', 'area_per_action', 'final_f1_05cm', 'f1_gain_05cm',
           'f1_gain_per_action', 'branch_joint_auc_05cm', 'branch_joint_auc_02cm',
           'paid_actions', 'planned_actions', 'unused_branch_budget_actions',
           'task_paid_actions', 'coverage_gain_m2', 'fixed_old_support_error_mean_change_m')
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values and all(finite(v) for v in values) else None


def delta(a, b):
    return a - b if finite(a) and finite(b) else None


def choose(scores, costs, ids):
    require(len(scores) == len(costs) == len(ids) and bool(ids), 'invalid candidate vectors')
    require(all(finite(v) for v in scores), 'nonfinite candidate score')
    return ids[min(range(len(ids)), key=lambda i: (-scores[i], costs[i], ids[i]))]


def transformed_scores(scores, planned_costs):
    require(len(scores) == len(planned_costs), 'score/cost lengths differ')
    require(all(type(c) is int and 1 <= c <= 48 for c in planned_costs), 'invalid planned costs')
    values = [s * c for s, c in zip(scores, planned_costs)]
    require(all(finite(s) for s in values), 'nonfinite transformed scores')
    return values


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


class Reader:
    def __init__(self, run):
        self.run = run.resolve(); self.inputs = {}
        self.manifest = self.read(self.run / 'artifact_hashes.json')
        for relative, digest in self.manifest.items():
            path = self.run / relative
            require(not Path(relative).is_absolute() and path.resolve().is_relative_to(self.run),
                    'manifest path escapes run')
            require(self.note(path) == digest, f'acquisition hash changed: {relative}')

    def note(self, path):
        path = Path(path).resolve(); digest = sha(path)
        require(str(path) not in self.inputs or self.inputs[str(path)] == digest, f'input changed: {path}')
        self.inputs[str(path)] = digest
        return digest

    def read(self, path):
        self.note(path)
        return json.loads(Path(path).read_text())

    def sealed(self, relative):
        require(relative in self.manifest, f'input not in raw seal: {relative}')
        return self.read(self.run / relative)


def metric_row(outcome):
    before, after = outcome['before'], outcome['after']
    paid = outcome['paid_actions']
    require(type(paid) is int and 0 <= paid <= 48, 'invalid actual paid actions')
    require(outcome['task_paid_actions'] == 150 + paid, 'prefix cost omitted')
    if paid:
        require(math.isclose(outcome['area_per_action'], outcome['new_area_m2'] / paid,
                             rel_tol=1e-9, abs_tol=1e-9), 'original area rate inconsistent')
    else:
        require(outcome['area_per_action'] is None, 'zero-paid area rate must be undefined')
    values = {k: outcome[k] for k in METRICS if k in outcome}
    values.update(final_f1_05cm=after['f1_05cm'], unused_branch_budget_actions=48-paid,
        fixed_old_support_error_mean_change_m=delta(after.get('fixed_old_support_error_mean_m'),
                                                    before.get('fixed_old_support_error_mean_m')))
    return {**values, 'failure': outcome['failure'], 'collision_count': outcome['collision_count'],
            'returned_to_anchor': outcome['returned_to_anchor'],
            'original_target_reached': outcome['original_target_reached'],
            'full_original_route_completed': outcome['full_original_route_completed'],
            'slot_new_fractions': outcome['slot_new_fractions']}


def run_probe(run, output, original_report=None):
    run, output = run.resolve(), output.resolve()
    require(not output.exists(), 'output exists; use a new diagnostic directory')
    require(not output.is_relative_to(run) and not run.is_relative_to(output),
            'diagnostic output must be outside acquisition tree')
    output.mkdir(parents=True)
    reader = None
    try:
        reader = Reader(run)
        metadata = reader.sealed('metadata.json')
        require(metadata.get('status') in ('complete', 'complete_execution_pending_independent_replay')
                and metadata.get('new_physical_branches') == 24, 'requires the complete original 24 branches')
        seal = reader.sealed('pre_execution_seal.json')
        require(seal.get('physical_branch_limit') == 24, 'wrong experiment scope')
        require(seal['protocol']['schema_version'] == 'competition_v8_1_protocol/1', 'not V8.1')
        require(seal['source_sha256'] == metadata['source_sha256'], 'source inventory mismatch')
        with zipfile.ZipFile(run / 'sources.zip') as archive:
            require(set(archive.namelist()) == set(metadata['source_sha256']), 'archive inventory mismatch')
            for name, digest in metadata['source_sha256'].items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == digest, f'archive changed: {name}')
        for relative, digest in seal['decision_assets'].items():
            require(relative in reader.manifest and reader.note(run / relative) == digest,
                    f'frozen choice input changed: {relative}')
        original = reader.read(original_report) if original_report else None
        if original:
            require(original.get('run') == str(run), 'original report refers to another acquisition')
        definitions = []; outcomes = {}; choices = []
        # This new rule is written after all outcomes have been seen. Loading
        # score vectors first is not advertised as prospective blinding.
        for parent in PARENTS:
            for arrangement in ARRANGEMENTS:
                relative = f'{parent}/{arrangement}'
                routes = reader.sealed(relative + '/candidates.json')
                prediction = reader.sealed(relative + '/predictions.json')
                ids = [r['candidate_id'] for r in routes]; costs = [r['cost'] for r in routes]
                require(ids == list(range(6)), 'six shared candidates required')
                require(set(prediction['scores']) == set(METHODS), 'scorer inventory mismatch')
                derived = {}
                for method in METHODS:
                    scores = prediction['scores'][method]
                    old_pick = choose(scores, costs, ids)
                    require(old_pick == prediction['selected_candidate_ids'][method], 'old sealed choice mismatch')
                    transformed = transformed_scores(scores, costs)
                    derived[method] = {'original_scores': scores, 'planned_costs': costs,
                        'posthoc_absolute_nominal_scores': transformed, 'original_choice': old_pick,
                        'posthoc_choice': choose(transformed, costs, ids)}
                definitions.append({'context': parent, 'arrangement': arrangement,
                                    'candidate_ids': ids, 'methods': derived})
                for candidate in routes:
                    cid = candidate['candidate_id']
                    outcome = reader.sealed(relative + f'/candidate_{cid:03d}/outcome.json')
                    require((outcome['context'], outcome['arrangement'], outcome['candidate_id'])
                            == (parent, arrangement, cid), 'outcome identity mismatch')
                    require(outcome['planned_actions'] == candidate['cost'], 'planned cost changed')
                    outcomes[(parent, arrangement, cid)] = outcome
                for method in METHODS:
                    for rule, pick_key in zip(RULES, ('original_choice', 'posthoc_choice')):
                        cid = derived[method][pick_key]; outcome = outcomes[(parent, arrangement, cid)]
                        choices.append({'context': parent, 'arrangement': arrangement, 'method': method,
                            'rule': rule, 'candidate_id': cid, 'group': outcome['group'],
                            **metric_row(outcome)})
        require(len(outcomes) == 24 and len(choices) == 48, 'incomplete fixed diagnostic matrix')
        summaries = {}; comparisons = {}; changes = []
        for rule in RULES:
            summaries[rule] = {}
            for method in METHODS:
                rows = [r for r in choices if r['rule'] == rule and r['method'] == method]
                require(len(rows) == 4, 'unequal history averaging')
                summaries[rule][method] = {
                    'histories': 4, 'means': {k: mean(r.get(k) for r in rows) for k in METRICS},
                    'defined_counts': {k: sum(finite(r.get(k)) for r in rows) for k in METRICS},
                    'per_parent': {p: {k: mean(r.get(k) for r in rows if r['context'] == p) for k in METRICS}
                                   for p in PARENTS},
                    'selected_failures': sum(r['failure'] is not None for r in rows),
                    'selected_collisions': sum(r['collision_count'] for r in rows),
                    'selected_not_returned': sum(not r['returned_to_anchor'] for r in rows)}
            comparisons[rule] = {}
            for other in ('G', 'O', 'N', 'X', 'M'):
                sm, cm = summaries[rule]['S']['means'], summaries[rule][other]['means']
                comparisons[rule][other] = {
                    'mean_differences': {k: delta(sm[k], cm[k]) for k in METRICS},
                    'per_parent': {p: {k: delta(summaries[rule]['S']['per_parent'][p][k],
                                                summaries[rule][other]['per_parent'][p][k]) for k in METRICS}
                                   for p in PARENTS}}
        for history in definitions:
            for method, d in history['methods'].items():
                old = outcomes[(history['context'], history['arrangement'], d['original_choice'])]
                new = outcomes[(history['context'], history['arrangement'], d['posthoc_choice'])]
                oldm, newm = metric_row(old), metric_row(new)
                changes.append({'context': history['context'], 'arrangement': history['arrangement'],
                    'method': method, 'original_choice': d['original_choice'], 'posthoc_choice': d['posthoc_choice'],
                    'choice_changed': d['original_choice'] != d['posthoc_choice'],
                    'original_group': old['group'], 'posthoc_group': new['group'],
                    'new_minus_original': {k: delta(newm[k], oldm[k]) for k in METRICS}})
        result = {'schema_version': 'competition_v8_1_objective_posthoc_diagnosis/1',
            'status': 'complete_posthoc_diagnosis', 'run': str(run),
            'rule': 'every method and candidate: posthoc_score=sealed_score*planned_cost; exact score/cost/id tie',
            'rules_examined': 1, 'methods': list(METHODS), 'physical_branches_reused': 24,
            'new_physical_branches': 0, 'new_training': False, 'new_calibration': False,
            'new_candidates': False, 'coefficients_changed': False,
            'rule_chosen_after_observing_original_outcomes': True,
            'independent_or_preregistered_validation': False,
            'original_V8_1_gate_decision_modified': False,
            'original_report': str(original_report.resolve()) if original_report else None,
            'original_report_status': original.get('status') if original else None,
            'original_failed_gates': original.get('failed_gates') if original else None,
            'original_positive_mechanism_claim_allowed': original.get('positive_mechanism_claim_allowed') if original else None,
            'protocol_gates_applied_as_new_validation': False,
            'summaries': summaries, 'S_comparisons': comparisons, 'choice_changes': changes,
            'changed_history_count_per_method': {m: sum(r['choice_changed'] for r in changes if r['method'] == m)
                                                  for m in METHODS},
            'all_physical_failures_retained': [{k: r[k] for k in ('context', 'arrangement', 'candidate_id',
                'failure', 'collision_count', 'returned_to_anchor', 'paid_actions', 'planned_actions')}
                for r in outcomes.values() if r['failure'] is not None or r['collision_count'] or not r['returned_to_anchor']],
            'limitations': ['Nominal potential is still an uncalibrated geometric/asset-prior proxy.',
                'Multiplying by planned cost does not simulate spending the remaining 48-action budget.',
                'New visible area and F1/old-support error are different outcomes.',
                'Original exclusivity, final-F1 and all other gate failures cannot be repaired by this posthoc table.',
                'Only two already observed development parents; no C/E or mainstream comparison.'],
            'outcome_values_modified': False}
        changed_inputs = [p for p, digest in reader.inputs.items() if sha(p) != digest]
        require(not changed_inputs, 'input changed during posthoc diagnosis')
        snapshot = output / Path(__file__).name; shutil.copyfile(__file__, snapshot)
        result['provenance'] = {'script_sha256': sha(snapshot), 'input_count': len(reader.inputs),
            'input_manifest_sha256': reader.note(run / 'artifact_hashes.json'),
            'source_archive_sha256': reader.note(run / 'sources.zip'),
            'pre_execution_seal_sha256': reader.note(run / 'pre_execution_seal.json')}
        write_json(output / 'summary.json', result)
        write_json(output / 'posthoc_scores.json', definitions)
        write_json(output / 'choices.json', choices)
        write_json(output / 'input_hashes.json', reader.inputs)
        write_csv(output / 'choices.csv', choices); write_csv(output / 'choice_changes.csv', changes)
        (output / 'REPORT.md').write_text(report(result, choices))
        write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
        print(json.dumps({'status': result['status'], 'changed': result['changed_history_count_per_method'],
                          'S_comparisons': comparisons}, ensure_ascii=False))
        return 0
    except Exception as exc:
        write_json(output / 'failure.json', {'status': 'failed_diagnosis', 'error': f'{type(exc).__name__}: {exc}',
            'original_protocol_decision_modified': False, 'inputs_sha256': reader.inputs if reader else {}})
        raise


def f(value):
    return f'{value:.6f}' if finite(value) else '不可判定'


def report(result, choices):
    lines = ['# V8.1 评分目标的单项事后诊断', '',
        '**这是查看原24条结果后提出的开发探针，不是预注册或独立验证。**'
        '只对G/O/S/N/X/M六方法统一做一次 `新分数=原冻结分数×计划动作数`，'
        '保持分数／代价／候选ID的tie规则；未调任何其他系数，没有新候选、训练或物理分支。', '',
        f'原V8.1主报告状态：`{result["original_report_status"]}`；原未通过门槛：'
        f'`{result["original_failed_gates"]}`。该判定及原协议完全不修改。', '',
        '|目标／方法|新增m²|面积／动作|终点F1@5cm|F1增量|J-AUC@5cm|实际动作|未用48步预算|150+实际动作|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rule in RULES:
        for method in METHODS:
            r = result['summaries'][rule][method]['means']
            lines.append('|' + ('原rate' if rule == RULES[0] else '事后名义总量') + '/' + method + '|'
                + '|'.join(f(r[k]) for k in ('new_area_m2', 'area_per_action', 'final_f1_05cm',
                    'f1_gain_05cm', 'branch_joint_auc_05cm', 'paid_actions', 'unused_branch_budget_actions', 'task_paid_actions')) + '|')
    lines += ['', '所有均值按相同四history等权，未定义值不补零。', '',
              '|history／方法|原选择→事后选择|原角色→事后角色|', '|---|---|---|']
    for r in result['choice_changes']:
        lines.append(f'|{r["context"]}/{r["arrangement"]}/{r["method"]}'
                     f'|{r["original_choice"]}→{r["posthoc_choice"]}'
                     f'|{r["original_group"]}→{r["posthoc_group"]}|')
    lines += ['', '|目标／S相对对照|新增面积差|面积率差|终点F1差|J-AUC差|实际动作差|',
              '|---|---:|---:|---:|---:|---:|']
    for rule in RULES:
        for other in ('G', 'O', 'N', 'X', 'M'):
            r = result['S_comparisons'][rule][other]['mean_differences']
            lines.append('|' + ('原rate' if rule == RULES[0] else '事后名义总量') + '/'+other+'|'
                + '|'.join(f(r[k]) for k in ('new_area_m2', 'area_per_action', 'final_f1_05cm',
                                              'branch_joint_auc_05cm', 'paid_actions'))+'|')
    lines += ['', '乘以计划成本只是消除原代理的单位动作归一化，不会把代理变成真实新增面积或F1，'
              '也没有实际补花未用预算。需要区分“更愿意走长路线”的共同目标效应和S相对G/O/N/X的细类别差异；'
              '所有对照及每父差值完整保留在JSON中。', '',
              '本表用于决定是否有理由另外冻结新的训练／校准与完整预算闭环方案。'
              '不把事后较好选择算作原V8.1通过，不重新计算一套有利门槛，不按结果继续搜索更多目标或追加样本。'
              '原传感器排他条件是否成立、自然语义迁移与主流领先均不由本表证明。', '']
    return '\n'.join(lines)


def self_test():
    scores, costs, ids = [2., 1.5], [2, 4], [0, 1]
    require(choose(scores, costs, ids) == 0, 'rate ranking')
    require(transformed_scores(scores, costs) == [4., 6.], 'planned-cost multiplication')
    require(choose(transformed_scores(scores, costs), costs, ids) == 1, 'absolute ranking')
    require(choose([4., 4.], [2, 4], ids) == 0, 'cost tie')
    require(choose([4., 4.], [2, 2], [7, 2]) == 2, 'candidate ID tie')
    require(transformed_scores([-2., 1.], [2, 4]) == [-4., 4.], 'signed scores clipped')
    require(mean([1., None]) is None, 'undefined outcome imputed')
    print(json.dumps({'status': 'synthetic_rule_test_passed', 'physical_data_read': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--original-report', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.run is None or args.output is None:
            parser.error('--run and --output are required')
        sys.exit(run_probe(args.run, args.output, args.original_report))
