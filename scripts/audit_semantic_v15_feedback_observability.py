#!/usr/bin/env python3
"""Reconcile recorded online feedback; no world, evaluator or policy execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    require(type(value) is int and value >= 0, 'invalid cell count')
    return value


def reconcile(calls, terminal, decisions):
    """Check causal prediction/outcome pairing and reconstruct both posteriors."""
    alpha = dict(camera=1, radar=1)
    beta = dict(camera=1, radar=1)
    pending, rows, by_action = {}, [], {}
    call_ids = []
    old_quality = None
    for call in calls:
        call_ids.append(call['call_id'])
        require(digest(call['inputs']) == call['input_sha256'], 'input digest mismatch')
        require(digest(call['outputs']) == call['output_sha256'], 'output digest mismatch')
        out = call['outputs']
        if call['method'] == 'bootstrap':
            require(old_quality is None, 'duplicate bootstrap')
            old_quality = out['old_quality_proxy']
        if call['method'] == 'assess_local_action':
            prediction = out['observed_gain_prediction']
            if prediction is not None:
                aid = prediction['action_id']
                require(out['allowed'] and aid == len(rows) + 1,
                        'prediction does not precede the next paid action')
                require(aid not in pending, 'duplicate pending prediction')
                expected = {s: alpha[s] / (alpha[s] + beta[s]) for s in alpha}
                require(prediction['posterior_before'] == expected, 'prior mismatch')
                pending[aid] = prediction
        if call['method'] != 'compute_reward':
            continue
        aid = out['action_id']
        require(aid == len(rows) + 1 and aid in pending, 'missing or reordered prediction')
        require(out['accepted'] and not out['duplicate'] and out['action_cost'] == 1,
                'invalid paid observation')
        require(out['old_quality_proxy_before'] == old_quality, 'quality history mismatch')
        change = out['old_quality_proxy'] - old_quality
        require(math.isfinite(change) and change == out['parts']['fixed_old_quality_proxy_change'],
                'signed quality delta mismatch')
        old_quality = out['old_quality_proxy']
        g = out['observed_gain_calibration']
        require(g['action_id'] == aid and g['feedback_consumed_for_planning'],
                'gain/action mismatch or disabled calibration')
        p = pending.pop(aid)
        before = {s: alpha[s] / (alpha[s] + beta[s]) for s in alpha}
        for sensor in alpha:
            e = g[sensor]
            predicted = count(e['predicted_cells'])
            realized = count(e['realized_predicted_cells'])
            failure = count(e['unrealized_predicted_cells'])
            count(e['unpredicted_realized_cells'])
            require(predicted == realized + failure == p['predicted_' + sensor + '_cells'],
                    'prediction/outcome count mismatch')
            alpha[sensor] += realized
            beta[sensor] += failure
        after = {s: alpha[s] / (alpha[s] + beta[s]) for s in alpha}
        require(after == g['posterior_after'], 'posterior does not match measured counts')
        new_support = count(out['new_measured_support_count'])
        no_camera_trial = g['camera']['predicted_cells'] == 0
        require(not no_camera_trial or before['camera'] == after['camera'],
                'zero-trial posterior unexpectedly changed')
        row = dict(action_id=aid, new_measured_support_count=new_support,
            fixed_old_quality_proxy_change=change,
            camera=g['camera'], camera_posterior_before=before['camera'],
            camera_posterior_after=after['camera'], zero_camera_trials=no_camera_trial,
            measured_3d_proxy_changed=new_support > 0 or change != 0,
            conditional_feedback_recorded=out['conditional_gain_residual'] is not None)
        rows.append(row)
        by_action[aid] = g
    require(call_ids == list(range(1, len(call_ids) + 1)), 'module calls missing or duplicated')
    require(not pending, 'unconsumed action prediction')
    modules = terminal['modules']
    feedback = modules['feedback']
    gain = modules['observed_gain_calibration']
    require(feedback['paid_actions'] == len(rows) == gain['observed_actions'], 'terminal count mismatch')
    require(gain['alpha'] == alpha and gain['beta'] == beta, 'terminal posterior counts mismatch')
    require(gain['events'] == [by_action[a] for a in sorted(by_action)], 'terminal event mismatch')
    require(sum(r['new_measured_support_count'] for r in rows) == feedback['paid_measurement_support_count'],
            'terminal support count mismatch')
    descriptor_count = 0
    for d in decisions:
        aid = d['action_id']
        expected = dict(camera=.5, radar=.5) if aid == 0 else by_action[aid]['posterior_after']
        for desc, feature in zip(d['descriptors'], d['features']):
            descriptor_count += 1
            require(desc['camera_yield_posterior'] == expected['camera'] and
                    desc['radar_yield_posterior'] == expected['radar'], 'future or incorrect feature posterior')
            for mode in ('semantic', 'geometry'):
                require(feature[mode][7:9] == [expected['camera'], expected['radar']],
                        'feature posterior is not shared across candidates/controls')
        require(len(d['descriptors']) == len(d['features']), 'feature count mismatch')
    zero = [r for r in rows if r['zero_camera_trials']]
    omitted = [r for r in zero if r['measured_3d_proxy_changed']]
    summary = dict(paid_actions=len(rows), decision_states=len(decisions), candidate_features=descriptor_count,
        zero_camera_trial_actions=len(zero), zero_trial_with_3d_proxy_change=len(omitted),
        zero_trial_with_new_support=sum(r['new_measured_support_count'] > 0 for r in zero),
        new_support_during_zero_trials=sum(r['new_measured_support_count'] for r in zero),
        total_new_measured_support=sum(r['new_measured_support_count'] for r in rows),
        zero_trial_positive_old_quality_changes=sum(r['fixed_old_quality_proxy_change'] > 0 for r in zero),
        zero_trial_negative_old_quality_changes=sum(r['fixed_old_quality_proxy_change'] < 0 for r in zero),
        zero_trial_signed_old_quality_change=math.fsum(r['fixed_old_quality_proxy_change'] for r in zero),
        conditional_feedback_events=sum(r['conditional_feedback_recorded'] for r in rows),
        conditional_posterior_rows=len(modules['conditional_gain_residual']['posteriors']),
        zero_trial_3d_change_action_ids=[r['action_id'] for r in omitted])
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / 'manifest.json').read_text())
    require(manifest['status'] == 'complete', 'source history is not complete')
    inventory = json.loads((args.source / 'artifact_hashes.json').read_text())
    for name, expected in inventory.items():
        require(sha(args.source / name) == expected, 'frozen input changed: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    source_names = [str(Path(__file__).resolve().relative_to(ROOT)),
        'nso/observed_feedback_v10.py', 'nso/observed_gain_calibration_v10_1.py',
        'nso/semantic_gain_v11.py', 'nso/cpu_four_modules_v10.py',
        'nso/semantic_opportunities_v14.py']
    sources = {n: sha(ROOT / n) for n in source_names}
    for n in source_names[1:]:
        require(sources[n] == manifest['source_sha256'][n], 'runtime source differs from frozen history')
    with zipfile.ZipFile(args.output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for n in source_names:
            z.write(ROOT / n, n)
    audit = dict(status='running', source_sha256=sources, input_root=str(args.source),
        input_inventory_sha256=sha(args.source / 'artifact_hashes.json'),
        evaluated_future_rewards_read=False, policy_modified=False, training_allowed=False)
    write(args.output / 'manifest.json', audit)
    summaries, all_rows = [], []
    try:
        seeds = manifest['protocol']['structure_seeds']
        for seed in seeds:
            folder = args.source / f'structure_{seed}'
            load = lambda n: json.loads((folder / n).read_text())
            summary, rows = reconcile(load('module_calls.json'), load('terminal.json'), load('all_decisions.json'))
            summaries.append(dict(structure_seed=seed, **summary))
            all_rows.extend(dict(structure_seed=seed, **row) for row in rows)
        write(args.output / 'actions.json', all_rows)
        result = dict(status='passed', summaries=summaries,
            paid_actions=sum(s['paid_actions'] for s in summaries), parent_layouts=1,
            zero_trial_with_3d_proxy_change=sum(s['zero_trial_with_3d_proxy_change'] for s in summaries),
            new_support_during_zero_trials=sum(s['new_support_during_zero_trials'] for s in summaries),
            independent_new_experiments=0, semantic_efficacy_proven=False,
            boundary='A 2D camera-yield channel misses some measured 3D changes. '
                     'IGCR separately records these proxies; they are not ground-truth accuracy, '
                     'unique surface area, or evidence of better semantic decisions. '
                     'Empty conditional feedback is expected for the current G policy.')
        write(args.output / 'result.json', result)
        for n, expected in sources.items():
            require(sha(ROOT / n) == expected, 'source changed during audit')
        audit['status'] = 'complete'
        print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}), flush=True)
    except Exception as error:
        audit.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'manifest.json', audit)
        write(args.output / 'artifact_hashes.json', {p.name: sha(p)
            for p in sorted(args.output.iterdir()) if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
