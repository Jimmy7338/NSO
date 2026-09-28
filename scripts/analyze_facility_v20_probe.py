#!/usr/bin/env python3
"""Read-only verification and plots for a complete frozen V20 N4 probe.

Never starts a simulator. Requires four completed cases and four passed
independent-process replays. Remaining-plan changes are not causal savings.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'eval_results/facility_v20_coverage_probe_20260915'
DEFAULT_OUTPUT = ROOT / 'audit_results/facility_v20_probe_analysis_20260915'
NOTES = [
    'Four physical trajectories in two parents with paired assignments. Four replays are equality checks, not four additional experiments.',
    'N throughout is a common online coverage feasibility probe. It does not test semantic information value or establish full-architecture efficacy.',
    'C2D is evaluation coverage recorded each paid action. C_plan is the online observed safe/possible-cell ratio reconstructed from log counts, not ground truth or a calibrated confidence bound.',
    'Q_external is 5 cm external-surface F1 averaged without class weights over all six facilities, including zero for missing assets. Primary reference seed: 2026.',
    'Quality is recorded every 20 actions plus the terminal observation. Lines joining these points are visual guides; no unobserved quality values are generated.',
    'Reference seeds 2026/2027/2028 measure sampling sensitivity on each same trajectory. They are not independent experiments or a confidence interval.',
    'First C2D >= 0.8 is not endpoint eligibility. Endpoint coverage, return, collision, failure and budget are checked separately.',
    'Observed frontier groups are connected components of observed frontiers, not true branches or room labels.',
    'For retained targets, old_remaining_outbound minus new_remaining_outbound describes a current-plan change. Sums across updates are not executed actions saved or an isolated causal contribution. Cancellations are not savings.',
    'Historical V19 continue_coverage installed only the first coverage option, followed by common G continuation. V20 uses N throughout, revised candidates and replanning. Historical differences describe different planning systems, not an isolated component effect.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(a, b, label):
    require(a is None and b is None or a is not None and b is not None and
            math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12),
            f'{label}: {a!r} != {b!r}')


def inside(folder, name):
    path = (folder / name).resolve()
    require(path.is_relative_to(folder.resolve()), 'Artifact path escapes evidence: ' + name)
    return path


def verify_inventory(folder):
    inventory = read(folder / 'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != 'artifact_hashes.json'}
    require(set(inventory) == actual, 'Artifact inventory differs: ' + str(folder))
    for name, wanted in inventory.items():
        require(sha(inside(folder, name)) == wanted, 'Artifact changed: ' + str(folder / name))
    return len(inventory)


def verify_source(source):
    m = read(source / 'manifest.json')
    # This gate precedes all output-directory creation.
    require(m.get('status') == 'complete', 'Formal analysis requires manifest status complete')
    require(m.get('mode') == 'N', 'Bounded analyzer requires N throughout')
    c = m['config']; cases = m['cases']
    expected = [(p, a) for p in ['D19-P00', 'D19-P01']
                for a in ['A_complex_B_simple', 'A_simple_B_complex']]
    require([(v['parent'], v['assignment']) for v in cases] == expected and
            [v['index'] for v in cases] == list(range(4)), 'Exactly four declared paired cases required')
    require(c['primary_reference_seed'] == 2026 and c['reference_seeds'] == [2026, 2027, 2028], 'Reference seeds changed')
    require(c['coverage_minimum'] == .8 and c['quality_curve_action_stride'] == 20, 'Gate or quality stride changed')
    require(c['initial_history_actions'] == 0 and not c['install_v19_declared_first_option'], 'Unexpected paid prefix or first-option intervention')
    require(not m['training_allowed'] and not m['semantic_information_test'], 'Unexpected training or semantic test')
    for case in cases:
        require(case['mode'] == 'N' and case['budget'] == {'D19-P00': 160, 'D19-P01': 184}[case['parent']], 'Mode or budget changed')
        require(case['initial_history_actions'] == 0 and not case['declared_first_option_installed'], 'Case prefix/first option differs')
        v = read(source / f'case_{case["index"]:02d}/verification.json')
        require(v.get('status') == 'passed' and v.get('independent_process') is True and
                v.get('fresh_world_sensor_action_metric_replay') is True and
                v['physical_process_id'] != v['replay_process_id'], 'Four passed fresh-process replays required')
    count = verify_inventory(source)
    for case in cases:
        verify_inventory(source / f'case_{case["index"]:02d}')
    frozen = m['source_sha256']
    with zipfile.ZipFile(source / 'sources.zip') as archive:
        require(set(archive.namelist()) == set(frozen), 'Frozen source archive inventory differs')
        for name, wanted in frozen.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == wanted, 'Frozen source changed: ' + name)
    changed = [name for name, wanted in frozen.items() if not (ROOT / name).is_file() or sha(ROOT / name) != wanted]
    historical = Path(m['reference_source_root']).resolve()
    require(sha(historical / 'manifest.json') == m['reference_source_manifest_sha256'], 'Historical V19 manifest changed')
    require(sha(historical / 'artifact_hashes.json') == m['reference_source_inventory_sha256'], 'Historical V19 inventory changed')
    require(len(m['reference_sha256']) == 12, 'Exactly 12 frozen reference caches required')
    for name, wanted in m['reference_sha256'].items():
        require(sha(inside(historical, name)) == wanted, 'Reference bytes changed: ' + name)
    return m, dict(artifact_hashes_verified=count, frozen_source_files_verified=len(frozen),
        reference_files_verified=12, current_workspace_source_mismatches=changed,
        source_archive_is_authoritative=True, source_manifest_sha256=sha(source / 'manifest.json'),
        source_inventory_sha256=sha(source / 'artifact_hashes.json'))


def identity(case):
    return {k: case[k] for k in ('index', 'parent', 'assignment', 'mode', 'budget')}


def analyze_case(folder, case, config):
    """Analyze recorded facts without modifying evidence or simulating sensors."""
    r = read(folder / 'result.json'); calls = read(folder / 'module_calls.json.gz')
    audit = read(folder / 'runtime_audit.json.gz'); replay = read(folder / 'verification.json')
    timing = read(folder / 'timing.json'); base = identity(case); n = r['paid_actions']
    require(r['case'] == case, 'Case identity differs')
    require(digest([{k: v for k, v in x.items() if k != 'elapsed_s'} for x in calls]) == r['module_calls_sha256'], 'Module log digest differs')
    require(digest(audit) == r['runtime_audit_sha256'], 'Runtime log digest differs')
    require([x['call_id'] for x in calls] == list(range(1, len(calls) + 1)), 'Module call IDs not contiguous')
    for call in calls:
        require(digest(call['inputs']) == call['input_sha256'] and digest(call['outputs']) == call['output_sha256'], 'Module I/O digest differs')
        require(call['truth_used'] is False and call['trained'] is False, 'Unexpected module truth/training flag')
    require({x['module'] for x in calls} == {'OV-SDF', 'STGHP', 'RPN-UQ', 'IGCR'}, 'Four-module logs missing/unexpected')
    require(len(r['actions']) == n == replay['actions'] == r['termination']['final_action_id'], 'Physical/replay/terminal action counts differ')
    require(replay['physical_process_id'] == timing['process_id'], 'Physical process receipt differs')
    require(r['primitive_budget_compliant'] == (n <= case['budget']), 'Budget flag inconsistent')
    import numpy as np
    with np.load(folder / 'final_mesh.npz', allow_pickle=False) as mesh:
        require(set(mesh.files) == set(r['final_mesh_sha256']), 'Mesh fields differ')
        for key, wanted in r['final_mesh_sha256'].items():
            arr = np.ascontiguousarray(mesh[key])
            actual = hashlib.sha256(f'{arr.dtype.str}:{arr.shape}:'.encode() + arr.tobytes()).hexdigest()
            require(actual == wanted, 'Final mesh array digest differs: ' + key)
    trace = r['coverage_trace']
    require([x['action_id'] for x in trace] == list(range(n + 1)), 'C2D must be recorded every action')
    boots = [x for x in calls if x['method'] == 'coverage_bootstrap_v20']
    observations = [x for x in calls if x['method'] == 'observe_unique_coverage_v20']
    require(len(boots) == 1 and boots[0]['action_id'] == 0, 'Exactly one action-zero bootstrap required')
    require([x['action_id'] for x in observations] == list(range(1, n + 1)), 'Ledger updates missing/duplicated')
    b = boots[0]['outputs']; safe = b['S_count']; possible = b['P_count']
    close(b['C_plan'], safe / possible if possible else None, 'Bootstrap C_plan')
    steps = [{**base, 'action_id': 0, 'coverage_2d': trace[0]['coverage_2d'],
              'C_plan': b['C_plan'], 'safe_cells': safe, 'possible_cells': possible}]
    for call, actual in zip(observations, trace[1:]):
        x = call['outputs']
        require(x['action_id'] == call['action_id'], 'Ledger action mismatch')
        require(x['possible_cells_before'] == possible, 'Possible-cell chain broken')
        safe += x['safe_added_cells'] - x['safe_lost_cells']; possible = x['possible_cells_after']
        require(0 <= safe <= possible, 'Invalid safe/possible counts')
        steps.append({**base, 'action_id': actual['action_id'], 'coverage_2d': actual['coverage_2d'],
            'C_plan': safe / possible if possible else None, 'safe_cells': safe, 'possible_cells': possible,
            **{k: x.get(k) for k in ('coverage_intent', 'actual_new_known_cells', 'known_lost_cells',
                'safe_added_cells', 'safe_lost_cells', 'predicted_union_cells', 'realized_predicted_cells',
                'unrealized_predicted_cells', 'unpredicted_realized_cells')}})
    selections = []; candidates = []; refreshes = []; crosschecks = 0
    for call in calls:
        x = call['outputs']; method = call['method']; action = call['action_id']
        state = (x.get('coverage_state') if method == 'select_topo_target' else
                 x.get('coverage_budget') if method == 'refresh_observed_route_v20' else None)
        if state is not None:
            close(state['C_plan'], steps[action]['C_plan'], f'C_plan snapshot at {action}')
            for logged, calc in [('S_count', 'safe_cells'), ('P_count', 'possible_cells')]:
                if logged in state:
                    require(state[logged] == steps[action][calc], 'Count snapshot differs: ' + logged)
            crosschecks += 1
        if method == 'select_topo_target':
            require(x['mode'] == 'N' and x['effective_objective'] == 'N', 'Probe objective changed')
            ca = x['candidate_audit']['coverage']; selected = x['selected']; selected_row = None
            require(not ca.get('class_used', False) and not ca.get('truth_used', False), 'Coverage candidates used class/truth')
            for row in ca.get('selected', []):
                chosen = selected is not None and row['pose'] == selected['pose'] and selected['group'].startswith('coverage_')
                if chosen:
                    selected_row = row
                candidates.append({**base, 'action_id': action, 'selection_call_id': call['call_id'], 'chosen': chosen, **row})
            score_row = next((s for s in x['score_audit'] if selected and s['candidate_id'] == selected['candidate_id']), None)
            if selected:
                require(selected['group'].startswith('coverage_') and selected_row is not None, 'N selection not audited coverage candidate')
            selections.append({**base, 'action_id': action, 'selection_call_id': call['call_id'],
                'option_id': selected.get('option_id') if selected else None,
                'selected_pose': selected.get('pose') if selected else None,
                'selected_group': selected.get('group') if selected else None,
                'candidate_count_total': len(x['candidates']), 'coverage_candidate_count': ca.get('candidate_count', 0),
                'observed_frontier_group_count': len(ca.get('frontier_groups', [])),
                'represented_frontier_group_count': len(ca.get('represented_frontier_groups', [])),
                'represented_frontier_groups': ca.get('represented_frontier_groups', []),
                'selection_reason': selected_row.get('selection_reason') if selected_row else None,
                'selected_frontier_group': selected_row.get('frontier_group') if selected_row else None,
                'endpoint_predicted_unknown_cells': selected_row.get('gain') if selected_row else None,
                'route_union_predicted_unknown_cells': score_row['v20_coverage']['unique_predicted_unknown_cells'] if score_row else None,
                'outbound_cost': selected.get('outbound_cost') if selected else None,
                'return_cost': selected.get('return_cost') if selected else None, 'C_plan': steps[action]['C_plan']})
        elif method == 'refresh_observed_route_v20':
            retained = x['target_retained']
            require((x['new_remaining_outbound'] is not None) == retained, 'Retained route/cost mismatch')
            delta = x['old_remaining_outbound'] - x['new_remaining_outbound'] if retained else None
            refreshes.append({**base, **x, 'planned_outbound_reduction': delta, 'actual_causal_actions_saved': None})
    require([{k: v for k, v in x.items() if k != 'event'} for x in audit if x['event'] == 'v20_route_update'] ==
            [x['outputs'] for x in calls if x['method'] == 'refresh_observed_route_v20'], 'Runtime/module refresh logs differ')
    auth = [x for x in audit if x['event'] == 'paid_action_authorized']
    obs = [x for x in audit if x['event'] == 'observation']
    require([x['next_action_id'] for x in auth] == list(range(1, n + 1)), 'Per-action authorization missing')
    require([x['action_id'] for x in obs] == list(range(1, n + 1)), 'Runtime observation missing')
    margins = []
    for i, (a, o, taken) in enumerate(zip(auth, obs, r['actions']), 1):
        assessment = a['assessment']; remaining = case['budget'] - i + 1
        require(assessment['allowed'] is True and assessment['remaining_budget'] == remaining, 'Authorization/budget mismatch')
        require(assessment['action'] == taken['action'], 'Authorized/executed action differs')
        require(o['packet_sha256'] == taken['packet_sha256'] and o['collision'] == taken['collision'], 'Observation receipt differs')
        if not taken['collision']:
            require(assessment['next_pose'] == taken['pose'], 'Authorized/executed pose differs')
        reserved = assessment['reserved_return_cost']
        require(reserved is not None and reserved <= remaining - 1, 'Action violates logged return reserve')
        margins.append(remaining - 1 - reserved)
        steps[i].update(action=taken['action'], collision=taken['collision'], phase=a['phase'],
                        reserved_return_cost=reserved, return_reserve_margin=margins[-1])
    require(sum(int(a['collision']) for a in r['actions']) == r['collisions'], 'Collision count differs')
    curve = r['quality_curve']; stride = config['quality_curve_action_stride']; primary = str(config['primary_reference_seed'])
    require([x['action_id'] for x in curve] == sorted(set(range(0, n + 1, stride)) | {n}), 'Quality checkpoints differ')
    quality = []
    for x in curve:
        metric = x['metrics']; action = x['action_id']
        close(metric['coverage_2d'], steps[action]['coverage_2d'], 'Quality/C2D alignment')
        close(metric['05cm']['joint_external'], metric['coverage_2d'] * metric['05cm']['external_macro_f1'], 'Joint score')
        quality.append({**base, 'action_id': action, 'reference_seed': int(primary),
            'coverage_2d': metric['coverage_2d'], 'external_macro_f1': metric['05cm']['external_macro_f1'],
            'joint_external': metric['05cm']['joint_external'], 'eligible': metric['eligible'],
            'returned': metric['returned'], 'collisions': metric['collisions'], 'failed': metric['failed']})
    require(curve[0]['metrics'] == r['before'][primary] and curve[-1]['metrics'] == r['after'][primary], 'Quality endpoint differs')
    reached = [x['action_id'] for x in trace if x['coverage_2d'] >= config['coverage_minimum']]
    require(r['first_evaluator_coverage_gate_action'] == (min(reached) if reached else None), 'First gate action differs')
    endpoints = []
    for seed in config['reference_seeds']:
        metric = r['after'][str(seed)]; before = r['before'][str(seed)]
        close(metric['coverage_2d'], trace[-1]['coverage_2d'], 'Final C across reference samples')
        eligible = metric['coverage_2d'] >= config['coverage_minimum'] and metric['returned'] and metric['collisions'] == 0 and not metric['failed']
        require(metric['eligible'] == eligible, 'Metric eligibility inconsistent')
        require(metric['returned'] == r['termination']['returned_to_anchor'] and metric['failed'] == r['termination']['failed'] and metric['collisions'] == r['collisions'], 'Metric/termination safety differs')
        endpoints.append({**base, 'reference_seed': seed, 'paid_actions': n,
            'coverage_2d': metric['coverage_2d'], 'external_macro_f1': metric['05cm']['external_macro_f1'],
            'joint_external': metric['05cm']['joint_external'], 'eligible': metric['eligible'],
            'initial_coverage_2d': before['coverage_2d'], 'initial_external_macro_f1': before['05cm']['external_macro_f1']})
    final = next(x for x in endpoints if x['reference_seed'] == int(primary))
    deltas = [x['planned_outbound_reduction'] for x in refreshes if x['target_retained']]
    summary = {**final, 'C_plan_final': steps[-1]['C_plan'],
        'first_evaluator_coverage_gate_action': r['first_evaluator_coverage_gate_action'],
        'termination': r['termination'], 'early_stop': n < case['budget'],
        'primitive_budget_compliant': r['primitive_budget_compliant'], 'path_distance_m': r['path_distance_m'],
        'collisions': r['collisions'], 'safety_authorizations_checked': n,
        'minimum_logged_return_reserve_margin': min(margins) if margins else None,
        'independent_replay': replay, 'module_call_counts': dict(Counter(x['module'] for x in calls)),
        'C_plan_snapshot_crosschecks': crosschecks, 'selection_count': len(selections),
        'selected_candidate_generation_reasons': dict(Counter(x['selection_reason'] if x['selection_reason'] is not None else 'no_selection' for x in selections)),
        'selections_with_multiple_observed_frontier_groups': sum(x['represented_frontier_group_count'] > 1 for x in selections),
        'route_refresh_count': len(refreshes), 'route_refresh_reasons': dict(Counter(x['reason'] for x in refreshes)),
        'route_refresh_retained': len(deltas), 'route_refresh_cancelled': len(refreshes) - len(deltas),
        'retained_plan_shortening_events': sum(d > 0 for d in deltas),
        'retained_plan_lengthening_events': sum(d < 0 for d in deltas),
        'retained_plan_unchanged_events': sum(d == 0 for d in deltas),
        'retained_plan_positive_reductions_sum': sum(max(d, 0) for d in deltas),
        'retained_plan_lengthenings_sum': sum(max(-d, 0) for d in deltas), 'actual_causal_actions_saved': None,
        'runtime_denial_reasons': dict(Counter(x['denial']['reason'] for x in audit if x['event'] == 'v14_route_denied')),
        'timing': timing}
    return dict(summary=summary, steps=steps, quality=quality, endpoints=endpoints,
                selections=selections, candidates=candidates, refreshes=refreshes)


def historical_context(manifest, summaries):
    folder = Path(manifest['reference_source_root']); m = read(folder / 'manifest.json')
    inventory = read(folder / 'artifact_hashes.json'); rows = []
    for current in summaries:
        matches = [c for c in m['cases'] if c['parent'] == current['parent'] and c['assignment'] == current['assignment'] and c['option'] == 'continue_coverage']
        require(len(matches) == 1, 'Exactly one V19 first-coverage case per world required')
        name = f'case_{matches[0]["index"]:02d}/result.json'
        require(sha(folder / name) == inventory[name], 'Historical result changed: ' + name)
        old = read(folder / name); metric = old['after'][str(manifest['config']['primary_reference_seed'])]
        row = {**{k: current[k] for k in ('index', 'parent', 'assignment')},
            'historical_case_index': matches[0]['index'], 'historical_result_sha256': inventory[name],
            'historical_policy': 'V19 coverage first option, then G continuation',
            'current_policy': 'V20 N throughout with revised candidates and replanning',
            'v19_paid_actions': old['paid_actions'], 'v20_paid_actions': current['paid_actions'],
            'v19_eligible': metric['eligible'], 'v20_eligible': current['eligible'], 'causal_comparison': False}
        for key in ('coverage_2d', 'external_macro_f1', 'joint_external'):
            value = metric[key] if key == 'coverage_2d' else metric['05cm'][key]
            row.update({f'v19_{key}': value, f'v20_{key}': current[key], f'difference_{key}': current[key] - value})
        rows.append(row)
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def make_plots(output, analyzed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 160})
    for kind in ('coverage', 'quality'):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
        for ax, data in zip(axes.flat, analyzed):
            s = data['summary']; rows = data['steps' if kind == 'coverage' else 'quality']
            actions = [r['action_id'] for r in rows]
            series = ([('coverage_2d', 'Actual C2D (evaluation)', '#2466a6', '-'), ('C_plan', 'C_plan (logged S / P)', '#db8125', '--')]
                if kind == 'coverage' else [('external_macro_f1', 'Q external, 5 cm', '#277d61', '-'), ('joint_external', 'J = C2D x Q external', '#7861a8', '--')])
            for key, label, color, style in series:
                ax.plot(actions, [r[key] for r in rows], style, color=color, linewidth=1.8,
                        marker='o' if kind == 'quality' else None, markersize=3, label=label)
                ax.scatter(actions[-1], rows[-1][key], marker='s' if s['early_stop'] else 'o', facecolors='white', edgecolors=color, s=42, zorder=4)
            if kind == 'coverage':
                ax.axhline(.8, color='#666666', linestyle=':', linewidth=1, label='Coverage gate 0.80')
            ax.axvline(s['budget'], color='#aaaaaa', linewidth=.8, linestyle=':')
            assignment = 'A complex / B simple' if s['assignment'] == 'A_complex_B_simple' else 'A simple / B complex'
            ax.set_title(s['parent'] + ' | ' + assignment)
            ax.text(.02, .97, f"{'Early stop' if s['early_stop'] else 'Budget endpoint'} {s['paid_actions']}/{s['budget']} | eligible={s['eligible']}", transform=ax.transAxes, va='top', fontsize=9)
            ax.set_xlim(0, 188); ax.set_ylim(0, 1); ax.grid(alpha=.2)
            ax.set_xlabel('Paid primitive actions'); ax.set_ylabel('Coverage fraction' if kind == 'coverage' else 'Exterior quality / joint score')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .97), ncol=len(labels), frameon=False)
        fig.suptitle('V20 N throughout: ' + ('actual coverage and online proxy' if kind == 'coverage' else 'primary-reference exterior reconstruction'), y=1.0)
        note = ('Every action observed; C_plan is an online proxy, not a coverage guarantee.' if kind == 'coverage' else
                'Recorded 20-action checkpoints + terminal only; lines are guides. Seed 2026; no confidence intervals.')
        fig.text(.5, .013, note + ' Endpoint: circle = budget, square = early stop.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .90)); fig.savefig(output / f'{kind}_progress.png', bbox_inches='tight'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE, help='Completed frozen N4 evidence directory')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='New directory; must not already exist')
    p.add_argument('--verify-only', action='store_true', help='Validate in memory; write no artifacts or figures')
    args = p.parse_args(); source = args.source.resolve(); output = args.output.resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Analysis and source trees must be separate')
    manifest, checks = verify_source(source)
    analyzed = [analyze_case(source / f'case_{c["index"]:02d}', c, manifest['config']) for c in manifest['cases']]
    summaries = [x['summary'] for x in analyzed]; aggregate = read(source / 'result.json')
    require(aggregate['status'] == 'complete' and aggregate['unique_physical_trajectories'] == 4 and aggregate['independent_process_replays'] == 4, 'Complete four-case aggregate required')
    for recorded, calculated in zip(aggregate['cases'], summaries):
        for key in ('index', 'parent', 'assignment', 'mode', 'budget', 'paid_actions', 'coverage_2d', 'external_macro_f1', 'joint_external', 'eligible', 'collisions', 'primitive_budget_compliant', 'first_evaluator_coverage_gate_action'):
            require(recorded[key] == calculated[key], 'Aggregate/case mismatch: ' + key)
    context = historical_context(manifest, summaries)
    summary = dict(status='complete', source_root=str(source), checks=checks,
        unique_physical_trajectories=4, independent_process_replays=4,
        total_physical_actions=sum(x['paid_actions'] for x in summaries), eligible_count=sum(x['eligible'] for x in summaries),
        common_online_coverage_passed=all(x['eligible'] and x['primitive_budget_compliant'] for x in summaries),
        semantic_information_test=False, full_architecture_efficacy_proven=False,
        reference_sampling_repeats_are_independent_experiments=False,
        C_plan_reconstruction='Bootstrap S/P, then safe_added-safe_lost and possible_after per observation; crosschecked against selection/refresh snapshots. Raw map masks not reconstructed.',
        fresh_physical_simulation_executed_by_analyzer=False, cases=summaries,
        historical_v19_descriptive_context=context, interpretation=NOTES)
    compact = {k: summary[k] for k in ('status', 'unique_physical_trajectories', 'independent_process_replays', 'total_physical_actions', 'eligible_count', 'common_online_coverage_passed')}
    if args.verify_only:
        print(json.dumps({**compact, 'artifacts_written': False, 'checks': checks}, indent=2)); return
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, key in [('coverage_trace', 'steps'), ('quality_checkpoints', 'quality'), ('endpoints', 'endpoints'), ('candidate_selections', 'selections'), ('coverage_candidates', 'candidates'), ('route_updates', 'refreshes')]:
            write_csv(output / (name + '.csv'), [row for case in analyzed for row in case[key]])
        write_csv(output / 'historical_v19_context.csv', context); make_plots(output, analyzed)
        source_name = str(Path(__file__).resolve().relative_to(ROOT))
        with zipfile.ZipFile(output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__).resolve(), source_name)
        write_json(output / 'source_sha256.json', {source_name: sha(Path(__file__).resolve())})
        write_json(output / 'result.json', summary)
        write_json(output / 'artifact_hashes.json', {str(f.relative_to(output)): sha(f) for f in sorted(output.rglob('*')) if f.is_file() and f.name != 'artifact_hashes.json'})
    except Exception as error:
        write_json(output / 'analysis_failure.json', dict(status='failed', error=repr(error))); raise
    print(json.dumps({**compact, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f'Analysis refused: {error}', file=sys.stderr)
        raise SystemExit(2)
