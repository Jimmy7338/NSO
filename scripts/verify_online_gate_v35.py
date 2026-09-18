#!/usr/bin/env python3
"""Independent saved V35 audit: arrays/arithmetic only, no policy execution."""
import ast
from collections import deque
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import ROOT, read, sha, write, freeze, seal, verify_sources, verify_inventory

SOURCE = ROOT/'audit_results/v35_saved_observation_gate_20260918'
OUT = ROOT/'audit_results/v35_saved_observation_review_20260918'
PIXELS = ROOT/'audit_results/v34_pixel_information_20260918'
OLD = ROOT/'audit_results/v33_direction_information_r1_20260917'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(left, right):
    require(math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10),
            f'numeric mismatch: {left} != {right}')


def digest_array(value):
    value = np.ascontiguousarray(value); digest = hashlib.sha256()
    for part in (value.dtype.str.encode(), str(value.shape).encode(), value.tobytes()):
        digest.update(len(part).to_bytes(8, 'big')); digest.update(part)
    return digest.hexdigest()


def digest_fields(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        for part in (name.encode(), value.dtype.str.encode(), str(value.shape).encode(), value.tobytes()):
            digest.update(len(part).to_bytes(8, 'big')); digest.update(part)
    return digest.hexdigest()


def potential(table, mask):
    areas = {str(k): 0. for k in table['exact_reference_vertical_areas']}
    for index, target in enumerate(table['targets']):
        if mask & (1 << index):
            areas[str(target['owner'])] += target['area']
    surface = sum(min(1., max(0., area/table['exact_reference_vertical_areas'][key]))
                  for key, area in areas.items())/len(areas)
    coverage = (mask >> table['target_bits']).bit_count()/len(table['floor_cells'])
    return dict(surface=surface, coverage=coverage, joint=surface*coverage,
                feasible=coverage >= .8-1e-12)


def residual(raw, templates, node, cfg):
    losses = []; diagnostics = {}
    for field, minimum in (('depth', 'minimum_depth_samples'), ('ranges', 'minimum_scan_samples')):
        observed = raw[field].astype(float)
        predicted = [value[field][node].astype(float) for value in templates]
        use = (np.abs(predicted[0]-predicted[1]) > cfg['template_difference_m']) & (observed > 0)
        count = int(use.sum()); label = 'depth' if field == 'depth' else 'scan'
        diagnostics[label+'_samples'] = count
        diagnostics[label+'_mean_losses'] = None
        if count < cfg[minimum]:
            continue
        pair = []
        for prediction in predicted:
            if field == 'depth':
                valid = prediction[use] > 0
                item = np.full(count, cfg['loss_cap'])
                difference = (cfg['stereo_fb_px_m']/observed[use][valid]
                    - cfg['stereo_fb_px_m']/prediction[use][valid])/cfg['stereo_sigma_disparity_px']
                item[valid] = np.minimum(.5*difference*difference, cfg['loss_cap'])
            else:
                difference = (observed[use]-prediction[use])/cfg['scan_sigma_m']
                item = np.minimum(.5*difference*difference, cfg['loss_cap'])
            pair.append(float(item.mean()))
        diagnostics[label+'_mean_losses'] = pair
        losses.append(pair[1]-pair[0])
    score = max(-cfg['pose_log_odds_cap'], min(cfg['pose_log_odds_cap'],
                sum(losses)/len(losses))) if losses else 0.
    diagnostics.update(log_likelihood_ratio=score, usable_modalities=len(losses))
    return score, diagnostics


def same_tree(actual, expected):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'dictionary fields differ')
        for key in expected: same_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'list length differs')
        for a, b in zip(actual, expected): same_tree(a, b)
    elif isinstance(expected, float): close(actual, expected)
    else: require(actual == expected, 'value differs')


def verify_selection(plan, node, left, p0, edges, distances, informative, visited, feedback):
    close(plan['probability0'], p0)
    require(plan['remaining'] == left and plan['geometry_feedback'] == feedback,
            'planner budget/feedback differs')
    require(plan['forecast_reliability'] == .99 and plan['both_template_terminal_qualification'],
            'forecast or qualification differs')
    candidates = plan['candidates']
    require([(r['action'], r['following']) for r in candidates] == edges[node], 'candidate graph differs')
    for row in candidates:
        nxt = row['following']
        require(row['return_feasible'] == (distances.get(nxt, 10**9) <= left-1), 'return gate differs')
        require(row['forecast_information'] == (feedback and nxt in informative and nxt not in visited),
                'future/visited information gate differs')
    best = max([plan['stop_value']] + [row['expected_proxy'] for row in candidates])
    close(best, plan['expected_proxy'])
    selected = None if plan['stop_value'] >= best-1e-12 else next(
        row['action'] for row in candidates if row['expected_proxy'] >= best-1e-12)
    require(plan['action'] == selected, 'saved argmax/tie-break differs')


def run():
    if OUT.exists(): raise FileExistsError('independent review is write-once')
    for folder in (SOURCE, PIXELS, OLD):
        verify_sources(folder); verify_inventory(folder)
    manifest = read(SOURCE/'manifest.json')
    for rel, expected in manifest['input_sha256'].items():
        require(sha(ROOT/rel) == expected, 'gate input changed: '+rel)
    result = read(SOURCE/'result.json')
    require(result['status'] == 'complete' and result['implementation_gate_passed'], 'source gate not passed')
    require(result['main_tasks_used'] == 19 and result['new_physical_tasks'] == 0, 'main count differs')
    cfg = manifest['likelihood_config']
    tables = read(OLD/'P00_geometry.json')['tables']
    edges = [[(a, int(n)) for a, n in row] for row in tables[0]['edges']]
    poses = tables[0]['poses']; anchor = tables[0]['anchor']
    require(tables[1]['poses'] == poses and tables[1]['edges'] == tables[0]['edges'], 'public graph differs')
    reverse = [[] for _ in edges]
    for n, links in enumerate(edges):
        for _, nxt in links: reverse[nxt].append(n)
    distances = {anchor: 0}; queue = deque([anchor])
    while queue:
        n = queue.popleft()
        for before in reverse[n]:
            if before not in distances:
                distances[before] = distances[n]+1; queue.append(before)
    raw = []
    for h in (0, 1):
        with np.load(PIXELS/f'P00_h{h}_pixels.npz', allow_pickle=False) as arrays:
            raw.append({k: arrays[k].copy() for k in ('depth', 'rgb', 'ranges')})
    forecasts = read(SOURCE/'public_forecast.json'); informative = set()
    for n, forecast in enumerate(forecasts):
        scores = [residual({k: raw[h][k][n] for k in raw[h]}, raw, n, cfg)[0] for h in (0, 1)]
        reliable = scores[0] >= math.log(99.) and scores[1] <= -math.log(99.)
        same_tree(forecast['public_clean_template_log_odds'], scores)
        require(forecast['node'] == n and forecast['pose'] == poses[n], 'forecast pose differs')
        require(forecast['informative'] == reliable, 'forecast information differs')
        require(forecast['correct_probability'] == (.99 if reliable else .5), 'forecast channel differs')
        if reliable: informative.add(n)
    rows = result['cached_rollouts']; require(len(rows) == 8, 'incomplete cache matrix')
    traces = {}; memo = 0; observations = 0; actions_count = 0; online = 0
    for row in rows:
        h, mode = row['hypothesis_fixture_only'], row['mode']
        trace = json.loads(gzip.decompress((SOURCE/row['trace_file']).read_bytes()))
        traces[h, mode] = trace
        same_tree(trace['result'], {k: v for k, v in row.items() if k != 'trace_file'})
        history, actions, plans, calls = trace['history'], row['actions'], trace['plans'], trace['calls']
        require(len(actions) == 42 and len(history) == 43 and len(plans) == 42, '42-step trace differs')
        require(len(calls) == 213, 'module call count differs')
        node = anchor; masks = [0, 0]; visited = set(); class_positions = {2: set(), 3: set()}
        geometry = 0.; semantic = 0.; offset = 0
        for step, item in enumerate(history):
            state, posterior = item['state'], item['posterior']
            require(state['step'] == step and state['node'] == node and state['pose'] == poses[node], 'paid state differs')
            require(state['remaining_budget'] == 42-step and state['measured_map_sha256'] is None, 'budget/map differs')
            observed = {k: raw[h][k][node] for k in raw[h]}
            for key in observed:
                require(posterior['observation_sha256'][key] == digest_array(observed[key]), 'wrong or future observation bytes')
            score, diagnostics = residual(observed, raw, node, cfg)
            same_tree(posterior['residuals'], diagnostics)
            new = node not in visited
            require(posterior['new_geometry_pose'] == new, 'repeat geometry differs')
            applied = score if new and mode != 'swapped_no_feedback' else 0.
            geometry = max(-24., min(24., geometry+applied))
            close(posterior['geometry_applied_log_odds'], applied)
            close(posterior['geometry_log_odds'], geometry)
            if mode != 'G':
                counts = {int(k): int(np.count_nonzero(np.all(observed['rgb'] == v, axis=-1)))
                          for k, v in cfg['marker_colors'].items()}
                supported = [k for k in counts if counts[k] >= 16]
                present = [k for k in counts if counts[k] > 0]
                selected = supported[0] if len(supported) == len(present) == 1 else None
                if selected is not None:
                    selected = 5-selected if mode.startswith('swapped') else selected
                    class_positions[selected].add(tuple(poses[node][:2]))
                if len(supported) == 2:
                    for k in class_positions: class_positions[k].add(tuple(poses[node][:2]))
                current = [k for k in class_positions if class_positions[k]]
                semantic = ((1 if current[0] == 2 else -1)*math.log(9.)
                    if len(current) == 1 and len(class_positions[current[0]]) >= 2 else 0.)
                same_tree(posterior['rgb_pixel_counts'], {str(k): v for k, v in counts.items()})
                require(posterior['observed_class_after_intervention'] == selected, 'class intervention differs')
            else:
                require(posterior['rgb_pixel_counts'] is None and semantic == 0., 'G used class')
            close(posterior['semantic_log_odds'], semantic)
            p0 = 1/(1+math.exp(-(geometry+semantic)))
            same_tree(posterior['probabilities'], [p0, 1-p0]); same_tree(state['probabilities'], [p0, 1-p0])
            previous = masks[:]
            masks = [m | int(t['observed_masks_hex'][node], 16) for m, t in zip(masks, tables)]
            require(state['masks_hex'] == [hex(m) for m in masks], 'mask OR differs')
            require(state['repeated_pose'] == (not new), 'repeated state differs')
            visited.add(node)
            potentials = [potential(t, m) for t, m in zip(tables, masks)]
            for saved, expected in zip(state['potential']['hypothesis_potential'], potentials):
                for key in expected: same_tree(saved[key], expected[key])
            close(state['potential']['probability_weighted_potential'], p0*potentials[0]['joint']+(1-p0)*potentials[1]['joint'])
            for got, t, old, current in zip(state['potential']['new_pose_potential_gain'], tables, previous, potentials):
                close(got, current['joint']-potential(t, old)['joint'])
            group = calls[offset:offset+(5 if step < 42 else 3)]
            require([c['module'] for c in group] == ['OV-SDF', 'IGCR', 'STGHP']+(['STGHP', 'RPN-UQ'] if step < 42 else []), 'module order differs')
            require(all(c['step'] == step for c in group), 'module consumed future step')
            require(group[0]['sensor_sha256'] == digest_fields(observed), 'OV-SDF input differs')
            same_tree(group[1]['probabilities'], [p0, 1-p0])
            require(group[2]['visited_nodes'] == sorted(visited) and group[2]['masks_hex'] == state['masks_hex'], 'topology uses future exposure')
            offset += len(group)
            if step < 42:
                plan = plans[step]; same_tree({k: v for k, v in group[3].items() if k not in ('module', 'operation')}, plan)
                require(plan['step'] == step and plan['from_node'] == node and plan['action'] == actions[step] == item['next_action'], 'decision timing differs')
                same_tree(plan['probabilities'], [p0, 1-p0])
                if step < 18:
                    require(dict(edges[node])[actions[step]] == tables[0]['prefix_nodes'][step+1], 'forced prefix differs')
                else:
                    verify_selection(plan['planning'], node, 42-step, p0, edges, distances, informative, visited, mode != 'swapped_no_feedback')
                    online += 1; memo += plan['planning']['memo_states']
                nxt = dict(edges[node])[actions[step]]
                require(plan['node'] == nxt and group[4]['node'] == nxt and group[4]['allowed'], 'local action differs')
                require(group[4]['return_actions'] == distances[nxt] and 1+distances[nxt] <= 42-step, 'local return budget differs')
                node = nxt
            else: require(item['next_action'] is None and node == anchor, 'exact stop/return differs')
        require(all(p['feasible'] for p in potentials), 'terminal coverage not qualified')
        for saved, expected in zip(row['final_potential'], potentials):
            for key in expected: same_tree(saved[key], expected[key])
        require(row['actual_C_map'] is None and row['actual_F1'] is None and not row['measured_efficacy'], 'cache score misclaimed physical')
        observations += len(history); actions_count += len(actions)
    require(set(traces) == {(h, m) for h in (0, 1) for m in ('G', 'S', 'swapped', 'swapped_no_feedback')}, 'matrix differs')
    interventions = result['correction_interventions']; require(len(interventions) == 2, 'two correction controls required')
    for entry in interventions:
        h, step = entry['hypothesis_fixture_only'], entry['step']; trace = traces[h, 'swapped']
        first = next(i for i, v in enumerate(trace['history']) if i >= 18 and v['posterior']['geometry_applied_log_odds'] != 0 and v['posterior']['probabilities'][h] > .5)
        require(step == first, 'intervention selected after first correction')
        row = trace['history'][step]; state, posterior = row['state'], row['posterior']
        require(entry['node'] == state['node'] and entry['masks_hex'] == state['masks_hex'], 'counterfactual state differs')
        close(entry['corrected_probability0'], posterior['probabilities'][0])
        prior = 1/(1+math.exp(-posterior['semantic_log_odds']))
        close(entry['semantic_only_probability0'], prior)
        visited = {r['state']['node'] for r in trace['history'][:step+1]}
        verify_selection(entry['forecast_intervention'], state['node'], 42-step, prior, edges, distances, informative, visited, True)
        require(entry['actual_action'] == row['next_action'] and entry['counterfactual_action'] == entry['forecast_intervention']['action'], 'correction action differs')
        require(entry['action_changed'] == (entry['actual_action'] != entry['counterfactual_action']) and entry['shared_future_geometry_feedback'], 'causal gate differs')
        memo += entry['forecast_intervention']['memo_states']
    for control in result['no_class_controls']:
        h = control['hypothesis_fixture_only']; baseline = traces[h, 'G']['history'][18]
        same_tree(control['probabilities'], [.5, .5])
        require(control['action'] == baseline['next_action'] and all(control[k] for k in ('masks_equal', 'probabilities_equal', 'action_equal')), 'no-class control differs')
        # Both diagnostic controllers have the same prefix masks, pose, prior
        # and forecast as G. Full erased-prefix trace was not saved; this
        # count uses that deterministic input equivalence, explicitly below.
        memo += traces[h, 'G']['plans'][18]['planning']['memo_states']
    for contrast in result['contrasts']:
        h = contrast['hypothesis_fixture_only']
        for key, pair in [('S_vs_G_first_action', ('S', 'G')), ('swapped_vs_S_first_action', ('swapped', 'S')), ('correction_vs_no_feedback_first_action', ('swapped', 'swapped_no_feedback'))]:
            a, b = [traces[h, m]['result']['actions'] for m in pair]
            first = next((i+1 for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            require(contrast[key] == first, 'action contrast differs')
    counts = result['counts']
    require(counts['controllers'] == 10 and counts['cached_observations'] == observations+38 == 382, 'observation count differs')
    require(counts['nominal_actions'] == actions_count+36 == 372, 'nominal action count differs')
    require(counts['online_planning_calls'] == counts['online_planning_attempts'] == online+2 == 194, 'online planner count differs')
    require(counts['correction_intervention_selects'] == counts['correction_intervention_attempts'] == 2, 'intervention count differs')
    require(counts['memo_states'] == memo == 305478, 'memo accounting differs')
    for key in ('new_worlds', 'renderer_queries', 'sensor_packets', 'mapper_updates', 'TSDF_integrations', 'Q_evaluations', 'new_main_tasks'):
        require(counts[key] == 0, 'unexpected physical execution')
    require(all(result['checks'].values()), 'source gate checks failed')
    # Inspect imports instead of importing/executing controller or planner.
    for name in ('cpu_four_modules_v35.py', 'online_planner_v35.py', 'observation_belief_v35.py'):
        tree = ast.parse((ROOT/'nso'/name).read_text())
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                       [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            require(not any(m.startswith('env.') or 'measurement' in m or 'mapping' in m for m in modules), 'controller imports world/metric/mapper')
    OUT.mkdir()
    freeze(OUT, [Path(__file__)], source_gate_result_sha256=sha(SOURCE/'result.json'),
           input_sha256={str(p.relative_to(ROOT)): sha(p) for p in SOURCE.iterdir() if p.is_file()})
    report = dict(status='passed', saved_evidence_verified=True,
        source_result_sha256=sha(SOURCE/'result.json'), source_inventory_sha256=sha(SOURCE/'artifact_hashes.json'),
        verified_full_traces=8, verified_saved_observations=344, verified_public_forecasts=len(forecasts),
        verified_posterior_interventions=2, verified_no_class_receipts=2,
        gate_count_reconciliation=counts, main_tasks_used=19,
        reviewer_execution=dict(controller_calls=0, planner_calls=0, DP_states=0, World=0, sensor_queries=0, mapper=0, TSDF=0, Q=0, new_main_tasks=0),
        limitations=['Planner branch values were not recomputed; saved candidate argmax, soft-information eligibility and independent toy-test source binding were checked.',
          'Two erased-prefix controls save summary receipts, not complete per-step traces; 38 reads and 2 memo counts are reconciled by frozen source and deterministic input equivalence to G.',
          'Static source isolation and saved module timing support the interface contract; this is not an adversarial sandbox proof.',
          'Cached clean observations are not new physical or measured online efficacy evidence.'])
    write(OUT, OUT/'result.json', report); seal(OUT)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': run()
