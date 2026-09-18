#!/usr/bin/env python3
"""Independent saved V33 table/witness arithmetic; no models or DP are run."""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'audit_results/v33_direction_information_review_r1_20260917'
FAILED_REVIEW = ROOT/'audit_results/v33_direction_information_review_20260917'
CHECKER_FAILURE = ROOT/'audit_results/v33_direction_information_review_checker_failure_20260917'
DOC = ROOT/'docs/research/V33_DIRECTION_INFORMATION_REVIEW_20260917.md'
CAP, RESERVE, EPS = 2*1024**2, 64*1024**2, 1e-7
HEADINGS = ((0, 1), (1, 0), (0, -1), (-1, 0))
POLICIES = ('G', 'class_oracle', 'swapped_class_with_correction')
GENERATOR = ROOT/'scripts/generate_direction_scene_v33_r1.py'
GENERATOR_SHA = 'cfbb1ce7e9f459f16d02304d4f5ebd35a701b0599245175c4f0c5b1bcd014ec6'


def need(condition, why):
    if not condition:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, why):
    need(math.isfinite(a) and math.isfinite(b) and
         math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), why)


def provenance(folder):
    inv = read(folder/'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p != folder/'artifact_hashes.json'}
    need(actual == set(inv), 'exact artifact set: '+str(folder))
    for rel, expected in inv.items():
        need(sha(folder/rel) == expected, 'artifact bytes: '+rel)
    manifest = read(folder/'manifest.json')
    sources = manifest['source_sha256']
    need(sha(folder/'sources.zip') == manifest['source_archive_sha256'], 'source ZIP SHA')
    with zipfile.ZipFile(folder/'sources.zip') as z:
        need(len(z.namelist()) == len(sources) and set(z.namelist()) == set(sources),
             'exact source ZIP set')
        for rel, expected in sources.items():
            need(hashlib.sha256(z.read(rel)).hexdigest() == expected, 'archived source: '+rel)
            need(sha(ROOT/rel) == expected, 'current frozen source: '+rel)
        configs = [rel for rel in sources if rel.startswith('configs/') and rel.endswith('.json')]
        need(len(configs) == 1, 'one declared frozen configuration')
        config = json.loads(z.read(configs[0]))
    need(manifest['main_tasks_used'] == 16 and manifest['physical_execution_forbidden'], 'physical scope')
    need(manifest['thresholds_cm'] == {'2': None, '5': None, '10': None}, 'unmeasured precision thresholds')
    return manifest, config, dict(root=str(folder.relative_to(ROOT)), artifacts=len(inv),
        sources=len(sources), inventory_sha256=sha(folder/'artifact_hashes.json'),
        manifest_sha256=sha(folder/'manifest.json'), source_zip_sha256=sha(folder/'sources.zip'),
        source_sha256=sources)


def safe(a, b, boxes, radius):
    for bounds in boxes:
        distance2 = 0.
        for axis in (0, 1):
            lo, hi = sorted((a[axis], b[axis]))
            distance = max(bounds[2*axis]-hi, lo-bounds[2*axis+1], 0.)
            distance2 += distance*distance
        if distance2 < radius*radius-EPS:
            return False
    return True


def reachable(edges, anchor):
    distance = {anchor: 0}
    todo = deque([anchor])
    while todo:
        node = todo.popleft()
        for _, other in edges[node]:
            if other not in distance:
                distance[other] = distance[node]+1
                todo.append(other)
    return distance


def walk(table, actions):
    node = table['anchor']; nodes = [node]
    for action in actions:
        links = dict(table['edges'][node])
        need(action in links, 'illegal saved path action')
        node = links[action]; nodes.append(node)
    return nodes


def terminal(table, mask):
    n = table['target_bits']; patches = table['targets']
    denom = {str(k): v for k, v in table['exact_reference_vertical_areas'].items()}
    area = {k: 0. for k in denom}
    for i, patch in enumerate(patches):
        if mask & (1 << i):
            area[str(patch['owner'])] += patch['area']
    per_asset = {k: min(1., max(0., area[k]/denom[k])) for k in denom}
    coverage = (mask >> n).bit_count()/len(table['floor_cells'])
    surface = sum(per_asset.values())/len(per_asset)
    return dict(coverage=coverage, surface=surface, joint=coverage*surface,
                feasible=coverage >= .8-1e-12, per_asset_surface=per_asset)


def compare_terminal(actual, saved):
    for key in ('coverage', 'surface', 'joint'):
        close(actual[key], saved[key], 'terminal '+key)
    need(actual['feasible'] == saved['feasible'], 'terminal coverage qualification')
    need(set(actual['per_asset_surface']) == set(saved['per_asset_surface']), 'asset denominators')
    for key, value in actual['per_asset_surface'].items():
        close(value, saved['per_asset_surface'][key], 'per-asset visible fraction')


def table_check(parent, table, hypothesis):
    need(table['hypothesis'] == hypothesis, 'hypothesis index')
    poses, edges = table['poses'], table['edges']
    cells = sorted(map(tuple, parent['nav_cells']))
    need(poses == [[x, y, h] for x, y in cells for h in range(4)], 'complete poses')
    need(poses[table['anchor']] == parent['anchor'], 'anchor including heading')
    boxes = list(parent['background_boxes']) + [b for asset in parent['hypotheses'][hypothesis]['assets']
                                               for b in asset['boxes']]
    index = {tuple(p): i for i, p in enumerate(poses)}
    radius = parent['robot_radius_m']
    need(len(edges) == len(poses), 'edge table length')
    for i, pose in enumerate(poses):
        x, y, h = pose
        need(safe(pose, pose, boxes, radius), 'unsafe retained position')
        expected = {'left': index[(x, y, (h-1) % 4)], 'right': index[(x, y, (h+1) % 4)]}
        dx, dy = HEADINGS[h]; next_pose = (x+dx, y+dy, h)
        if next_pose in index and safe(pose, next_pose, boxes, radius):
            expected['forward'] = index[next_pose]
        need(len(edges[i]) == len(dict(edges[i])) and dict(edges[i]) == expected,
             'complete and safe primitive action row')
    # Enumerate only the declared small lattice, not sensor/world geometry.
    grid = parent['local_grid']; turns = parent['device_frame']['quarter_turns_ccw']
    expected_cells = []
    for x in range(grid['x_min'], grid['x_max']+1):
        for y in range(grid['y_min'], grid['y_max']+1):
            p = (x, y)
            for _ in range(turns % 4): p = (-p[1], p[0])
            if safe(p, p, boxes, radius): expected_cells.append(p)
    need(cells == sorted(expected_cells), 'all safe centres in declared rectangle')
    distances = reachable(edges, table['anchor'])
    connected = len(distances) == len(poses)
    need(table['all_safe_connected'] == connected and table['reachable_poses'] == len(distances),
         'saved connectivity claims')
    floor = [[x, y] for x, y in cells if index[(x, y, 0)] in distances]
    need(floor == table['floor_cells'] and floor, 'fixed reachable floor denominator')
    patches = table['targets']; n = table['target_bits']
    need(n == len(patches) > 0, 'target bit count')
    need(len({tuple(p['physical_key']) for p in patches}) == n, 'duplicate target physical keys')
    areas = {}
    for p in patches:
        need(p['vertical'] and p['owner'] is not None and p['area'] > 0, 'external vertical target contract')
        need(p['physical_key'] == p['point']+p['normal'], 'physical key excludes bookkeeping')
        close(sum(v*v for v in p['normal']), 1., 'unit normal')
        need(p['normal'][2] == 0, 'vertical target surface')
        owner = str(p['owner']); areas[owner] = areas.get(owner, 0.)+p['area']
    need(set(areas) == set(table['exact_reference_vertical_areas']), 'reference owner set')
    for owner, area in areas.items():
        close(area, table['exact_reference_vertical_areas'][owner], 'sum of saved quadrature areas')
    masks = [int(v, 16) for v in table['observed_masks_hex']]
    need(len(masks) == len(poses) == len(table['signature_sha256']) == len(table['signature_counts']),
         'observation table lengths')
    valid_mask = (1 << (n+len(floor)))-1; target_mask = (1 << n)-1
    for mask, sig, counts in zip(masks, table['signature_sha256'], table['signature_counts']):
        need(mask >= 0 and mask & ~valid_mask == 0, 'mask outside declared evidence universe')
        need(len(sig) == 64 and all(ch in '0123456789abcdef' for ch in sig), 'signature digest encoding')
        need(counts['target_points'] == (mask & target_mask).bit_count(), 'target support count')
        need(counts['visible_floor_centers'] == (mask >> n).bit_count(), 'floor support count')
        need(counts['camera_points'] >= counts['target_points'], 'target subset of camera signature')
    union = 0
    for node in distances: union |= masks[node]
    missing = [dict(index=i, **p) for i, p in enumerate(patches) if not union & (1 << i)]
    need(missing == table['unobservable_targets'], 'unobservable representatives from stored masks')
    need(table['all_target_representatives_observable'] == (not missing), 'observability gate')
    nodes = walk(table, parent['prefix_actions'])
    need(nodes == table['prefix_nodes'] and len(nodes) == 19 and nodes[-1] == nodes[0],
         'paid18 prefix and exact return')
    need([table['signature_sha256'][i] for i in nodes] == table['prefix_signatures'], 'prefix signature sequence')
    mask = 0
    for node in nodes: mask |= masks[node]
    need(mask == int(table['prefix_mask_hex'], 16), 'cumulative prefix includes initial frame')
    calculated = terminal(table, mask); compare_terminal(calculated, table['prefix_terminal'])
    need(table['semantic_cue_measured'] is False and table['geometric_pose_error'] is False, 'ideal scope')
    return dict(hypothesis=hypothesis, poses=len(poses), safe_floor_cells=len(floor),
        target_representatives=n, reference_vertical_areas=areas, prefix=calculated,
        unobservable_count=len(missing), unobservable_area_m2=sum(p['area'] for p in missing)), distances


def pair_check(parent, saved):
    tables = saved['tables']; need(len(tables) == 2, 'two hypotheses')
    checks = [table_check(parent, t, h) for h, t in enumerate(tables)]
    a, b = tables; p = saved['pair']; distances = checks[0][1]
    shared = a['poses'] == b['poses'] and a['edges'] == b['edges'] and a['anchor'] == b['anchor']
    paired = shared and a['prefix_signatures'] == b['prefix_signatures']
    need(p['identical_legal_graph'] == shared and p['identical_prefix_geometry'] == paired, 'paired input gates')
    connected = all(t['all_safe_connected'] for t in tables)
    need(p['all_safe_connected'] == connected, 'paired connectivity')
    different = [i for i in distances if a['signature_sha256'][i] != b['signature_sha256'][i]] if shared else []
    first = min((distances[i] for i in different), default=None)
    need(first == p['first_information_action_layer'], 'first information layer from stored signature table')
    expected_nodes = {i for i in different if distances[i] == first}
    records = p['first_information_witnesses']
    need(len(records) == len(expected_nodes) and {w['node'] for w in records} == expected_nodes,
         'all earliest separating nodes')
    for w in records:
        nodes = walk(a, w['actions'])
        need(len(w['actions']) == first and nodes[-1] == w['node'], 'earliest information witness path')
        need(w['pose'] == a['poses'][w['node']], 'information witness pose')
        need(w['signature_sha256'] == [t['signature_sha256'][w['node']] for t in tables], 'information signature pair')
    need(set(a['exact_reference_vertical_areas']) == set(b['exact_reference_vertical_areas']), 'paired asset set')
    equal_area = all(abs(a['exact_reference_vertical_areas'][k]-b['exact_reference_vertical_areas'][k]) < 1e-9
                     for k in a['exact_reference_vertical_areas'])
    gates = dict(shared_action_graph=shared, connected=connected, paired_prefix=paired,
                 all_targets_observable=all(t['all_target_representatives_observable'] for t in tables),
                 equal_mirror_area=equal_area)
    need(gates == saved['geometry_gates'] and saved['geometry_passed'] == all(gates.values()), 'geometry gate arithmetic')
    return dict(parent_id=parent['id'], geometry_gates=gates, hypotheses=[c[0] for c in checks],
                first_information_action_layer=first,
                signature_verification='Compared stored hashes; no raycast or raw signature payload regeneration.')


def witness_check(parent, tables, w):
    h = w['actual_hypothesis']; policy = w['policy']; table = tables[h]
    need(type(h) is int and h in (0, 1) and policy in POLICIES, 'witness identity')
    need(w['initial_hint'] == (1-h if policy.startswith('swapped') else None), 'class information authorization')
    actions = w['suffix_actions']; nodes = walk(table, actions)
    need(nodes == w['suffix_nodes'], 'witness saved action/node transitions')
    need(w['suffix_poses'] == [table['poses'][i] for i in nodes], 'witness saved poses')
    need(nodes[-1] == table['anchor'] and w['returned_exact_pose'], 'witness exact return')
    need(w['suffix_paid_actions'] == len(actions) <= 24, 'suffix cost')
    need(w['total_paid_actions'] == 18+len(actions) <= parent['total_action_budget'] == 42,
         'full prefix plus suffix paid budget')
    need(w['budget_compliant'], 'budget flag')
    mask = int(table['prefix_mask_hex'], 16)
    for node in nodes: mask |= int(table['observed_masks_hex'][node], 16)
    need(int(w['final_observed_mask_hex'], 16) == mask, 'actual final observed union')
    calculated = terminal(table, mask); compare_terminal(calculated, w['terminal'])
    need(w['actual_coverage_qualified'] == calculated['feasible'], 'actual qualification flag')
    if policy in ('G', 'class_oracle'): need(calculated['feasible'], 'primary witness must be qualified')
    different = [step for step, node in enumerate(nodes)
                 if tables[0]['signature_sha256'][node] != tables[1]['signature_sha256'][node]]
    first = min(different, default=None)
    need(w['first_information_suffix_action'] == first, 'first separating observation, including step0')
    sources = [('revealed_geometry' if first is not None and i >= first else
                'geometry_belief' if policy == 'G' else 'class_hypothesis') for i in range(len(actions))]
    need(sources == w['decision_sources'], 'action uses only prior actual observations')
    events = [dict(action=action, node=node, remaining=24-step,
                   geometric_hypothesis_revealed=step in different)
              for step, (action, node) in enumerate(zip(actions, nodes[1:]), 1)]
    need(events == w['events'], 'per-step budget and information events')
    return dict(hypothesis=h, policy=policy, suffix_paid_actions=len(actions),
        total_paid_actions=18+len(actions), first_information_suffix_action=first, terminal=calculated)


def policies_check(parent, geometry, row):
    need(row['parent_id'] == parent['id'] and row['exact_search_completed'], 'policy identity/completion claim')
    witnesses = row['witnesses']; checks = []; first_action_divergence = None
    first_information = None
    if row['feasible']:
        need(len(witnesses) == 6 and {(w['actual_hypothesis'], w['policy']) for w in witnesses} ==
             {(h, p) for h in (0, 1) for p in POLICIES}, 'six fixed policy witnesses')
        checks = [witness_check(parent, geometry['tables'], w) for w in witnesses]
        group = {(w['actual_hypothesis'], w['policy']): w for w in witnesses}
        g = sum(group[h, 'G']['terminal']['joint'] for h in (0, 1))/2
        known = [group[h, 'class_oracle']['terminal']['joint'] for h in (0, 1)]
        close(g, row['G_optimal'], 'mean G from actual branch witnesses')
        for h in (0, 1):
            close(known[h], row['class_optimal_by_hypothesis'][h], 'class conditional value')
            need(group[h, 'G']['terminal']['joint'] <= known[h]+1e-10, 'G witness not above known optimum claim')
        mean = sum(known)/2; close(mean, row['class_optimal_mean'], 'oracle mean')
        close(mean-g, row['information_absolute'], 'information absolute')
        relative = (mean-g)/g if g > 0 else None
        if relative is None: need(row['information_relative'] is None, 'zero denominator unavailable')
        else: close(relative, row['information_relative'], 'information relative')
        # Until the observation actually branches, G must share every action.
        a, b = (group[h, 'G'] for h in (0, 1))
        da, db = a['first_information_suffix_action'], b['first_information_suffix_action']
        first_information = da
        first_action_divergence = next((i+1 for i, (left, right) in
            enumerate(zip(a['suffix_actions'], b['suffix_actions'])) if left != right), None)
        if first_action_divergence is None and len(a['suffix_actions']) != len(b['suffix_actions']):
            first_action_divergence = min(len(a['suffix_actions']), len(b['suffix_actions']))+1
        need(da == db, 'common G first information time')
        if da is None:
            need(a['suffix_actions'] == b['suffix_actions'], 'nonrevealed G cannot condition on hidden identity')
        else:
            need(a['suffix_actions'][:da] == b['suffix_actions'][:da] and
                 a['suffix_nodes'][:da+1] == b['suffix_nodes'][:da+1], 'G shares prefix through information-producing action')
    else:
        need(not witnesses, 'infeasible policy cannot have main witness success')
        need(all(row[k] is None for k in ('G_optimal', 'class_optimal_by_hypothesis',
             'class_optimal_mean', 'information_absolute', 'information_relative')), 'infeasible values remain unavailable')
        g, mean, relative = None, None, None
    controls = row['controls']; ordinary = controls['identical_structure']
    ordinary_pass = ordinary['G'] > -5e29 and abs(ordinary['G']-ordinary['known']) <= 1e-10
    need(ordinary['passed'] == ordinary_pass, 'identical structure control equality')
    if row['feasible']: close(ordinary['known'], row['class_optimal_by_hypothesis'][0], 'same model known control')
    for actual, expected in ((controls['early_geometry_reveal']['value'], mean),
                             (controls['independent_class']['value'], g)):
        if expected is None: need(actual is None, 'infeasible algebraic control remains unavailable')
        else: close(actual, expected, 'algebraic control arithmetic')
    need(controls['swapped_with_geometric_correction']['values'] ==
         [w['terminal'] for w in witnesses if w['policy'].startswith('swapped')], 'swapped control retains qualification')
    passed = bool(ordinary_pass and relative is not None and relative > .05)
    need(row['screening_passed'] == passed, 'strict5percent finite gate')
    return dict(parent_id=parent['id'], feasible=row['feasible'], screening_passed=passed,
        G_optimal_claim=g, class_optimal_mean_claim=mean, information_relative=relative,
        witnesses=checks, optimum_independently_resolved=False,
        G_first_information_suffix_action=first_information,
        G_first_action_divergence_between_hypotheses=first_action_divergence,
        G_policy_can_adapt=True, saved_G_routes_actually_differ=first_action_divergence is not None,
        algebraic_controls_are_independent_measurements=False)


def revision_check(revision):
    folder = ROOT/f'audit_results/v33_direction_information_{revision}_20260917'
    manifest, cfg, prov = provenance(folder)
    need(cfg['hypothesis_prior'] == [.5, .5] and len(cfg['parents']) == 2, 'fixed priors and parent count')
    need(manifest['revision'] == revision, 'manifest revision')
    summaries, policy_checks, geometries = [], [], []
    for parent in cfg['parents']:
        path = folder/(parent['id']+'_geometry.json')
        if not path.exists(): continue
        geometry = read(path); geometries.append(geometry)
        summaries.append(pair_check(parent, geometry))
        policy_path = folder/(parent['id']+'_policies.json')
        if policy_path.exists(): policy_checks.append(policies_check(parent, geometry, read(policy_path)))
    complete = (folder/'result.json').exists()
    result = read(folder/('result.json' if complete else 'failure.json'))
    counts = result['counts']
    need(all(counts[k] == 0 for k in ('worlds', 'sensor_packets', 'mapper_updates',
                                    'TSDF_integrations', 'Q_evaluations', 'new_main_tasks')), 'zero physical counters')
    need(result['main_tasks_used'] == 16, 'unchanged physical quota')
    if complete:
        need(len(summaries) == 2 and counts['analytic_geometry_models'] == 4, 'complete four model tables')
        all_geometry = all(all(r['geometry_gates'].values()) for r in summaries)
        need(result['all_geometry_gates_passed'] == all_geometry, 'all geometry gate')
        need(result['status'] == ('complete' if all_geometry else 'geometry_gate_failed'), 'terminal outcome')
        if not all_geometry:
            need(not policy_checks and counts['exact_solver_instances'] == counts['memo_states'] == 0,
                 'geometry failure prevents both parents policy optimization')
        else:
            need(len(policy_checks) == 2 and counts['exact_solver_instances'] == 4, 'two main plus two control solvers')
        need(result['all_finite_information_gates_passed'] == bool(all_geometry and len(policy_checks) == 2 and
             all(p['screening_passed'] for p in policy_checks)), 'two-parent information gate')
        need(result['parents'] == [dict(parent_id=r['parent_id'], geometry_gates=r['geometry_gates'])
                                  for r in summaries], 'geometry root summary')
        policies = [read(folder/(p['id']+'_policies.json')) for p in cfg['parents']
                    if (folder/(p['id']+'_policies.json')).exists()]
        need(result['policy_summaries'] == [{k: v for k, v in r.items() if k != 'witnesses'} for r in policies],
             'complete root policy summaries')
        need(counts['memo_states'] == sum(r['main_memo_states']+r['ordinary_memo_states'] for r in policies),
             'total reported solver memo states')
        need(counts['memo_states'] <= cfg['solver_limit']['maximum_memo_states'], 'completed search obeys total memo cap')
        need(result['source_count'] == prov['sources'], 'source count')
        need(result['precision_thresholds_cm'] == {'2': None, '5': None, '10': None}, 'unmeasured thresholds')
        need(not any(result[k] for k in ('actual_C80_verified', 'physical_experiment_ready',
             'full_architecture_advantage_proven', 'semantic_efficacy_proven')), 'no physical efficacy certification')
    else:
        need(result['status'] in ('censored', 'failed') and not result['exact_optimum_certified'],
             'failure or resource censoring does not certify optimum')
    return dict(revision=revision, provenance=prov, terminal_status=result['status'],
        result_sha256=sha(folder/('result.json' if complete else 'failure.json')),
        complete_tables=complete, table_checks=summaries, policies=policy_checks, reported_counts=counts,
        all_finite_information_gates_passed=result.get('all_finite_information_gates_passed', False),
        independent_scope='Saved table/mask/witness arithmetic and collision paths; no visibility or DP recomputation.')


def revision_difference(checks):
    old = read(ROOT/'configs/virtual3d/v33_direction_scene_20260917.json')
    new = read(ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json')
    expected = deepcopy(old)
    expected['version'] = 'v33-direction-service-scene-r1'
    expected['geometry_freeze_scope'] = ('one defect-grounded r1 after retained r0 observability failure; '
                                        'no further correction or reward/budget sweep')
    for parent in expected['parents']:
        turns = parent['device_frame']['quarter_turns_ccw']
        need(turns in (0, 1), 'declared two parent orientations')
        parent['geometry_facts']['ribs_local_y'][0][0] = 1.5
        for h in parent['hypotheses']:
            boxes = h['assets'][0]['boxes']
            need(boxes[0][5] == 2., 'original guard height')
            boxes[0][5] = 1.8
            if turns == 0:
                need(boxes[3][2] == 1.6, 'original first rib front')
                boxes[3][2] = 1.5
            else:
                need(boxes[3][1] == -1.6, 'original rotated first rib front')
                boxes[3][1] = -1.5
    need(new == expected, 'r1 contains only declared guard-height and first-rib closure plus metadata changes')
    need(sha(GENERATOR) == GENERATOR_SHA, 'approved standalone generator SHA')
    correction = ROOT/'docs/research/V33_R1_GEOMETRY_CORRECTION_20260917.md'
    text = correction.read_text()
    need(GENERATOR_SHA in text and sha(ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json') in text,
         'frozen r1 correction links exact generator and configuration')
    deltas = []
    for old_row, new_row in zip(checks[0]['table_checks'], checks[1]['table_checks']):
        need(old_row['parent_id'] == new_row['parent_id'], 'parent pairing across revisions')
        parent = old_row['parent_id']
        folders = [ROOT/f'audit_results/v33_direction_information_{r}_20260917' for r in ('r0', 'r1')]
        tables = [read(folder/(parent+'_geometry.json'))['tables'] for folder in folders]
        for h in (0, 1):
            a, b = tables[0][h], tables[1][h]
            for key in ('poses', 'edges', 'anchor', 'floor_cells', 'prefix_nodes'):
                need(a[key] == b[key], 'r1 leaves graph and paid prefix unchanged: '+key)
            area0 = sum(a['exact_reference_vertical_areas'].values())
            area1 = sum(b['exact_reference_vertical_areas'].values())
            deltas.append(dict(parent=parent, hypothesis=h, reference_area_r0=area0,
                               reference_area_r1=area1, reference_area_delta=area1-area0))
    return dict(generator_sha256=GENERATOR_SHA, correction_doc_sha256=sha(correction),
        approved_config_changes_only=True, graph_and_prefix_unchanged=True,
        observed_table_area_deltas=deltas,
        scope='Configuration-byte differences and saved table sums, not new union-surface computation or reward optimization.')


def write(path, payload):
    if not isinstance(payload, bytes): payload = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    need(not path.exists(), 'no overwrite '+str(path))
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    need(used+len(payload) <= CAP and shutil.disk_usage(OUT).free-len(payload) >= RESERVE, 'review capacity/reserve')
    with path.open('xb') as stream: stream.write(payload)


def main():
    need(not OUT.exists(), 'review output already exists')
    need(DOC.exists(), 'review report must be ready before terminal seal')
    for revision in ('r0', 'r1'):
        folder = ROOT/f'audit_results/v33_direction_information_{revision}_20260917'
        need((folder/'artifact_hashes.json').exists() and
             ((folder/'result.json').exists() or (folder/'failure.json').exists()), 'both revisions must be terminal and sealed')
    need(shutil.disk_usage(ROOT).free >= RESERVE+CAP, 'review requires2MiB plus64MiB reserve')
    OUT.mkdir()
    try:
        for prior in (FAILED_REVIEW, CHECKER_FAILURE):
            inv = read(prior/'artifact_hashes.json')
            need({str(p.relative_to(prior)) for p in prior.rglob('*') if p.is_file()
                  and p != prior/'artifact_hashes.json'} == set(inv), 'preserved failed checker exact file set')
            for name, expected in inv.items():
                need(sha(prior/name) == expected, 'preserved failed checker SHA')
        checks = [revision_check(revision) for revision in ('r0', 'r1')]
        difference = revision_difference(checks)
        write(OUT/'result.json', dict(status='saved_evidence_verified', revisions=checks,
            single_revision_audit=difference,
            checker_correction=dict(failed_review_root=str(FAILED_REVIEW.relative_to(ROOT)),
                failed_review_inventory_sha256=sha(FAILED_REVIEW/'artifact_hashes.json'),
                failed_review_failure_sha256=sha(FAILED_REVIEW/'failure.json'),
                failed_source_archive=str(CHECKER_FAILURE.relative_to(ROOT)),
                failed_source_archive_inventory_sha256=sha(CHECKER_FAILURE/'artifact_hashes.json'),
                receipt=read(CHECKER_FAILURE/'receipt.json'),
                primary_results_modified=False, geometry_or_DP_repeated=False),
            review_script_sha256=sha(Path(__file__)), no_visibility_recomputation=True,
            no_DP_recomputation=True, no_new_physical_or_Q_evaluations=True,
            mandatory_prefix_actions=18, optimized_suffix_budget=24, total_declared_budget=42,
            precision_thresholds_cm={'2': None, '5': None, '10': None},
            limits=['Only stored geometric-signature equality is checked; raw signature payloads are not regenerated.',
                    'Exactness of visibility quadrature and dynamic-program optimum are not independently recomputed.',
                    'An optimum after the compulsory18-action prefix is not an unconstrained42-action optimum.',
                    'C_grid and visible exterior area are ideal proxies, not physical C80 or3D accuracy.',
                    'Algebraic controls are not independent measurements.']))
        write(OUT/Path(__file__).name, Path(__file__).read_bytes())
        write(OUT/GENERATOR.name, GENERATOR.read_bytes())
        write(OUT/'REVIEW.md', DOC.read_bytes())
    except BaseException:
        write(OUT/'failure.json', dict(status='review_failed', error=traceback.format_exc(),
            no_new_physical_or_Q_evaluations=True))
        raise
    finally:
        write(OUT/'artifact_hashes.json', {str(p.relative_to(OUT)): sha(p)
            for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps(dict(status='saved_evidence_verified', output=str(OUT),
                         result_sha256=sha(OUT/'result.json')), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='After both revisions seal; no physics or solver calls.')
    args = parser.parse_args()
    if args.run: main()
    else: parser.print_help()
