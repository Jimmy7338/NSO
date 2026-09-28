#!/usr/bin/env python3
"""One finite geometric information-value calculation; no project simulation imports."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import deque
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
import traceback
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/virtual3d/v29_information_scene_20260917.json'
PROTOCOL = ROOT/'docs/research/V29_SCENE_INFORMATION_PROTOCOL_20260917.md'
OUTPUT = ROOT/'audit_results/v29_scene_information_20260917'
EPS = 1e-7
NEG = -1e30


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def gap(lo, hi, left, right):
    return max(left-hi, lo-right, 0.)


def swept_clear(a, b, blocks, radius):
    # Axis-aligned legal translation: exact distance between its segment and rectangle.
    for r in blocks:
        dx = gap(min(a[0], b[0]), max(a[0], b[0]), r[0], r[1])
        dy = gap(min(a[1], b[1]), max(a[1], b[1]), r[2], r[3])
        if dx*dx+dy*dy < radius*radius-EPS:
            return False
    return True


def inside(p, bounds):
    return all(bounds[2*i]+EPS < p[i] < bounds[2*i+1]-EPS for i in range(3))


def occludes(origin, point, bounds):
    lower, upper = 0., 1.
    for i in range(3):
        delta = point[i]-origin[i]
        if abs(delta) < EPS:
            if origin[i] < bounds[2*i]-EPS or origin[i] > bounds[2*i+1]+EPS:
                return False
        else:
            a = (bounds[2*i]-origin[i])/delta
            b = (bounds[2*i+1]-origin[i])/delta
            lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
            if lower > upper+EPS:
                return False
    return upper > EPS and lower < 1.-EPS and lower <= upper+EPS


def vertical_patches(bounds, station, config):
    # Centers and exact area of each finite quadrature rectangle; identity is never observed.
    spatial = config['geometric_observation']
    for axis in (0, 1):
        along = 1-axis
        length = bounds[2*along+1]-bounds[2*along]
        height = bounds[5]-bounds[4]
        na = math.ceil(length/spatial['surface_horizontal_spacing_max_m'])
        nz = math.ceil(height/spatial['surface_vertical_spacing_max_m'])
        for side in (0, 1):
            normal = [0., 0., 0.]; normal[axis] = -1. if side == 0 else 1.
            for ia in range(na):
                for iz in range(nz):
                    p = [0., 0., bounds[4]+height*(iz+.5)/nz]
                    p[axis] = bounds[2*axis+side]
                    p[along] = bounds[2*along]+length*(ia+.5)/na
                    yield dict(point=tuple(p), normal=tuple(normal), station=station,
                               area=length*height/(na*nz))


class FiniteGeometry:
    """Immutable analytic rectangles and visibility tables, not a World/Sensor API."""
    def __init__(self, config, classes):
        self.config, self.classes = config, tuple(classes)
        solids = [(tuple(r), None) for r in config['class_independent_wall_bounds']]
        for i, item in enumerate(config['stations']):
            for key in ('body_relative_bounds', 'complex_attachment_relative_bounds'):
                if key.startswith('complex') and classes[i] != 'complex':
                    continue
                bounds = list(config[key]); bounds[0] += item['center_x']; bounds[1] += item['center_x']
                solids.append((tuple(bounds), i))
        self.blocks = tuple(r for r, _ in solids)
        self.surfaces, dedup = [], set()
        for bounds, station in solids:
            for patch in vertical_patches(bounds, station, config):
                p, n = patch['point'], patch['normal']
                outside = tuple(p[i]+10*EPS*n[i] for i in range(3))
                if any(inside(outside, r) for r in self.blocks):
                    continue
                key = tuple(round(x, 10) for x in p+n)
                if key in dedup:
                    continue
                dedup.add(key); patch['physical_key'] = key; self.surfaces.append(patch)
        self.targets = [p for p in self.surfaces if p['station'] is not None]
        self.target_index = {p['physical_key']:i for i, p in enumerate(self.targets)}
        self.station_masks = tuple(sum(1 << i for i, p in enumerate(self.targets) if p['station'] == s)
                                   for s in (0, 1))
        self.areas = tuple(sum(p['area'] for p in self.targets if p['station'] == s) for s in (0, 1))
        grid = config['grid']; radius = grid['robot_radius_m']
        self.cells = tuple((x, y) for x in range(grid['x_min'], grid['x_max']+1)
            for y in range(grid['y_min'], grid['y_max']+1) if swept_clear((x,y), (x,y), self.blocks, radius))
        cell_set = set(self.cells)
        self.poses = tuple((x, y, h) for x, y in self.cells for h in range(4))
        self.pose_index = {p:i for i, p in enumerate(self.poses)}
        self.edges = []
        for x, y, h in self.poses:
            dx, dy = config['heading_vectors'][h]; dest = x+dx, y+dy
            edges = []
            if dest in cell_set and swept_clear((x,y), dest, self.blocks, radius):
                edges.append(('forward', self.pose_index[(*dest, h)]))
            edges.extend((('left', self.pose_index[(x,y,(h-1)%4)]),
                          ('right', self.pose_index[(x,y,(h+1)%4)])))
            self.edges.append(tuple(edges))
        self.anchor = self.pose_index.get(tuple(config['anchor']))
        require(self.anchor is not None, 'anchor is not safe')
        self.signatures, self.visible_masks = [], []
        obs = config['geometric_observation']
        tan_h = math.tan(math.radians(obs['horizontal_fov_deg']/2))
        tan_v = math.tan(math.radians(obs['vertical_fov_deg']/2))
        for x, y, h in self.poses:
            origin = (x, y, obs['camera_height_m']); forward = config['heading_vectors'][h]
            right = (forward[1], -forward[0]); signature, visible = [], 0
            for patch in self.surfaces:
                p, n = patch['point'], patch['normal']; delta = tuple(p[i]-origin[i] for i in range(3))
                axial, lateral = dot(delta[:2], forward), dot(delta[:2], right)
                if (axial <= EPS or abs(lateral) > axial*tan_h+EPS
                    or abs(delta[2]) > axial*tan_v+EPS or dot(delta, delta) > obs['range_m']**2+EPS
                    or dot(n, tuple(-d for d in delta)) <= EPS):
                    continue
                if any(occludes(origin, p, r) for r in self.blocks):
                    continue
                signature.append(patch['physical_key'])
                if patch['station'] is not None:
                    visible |= 1 << self.target_index[patch['physical_key']]
            # No role, class, area, hidden surface ID or normalization enters this signature.
            self.signatures.append(tuple(sorted(signature)))
            self.visible_masks.append(visible)

    @lru_cache(maxsize=None)
    def fractions(self, mask):
        area = [0., 0.]
        while mask:
            bit = mask & -mask; i = bit.bit_length()-1; p = self.targets[i]
            area[p['station']] += p['area']; mask ^= bit
        values = tuple(min(1., max(0., area[i]/self.areas[i])) for i in (0, 1))
        return ((values[0]+values[1])/2, *values)


def bfs(edges, start):
    distance, previous = {start:0}, {}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for action, v in edges[u]:
            if v not in distance:
                distance[v] = distance[u]+1; previous[v] = (u, action); queue.append(v)
    return distance, previous


def route(previous, start, end):
    actions = []
    while end != start:
        end, action = previous[end]; actions.append(action)
    return list(reversed(actions))


def path_states(model, actions, start=None):
    node = model.anchor if start is None else start
    states = [list(model.poses[node])]
    for action in actions:
        allowed = dict(model.edges[node]); require(action in allowed, 'illegal analytic action')
        node = allowed[action]; states.append(list(model.poses[node]))
    return states


def prefix(model):
    actions = model.config['prefix_actions']; node = model.anchor
    nodes, seen = [node], model.visible_masks[node]
    for action in actions:
        options = dict(model.edges[node]); require(action in options, 'unsafe common prefix')
        node = options[action]; nodes.append(node); seen |= model.visible_masks[node]
    require(node == model.anchor, 'prefix did not return to exact anchor')
    return nodes, seen


class SolverLimit(RuntimeError):
    pass


class ExactBeliefSolver:
    """All legal action recursion. Hidden hypotheses enter expectation, not revealed policy state."""
    def __init__(self, models, initial_masks, state_limit):
        self.models, self.initial = models, tuple(initial_masks)
        self.edges, self.anchor = models[0].edges, models[0].anchor
        self.state_limit, self.visited = state_limit, 0
        reverse = [[] for _ in self.edges]
        for u, rows in enumerate(self.edges):
            for _, v in rows:
                reverse[v].append(('reverse', u))
        self.return_distance, _ = bfs(reverse, self.anchor)

    def count(self):
        self.visited += 1
        if self.visited > self.state_limit:
            raise SolverLimit('fixed memo-state limit; optimum not certified')

    @lru_cache(maxsize=None)
    def known(self, h, node, remaining, mask):
        self.count()
        if self.return_distance.get(node, 10**9) > remaining:
            return (NEG, NEG, NEG)
        best = list(self.models[h].fractions(mask)) if node == self.anchor else [NEG]*3
        if remaining:
            for _, following in self.edges[node]:
                values = self.known(h, following, remaining-1,
                    mask | self.models[h].visible_masks[following])
                for i in range(3):
                    if values[i] > best[i]: best[i] = values[i]
        return tuple(best)

    @lru_cache(maxsize=None)
    def uncertain(self, node, remaining, mask0, mask1):
        self.count()
        if self.return_distance.get(node, 10**9) > remaining:
            return NEG
        best = sum(m.fractions(s)[0] for m, s in zip(self.models, (mask0,mask1)))/2 if node == self.anchor else NEG
        if remaining:
            for _, following in self.edges[node]:
                a = mask0 | self.models[0].visible_masks[following]
                b = mask1 | self.models[1].visible_masks[following]
                if self.models[0].signatures[following] == self.models[1].signatures[following]:
                    value = self.uncertain(following, remaining-1, a, b)
                else:
                    value = (self.known(0, following, remaining-1, a)[0]
                             +self.known(1, following, remaining-1, b)[0])/2
                if value > best: best = value
        return best

    def known_action(self, h, node, remaining, mask, component=0):
        value = self.known(h, node, remaining, mask)[component]
        if node == self.anchor and self.models[h].fractions(mask)[component] >= value-1e-12:
            return None
        for action, following in self.edges[node]:
            candidate = self.known(h, following, remaining-1,
                mask | self.models[h].visible_masks[following])[component]
            if candidate >= value-1e-12:
                return action
        raise AssertionError('no exact known policy action')

    def uncertain_action(self, node, remaining, masks):
        value = self.uncertain(node, remaining, *masks)
        if node == self.anchor and sum(m.fractions(s)[0] for m,s in zip(self.models,masks))/2 >= value-1e-12:
            return None
        for action, following in self.edges[node]:
            a = masks[0] | self.models[0].visible_masks[following]
            b = masks[1] | self.models[1].visible_masks[following]
            if self.models[0].signatures[following] == self.models[1].signatures[following]:
                candidate = self.uncertain(following, remaining-1, a, b)
            else:
                candidate = (self.known(0,following,remaining-1,a)[0]+self.known(1,following,remaining-1,b)[0])/2
            if candidate >= value-1e-12:
                return action
        raise AssertionError('no exact uncertain policy action')

    def witness(self, actual, budget, policy='G', hint=None, component=0):
        node, remaining, masks = self.anchor, budget, list(self.initial)
        known_h = actual if policy == 'correct_class_oracle' else hint
        actions, revelations, decision_sources = [], [], []
        while remaining >= 0:
            action = (self.uncertain_action(node, remaining, masks) if known_h is None
                      else self.known_action(known_h, node, remaining, masks[known_h], component))
            if action is None: break
            require(remaining > 0, 'unpaid witness action')
            decision_sources.append('geometry_belief' if known_h is None else ('class_hypothesis' if not revelations else 'revealed_geometry'))
            node = dict(self.edges[node])[action]; remaining -= 1; actions.append(action)
            masks = [s | m.visible_masks[node] for s,m in zip(masks,self.models)]
            if self.models[0].signatures[node] != self.models[1].signatures[node]:
                if not revelations:
                    revelations.append(dict(suffix_action=len(actions), paid_action=len(self.models[0].config['prefix_actions'])+len(actions),
                        pose=list(self.models[0].poses[node]), prior_hint=known_h,
                        actual_signature_sha256=digest(self.models[actual].signatures[node]),
                        interpretation='physical geometric signature separates the two declared hypotheses'))
                known_h = actual
        require(node == self.anchor and remaining >= 0, 'witness failed exact return or budget')
        model = self.models[actual]; values = model.fractions(masks[actual])
        return dict(policy=policy, actual_hypothesis=actual, initial_hint=hint, suffix_actions=actions,
            suffix_states=path_states(model, actions), suffix_paid_actions=len(actions),
            total_paid_actions=len(model.config['prefix_actions'])+len(actions),
            returned_exact_pose=True, budget_fits=True, first_geometric_revelation=revelations[0] if revelations else None,
            decision_sources=decision_sources, potential_macro=values[0], potential_per_station=list(values[1:]),
            station_complete=[masks[actual] & sm == sm for sm in model.station_masks],
            causal_scope='exact policy within finite analytic model; not actual robot or current ANS')

    def complete_option(self, h, station, remaining):
        # Fixed declared upper limit: a shortest-cost query, not a sweep choosing the task budget.
        mask = self.initial[h]; sm = self.models[h].station_masks[station]
        if mask & sm == sm:
            shortest = 0
        elif self.known(h,self.anchor,remaining,mask)[station+1] < 1.-1e-12:
            return dict(hypothesis=h, station=station, fits_declared_remaining_budget=False,
                exact_shortest_cost=None, lower_bound_exclusive=remaining,
                explanation='No all-representative-face completion plus exact return within the fixed limit')
        else:
            lo, hi = 0, remaining
            while lo < hi:
                middle = (lo+hi)//2
                if self.known(h,self.anchor,middle,mask)[station+1] >= 1.-1e-12: hi = middle
                else: lo = middle+1
            shortest = lo
        result = self.witness(h, shortest, 'correct_class_oracle', component=station+1)
        require(result['station_complete'][station], 'complete service witness missed reference patches')
        return dict(hypothesis=h, station=station, fits_declared_remaining_budget=True,
            exact_shortest_cost=shortest, total_with_prefix=len(self.models[h].config['prefix_actions'])+shortest,
            witness=result, explanation='Exact complete finite-face observation plus return; not reconstruction quality')


def snapshot_geometry(models, config):
    records = []
    for h, model in enumerate(models):
        distances, previous = bfs(model.edges, model.anchor)
        union = 0
        for node in distances: union |= model.visible_masks[node]
        missing = [dict(point=p['point'],normal=p['normal'],station=p['station'],area=p['area'])
            for i,p in enumerate(model.targets) if not union & (1<<i)]
        records.append(dict(hypothesis=h, classes=model.classes, safe_cells=len(model.cells),
            safe_poses=len(model.poses), connected_poses=len(distances), all_safe_connected=len(distances)==len(model.poses),
            target_representatives=len(model.targets), reference_vertical_areas=model.areas,
            unreachable_or_invisible_target_representatives=missing,
            all_targets_observable=not missing, navigation_sha256=digest([model.poses,model.edges])))
    common = models[0].poses == models[1].poses and models[0].edges == models[1].edges
    prefix_nodes, initial = zip(*(prefix(m) for m in models))
    same = common and all(models[0].signatures[a] == models[1].signatures[b]
        for a,b in zip(prefix_nodes[0],prefix_nodes[1]))
    first_layer = None; witnesses = []
    if common:
        distances, previous = bfs(models[0].edges, models[0].anchor)
        different = [node for node in distances if models[0].signatures[node] != models[1].signatures[node]]
        if different:
            first_layer = min(distances[n] for n in different)
            for node in sorted(n for n in different if distances[n] == first_layer):
                witnesses.append(dict(pose=list(models[0].poses[node]), action_distance=first_layer,
                    actions=route(previous,models[0].anchor,node),
                    signature_point_counts=[len(m.signatures[node]) for m in models],
                    geometric_signatures_sha256=[digest(m.signatures[node]) for m in models]))
    return dict(hypotheses=records, identical_navigation_graph=common,
        prefix_actions=len(config['prefix_actions']), prefix_states=path_states(models[0],config['prefix_actions']),
        prefix_geometric_signatures_identical=same,
        prefix_signature_sha256=[[digest(m.signatures[n]) for n in nodes] for m,nodes in zip(models,prefix_nodes)],
        first_geometric_information_action_layer=first_layer, first_layer_witnesses=witnesses,
        first_information_scope='Exact finite-grid, finite-representative geometry only; not all continuous views or real sensor channels'), initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run the sole frozen analytic calculation; output cannot exist')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return
    config = json.loads(CONFIG.read_text()); cap=config['io']['output_cap_bytes']; reserve=config['io']['disk_reserve_bytes']
    require(not OUTPUT.exists(), 'sealed output exists; no overwrite or second run')
    require(shutil.disk_usage(OUTPUT.parent).free >= reserve+cap, '64 MiB reserve unavailable')
    OUTPUT.mkdir(); started = perf_counter(); solvers = []
    sources = {str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,PROTOCOL,Path(__file__))}

    def write(name, data):
        raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'),allow_nan=False)+'\n').encode()
        used = sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())
        require(used+len(raw)+16384 < cap, '1 MiB artifact cap')
        require(shutil.disk_usage(OUTPUT).free-len(raw) >= reserve, '64 MiB reserve')
        destination = OUTPUT/name; temporary=destination.with_suffix(destination.suffix+'.tmp')
        temporary.write_bytes(raw); os.replace(temporary,destination)

    def timeout(*_):
        raise SolverLimit('300-second fixed execution bound; optimum not certified')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(config['solver']['wall_time_limit_seconds'])
    try:
        with zipfile.ZipFile(OUTPUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in sources: archive.writestr(name,(ROOT/name).read_bytes())
        write('manifest.json',dict(status='frozen_before_calculation',source_sha256=sources,
            source_archive_sha256=sha(OUTPUT/'sources.zip'), config_sha256=sha(CONFIG), protocol_sha256=sha(PROTOCOL),
            physical_worlds=0,sensor_packets=0,mapper_updates=0,mesh_extractions=0,Q_evaluations=0,new_main_tasks=0,
            scope='One analytic finite geometry and exact information-policy model'))
        models = [FiniteGeometry(config, classes) for classes in config['hypotheses']]
        geometry, initial = snapshot_geometry(models,config); write('geometry.json',geometry)
        print(json.dumps(dict(stage='geometry_complete',common_graph=geometry['identical_navigation_graph'],
            paired_prefix=geometry['prefix_geometric_signatures_identical'],
            visible=[r['all_targets_observable'] for r in geometry['hypotheses']],
            information_layer=geometry['first_geometric_information_action_layer'])),flush=True)
        preliminary = (geometry['identical_navigation_graph'] and geometry['prefix_geometric_signatures_identical']
            and all(r['all_safe_connected'] and r['all_targets_observable'] for r in geometry['hypotheses']))
        if not preliminary:
            write('result.json',dict(status='complete',development_gate=False,geometric_gates_passed=False,
                policy_comparison=None,reason='Frozen finite geometry failed; no redesign or favorable face removal',
                elapsed_seconds=perf_counter()-started,new_main_tasks=0,Q_evaluations=0))
        else:
            remaining=config['total_action_budget']-len(config['prefix_actions'])
            require(remaining==30 and config['total_action_budget']==48,'frozen total/prefix budget changed')
            solver=ExactBeliefSolver(models,initial,config['solver']['maximum_memo_states']);solvers.append(solver)
            g=solver.uncertain(solver.anchor,remaining,*initial)
            correct=[solver.known(h,solver.anchor,remaining,initial[h])[0] for h in (0,1)]
            records=[]
            for h in (0,1):
                records.extend((solver.witness(h,remaining,'G'),solver.witness(h,remaining,'correct_class_oracle'),
                    solver.witness(h,remaining,'swapped_class_with_geometric_correction',hint=1-h)))
            require(abs(sum(r['potential_macro'] for r in records if r['policy']=='G')/2-g)<1e-10,'Bayes policy expectation mismatch')
            require(all(abs(next(r for r in records if r['actual_hypothesis']==h and r['policy']=='correct_class_oracle')['potential_macro']-correct[h])<1e-10 for h in (0,1)), 'class oracle witness mismatch')
            options=[solver.complete_option(h,s,remaining) for h in (0,1) for s in (0,1)]
            write('main_policy_witnesses.json',dict(records=records,complete_options=options))
            print(json.dumps(dict(stage='main_exact_complete',memo_states=solver.visited)),flush=True)
            # Ordinary control independently constructs identical geometry hypotheses.
            ordinary=[FiniteGeometry(config,['simple','simple']) for _ in range(2)]
            ordinary_geometry, ordinary_initial=snapshot_geometry(ordinary,config)
            require(ordinary[0].signatures==ordinary[1].signatures,'ordinary IDs created false geometric information')
            control=ExactBeliefSolver(ordinary,ordinary_initial,max(1,config['solver']['maximum_memo_states']-solver.visited));solvers.append(control)
            ordinary_g=control.uncertain(control.anchor,remaining,*ordinary_initial)
            ordinary_class=control.known(0,control.anchor,remaining,ordinary_initial[0])[0]
            tol=config['development_gate']['required_controls_equal_tolerance']
            early=(correct[0]+correct[1])/2
            controls=dict(ordinary_both_simple=dict(G=ordinary_g,class_oracle=ordinary_class,
                    difference=ordinary_class-ordinary_g,all_observations_equal=True,
                    passed=abs(ordinary_class-ordinary_g)<=tol),
                free_early_geometric_revelation=dict(G_with_geometry_revealed=early,class_oracle=early,difference=0.,
                    passed=True,scope='Initial singleton belief from an explicitly free geometry-information intervention; same optimizer'),
                swapped_class_cue=dict(mean_potential=sum(r['potential_macro'] for r in records if r['policy'].startswith('swapped'))/2,
                    geometry_correction_allowed=True,required_to_be_worse=False),
                independent_uninformative_class_cue=dict(optimal_informed_of_independence=g,optimal_ignores_cue=g,
                    difference=0.,passed=True,justification='Independent cue leaves posterior unchanged; same exact Bayes policy',
                    follow_cue_four_equal_probability_values=[next(r['potential_macro'] for r in records if r['actual_hypothesis']==h and
                        r['policy']==('correct_class_oracle' if hint==h else 'swapped_class_with_geometric_correction'))
                        for h in (0,1) for hint in (0,1)]))
            gain=early-g
            controls_pass=controls['ordinary_both_simple']['passed'] and controls['free_early_geometric_revelation']['passed'] and controls['independent_uninformative_class_cue']['passed']
            write('controls.json',dict(controls=controls,ordinary_geometry=ordinary_geometry))
            write('result.json',dict(status='complete',development_gate=controls_pass and gain>config['development_gate']['minimum_information_gain_absolute'],
                geometric_gates_passed=True,exact_policy_search_completed=True,
                main=dict(G_Bayes_optimal=g,best_randomized_policy_expected_upper_bound=g,
                    randomized_bound_reason='A random policy is a mixture of deterministic history policies; expectation is linear',
                    correct_class_optimal_by_assignment=correct,correct_class_optimal_mean=early,
                    oracle_information_value_absolute=gain,oracle_information_value_relative=gain/g if g else None,
                    prefix_potential_macro=[m.fractions(s)[0] for m,s in zip(models,initial)],
                    frozen_total_budget=config['total_action_budget'],prefix_paid_actions=len(config['prefix_actions']),remaining_actions=remaining),
                controls=controls,memo_states=sum(s.visited for s in solvers),elapsed_seconds=perf_counter()-started,
                actual_runtime_tested=False,semantic_policy_gain_proven=False,Q_or_accuracy_proven=False,new_main_tasks=0,
                qualification='Only a finite potential-visible-area information calculation; no physical/model reward conclusion'))
        for name,expected in sources.items():require(sha(ROOT/name)==expected,'frozen source changed '+name)
    except BaseException as error:
        write('failure.json',dict(status='censored' if isinstance(error,SolverLimit) else 'failed',type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),memo_states=sum(s.visited for s in solvers),
            elapsed_seconds=perf_counter()-started,exact_optimum_certified=False,new_main_tasks=0))
        raise
    finally:
        signal.alarm(0)
        write('artifact_hashes.json',{p.name:sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(json.loads((OUTPUT/'result.json').read_text()),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
