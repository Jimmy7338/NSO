#!/usr/bin/env python3
"""Independent saved-witness check, standard library only; never solve the DP."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'audit_results/v29_scene_information_20260917'
OUT = ROOT / 'audit_results/v29_scene_information_review_20260917'
EPS = 1e-7


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(name):
    return json.loads((SOURCE / name).read_text())


def close(a, b, why):
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), why)


def solids(config, classes):
    rows = [(tuple(b), None) for b in config['class_independent_wall_bounds']]
    for owner, station in enumerate(config['stations']):
        keys = ['body_relative_bounds']
        if classes[owner] == 'complex':
            keys.append('complex_attachment_relative_bounds')
        for key in keys:
            bounds = list(config[key])
            bounds[0] += station['center_x']; bounds[1] += station['center_x']
            rows.append((tuple(bounds), owner))
    return rows


def safe_segment(start, end, blocks, radius):
    # Every checked translation is cardinal; distance to its bounding rectangle
    # equals distance to the line segment. Rotations use a stationary disk.
    for bounds, _ in blocks:
        distance_squared = 0.
        for axis in (0, 1):
            low, high = sorted((start[axis], end[axis]))
            wall_low, wall_high = bounds[2*axis:2*axis+2]
            distance = wall_low-high if high < wall_low else low-wall_high if wall_high < low else 0.
            distance_squared += distance*distance
        if distance_squared < radius*radius-EPS:
            return False
    return True


def validate_path(config, blocks, actions, states, must_return):
    anchor = config['anchor']; require(states[0] == anchor, 'wrong initial anchor')
    require(len(states) == len(actions)+1, 'path/action length mismatch')
    grid = config['grid']; current = list(anchor)
    for i, action in enumerate(actions):
        previous = list(current)
        require(action in config['actions'], 'unknown paid action')
        if action == 'forward':
            delta = config['heading_vectors'][current[2]]
            current[0] += delta[0]; current[1] += delta[1]
        else:
            current[2] = (current[2] + (1 if action == 'right' else -1)) % 4
        require(current == states[i+1], 'wrong saved transition at '+str(i+1))
        require(grid['x_min'] <= current[0] <= grid['x_max']
                and grid['y_min'] <= current[1] <= grid['y_max'], 'outside declared action grid')
        require(safe_segment(previous, current, blocks, grid['robot_radius_m']), 'inflated collision')
    if must_return:
        require(current == anchor, 'missing exact return heading')


def physical_surface_samples(config, blocks):
    # Reconstruct the declared quadrature only, including its center-based
    # contact approximation. This is not continuous-area certification.
    obs = config['geometric_observation']; rows = []; keys = set()
    for bounds, owner in blocks:
        for normal_axis in (0, 1):
            tangent_axis = 1-normal_axis
            length = bounds[2*tangent_axis+1]-bounds[2*tangent_axis]
            height = bounds[5]-bounds[4]
            nx = math.ceil(length/obs['surface_horizontal_spacing_max_m'])
            nz = math.ceil(height/obs['surface_vertical_spacing_max_m'])
            for side in (0, 1):
                normal = [0., 0., 0.]; normal[normal_axis] = 2.*side-1.
                for column in range(nx):
                    for row in range(nz):
                        p = [0., 0., bounds[4]+height*(row+.5)/nz]
                        p[normal_axis] = bounds[2*normal_axis+side]
                        p[tangent_axis] = bounds[2*tangent_axis]+length*(column+.5)/nx
                        outside = [p[k]+10*EPS*normal[k] for k in range(3)]
                        internal = any(all(other[2*k]+EPS < outside[k] < other[2*k+1]-EPS
                                           for k in range(3)) for other, _ in blocks)
                        key = tuple(round(x, 10) for x in p+normal)
                        if internal or key in keys:
                            continue
                        keys.add(key)
                        rows.append(dict(point=tuple(p), normal=tuple(normal), key=key,
                                         owner=owner, area=length*height/(nx*nz)))
    return rows


def blocked_line(origin, endpoint, box):
    entry, exit_at = 0., 1.
    for axis in range(3):
        direction = endpoint[axis]-origin[axis]
        low, high = box[2*axis:2*axis+2]
        if abs(direction) < EPS:
            if not low-EPS <= origin[axis] <= high+EPS:
                return False
        else:
            near, far = sorted(((low-origin[axis])/direction, (high-origin[axis])/direction))
            entry = max(entry, near); exit_at = min(exit_at, far)
            if entry > exit_at+EPS:
                return False
    return exit_at > EPS and entry < 1.-EPS and entry <= exit_at+EPS


def model(config, classes):
    blocks = solids(config, classes); surfaces = physical_surface_samples(config, blocks)
    obs = config['geometric_observation']

    @lru_cache(None)
    def observed(pose):
        origin = (*pose[:2], obs['camera_height_m'])
        forward = config['heading_vectors'][pose[2]]; right = (forward[1], -forward[0])
        result = []
        for patch in surfaces:
            d = tuple(patch['point'][k]-origin[k] for k in range(3))
            axial = sum(d[k]*forward[k] for k in (0, 1))
            lateral = sum(d[k]*right[k] for k in (0, 1))
            if (axial <= EPS or abs(lateral) > axial*math.tan(math.radians(obs['horizontal_fov_deg']/2))+EPS
                    or abs(d[2]) > axial*math.tan(math.radians(obs['vertical_fov_deg']/2))+EPS
                    or sum(x*x for x in d) > obs['range_m']**2+EPS
                    or sum(-d[k]*patch['normal'][k] for k in range(3)) <= EPS):
                continue
            if not any(blocked_line(origin, patch['point'], box) for box, _ in blocks):
                result.append(patch['key'])
        return tuple(sorted(result))

    def fractions(states):
        seen = set().union(*(set(observed(tuple(p))) for p in states))
        totals = [sum(p['area'] for p in surfaces if p['owner'] == s) for s in (0, 1)]
        areas = [sum(p['area'] for p in surfaces if p['owner'] == s and p['key'] in seen) for s in (0, 1)]
        fractions = [min(1., areas[s]/totals[s]) for s in (0, 1)]
        completed = [all(p['key'] in seen for p in surfaces if p['owner'] == s) for s in (0, 1)]
        return totals, fractions, completed

    return blocks, observed, fractions


def verify():
    inventory = read('artifact_hashes.json'); manifest = read('manifest.json')
    require(set(p.name for p in SOURCE.iterdir() if p.is_file()) == set(inventory)|{'artifact_hashes.json'},
            'source artifact set mismatch')
    for name, expected in inventory.items():
        require(sha(SOURCE/name) == expected, 'artifact hash mismatch: '+name)
    require(len(inventory) == 6, 'expected six archived artifacts plus inventory')
    require(sha(SOURCE/'sources.zip') == manifest['source_archive_sha256'], 'archive SHA')
    with zipfile.ZipFile(SOURCE/'sources.zip') as archive:
        require(set(archive.namelist()) == set(manifest['source_sha256']), 'archive source set')
        for name, expected in manifest['source_sha256'].items():
            require(sha(ROOT/name) == expected, 'current frozen source changed: '+name)
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'archived source changed')
    config = json.loads((ROOT/'configs/virtual3d/v29_information_scene_20260917.json').read_text())
    require(config['public_hypothesis_prior'] == [.5,.5] and config['grid']['spacing_m'] == 1., 'fixed model assumptions')
    geometry = read('geometry.json'); witnesses = read('main_policy_witnesses.json'); result = read('result.json')
    control = read('controls.json'); models = [model(config, c) for c in config['hypotheses']]
    prefix = geometry['prefix_states']; prefix_actions = config['prefix_actions']
    require(len(prefix_actions) == 18 and config['total_action_budget'] == 48, 'fixed paid budget')
    for h, (blocks, observed, fractions) in enumerate(models):
        validate_path(config, blocks, prefix_actions, prefix, True)
        signatures = [digest(observed(tuple(p))) for p in prefix]
        require(signatures == geometry['prefix_signature_sha256'][h], 'prefix observed signature hashes')
        totals, values, _ = fractions(prefix)
        for value, saved in zip(totals, geometry['hypotheses'][h]['reference_vertical_areas']):
            close(value, saved, 'reference finite area')
        close(sum(values)/2, result['main']['prefix_potential_macro'][h], 'prefix finite-area fraction')
    require(geometry['prefix_signature_sha256'][0] == geometry['prefix_signature_sha256'][1], 'prefix separation')
    records = witnesses['records']; require(len(records) == 6, 'six policies required')
    require({(r['actual_hypothesis'],r['policy']) for r in records} == {
        (h,p) for h in (0,1) for p in ('G','correct_class_oracle','swapped_class_with_geometric_correction')}, 'policy grid')
    summaries = []
    for row in records+[r['witness'] for r in witnesses['complete_options']]:
        h = row['actual_hypothesis']; blocks, observed, fractions = models[h]
        actions, states = row['suffix_actions'], row['suffix_states']
        validate_path(config, blocks, actions, states, True)
        require(len(actions) == row['suffix_paid_actions'], 'paid suffix count')
        require(len(actions)+len(prefix_actions) == row['total_paid_actions'] <= 48, 'full paid budget')
        require(row['returned_exact_pose'] and row['budget_fits'], 'saved feasibility claims')
        _, values, completed = fractions(prefix+states[1:])
        for value, saved in zip(values, row['potential_per_station']): close(value, saved, 'path finite-area value')
        close(sum(values)/2, row['potential_macro'], 'macro from actual witnessed visible union')
        require(completed == row['station_complete'], 'all finite patch completion')
        differing = [i for i,p in enumerate(states) if models[0][1](tuple(p)) != models[1][1](tuple(p))]
        first = differing[0] if differing else None; saved = row['first_geometric_revelation']
        if first is None:
            require(saved is None, 'spurious revelation')
        else:
            require(first == saved['suffix_action'] and saved['paid_action'] == 18+first, 'first difference step')
            require(saved['pose'] == states[first], 'first difference pose')
            require(saved['actual_signature_sha256'] == digest(observed(tuple(states[first]))), 'revelation signature hash')
        for i, source in enumerate(row['decision_sources']):
            expected = ('geometry_belief' if row['policy'] == 'G' else 'class_hypothesis') if first is None or i < first else 'revealed_geometry'
            require(source == expected, 'decision source/revelation causality')
        require(len(row['decision_sources']) == len(actions), 'decision source length')
        summaries.append(dict(hypothesis=h,policy=row['policy'],paid_total=row['total_paid_actions'],
            potential_macro=sum(values)/2,first_revelation_suffix_action=first,returned=True,
            action_sha256=digest(actions),finite_signature_verified=True))
    gs = sorted((r for r in records if r['policy']=='G'),key=lambda r:r['actual_hypothesis'])
    first = min(r['first_geometric_revelation']['suffix_action'] for r in gs)
    require(gs[0]['suffix_actions'][:first] == gs[1]['suffix_actions'][:first], 'G action leakage before actual information')
    require(gs[0]['suffix_states'][:first+1] == gs[1]['suffix_states'][:first+1], 'G pose leakage before actual information')
    for row in geometry['first_layer_witnesses']:
        states = [config['anchor']]; current = list(config['anchor'])
        for action in row['actions']:
            current = list(current)
            if action == 'forward':
                d = config['heading_vectors'][current[2]]; current[0]+=d[0]; current[1]+=d[1]
            else: current[2]=(current[2]+(1 if action=='right' else -1))%4
            states.append(current)
        require(states[-1] == row['pose'] and len(row['actions']) == row['action_distance'], 'information witness')
        for h, (blocks, observed, _) in enumerate(models):
            validate_path(config, blocks, row['actions'], states, False)
            signature = observed(tuple(states[-1]))
            require(digest(signature) == row['geometric_signatures_sha256'][h], 'information witness signature')
            require(len(signature) == row['signature_point_counts'][h], 'information witness point count')
    means = {p:sum(r['potential_macro'] for r in records if r['policy']==p)/2
             for p in ('G','correct_class_oracle','swapped_class_with_geometric_correction')}
    main = result['main']; close(means['G'],main['G_Bayes_optimal'],'G witnessed mean')
    close(means['correct_class_oracle'],main['correct_class_optimal_mean'],'class witnessed mean')
    gain = means['correct_class_oracle']-means['G']
    close(gain,main['oracle_information_value_absolute'],'absolute information arithmetic')
    close(gain/means['G'],main['oracle_information_value_relative'],'relative information arithmetic')
    require(control['controls'] == result['controls'],'controls duplicated faithfully')
    controls = control['controls']
    close(means['swapped_class_with_geometric_correction'],controls['swapped_class_cue']['mean_potential'],'swapped mean')
    close(sum(controls['independent_uninformative_class_cue']['follow_cue_four_equal_probability_values'])/4,
          (means['correct_class_oracle']+means['swapped_class_with_geometric_correction'])/2,'random independent cue mean')
    return dict(status='saved_witnesses_and_arithmetic_verified_not_independent_optimum',
        source_artifacts_verified=len(inventory),source_sha256=manifest['source_sha256'],
        input_inventory_sha256=sha(SOURCE/'artifact_hashes.json'),input_result_sha256=sha(SOURCE/'result.json'),
        six_main_witnesses=summaries[:6],four_complete_option_witnesses=summaries[6:],
        prefix_actions=18,full_budget=48,mean_values=means,information_value_absolute=gain,
        information_value_relative=gain/means['G'],G_same_paid_action_prefix_through_revelation=first,
        shortest_information_layer_independently_certified=False,DP_reruns=0,
        independent_optimality_certification=False,project_worlds=0,mapper_updates=0,Q_evaluations=0,
        finite_complex_reference_area=17.28,continuous_box_union_vertical_area=17.12,
        controls_algebraic_not_independent_measurements=['free_early_geometric_revelation','independent_uninformative_class_cue'])


def main():
    require(not OUT.exists(), 'review output exists; no overwrite')
    require(shutil.disk_usage(SOURCE).free > 64*1024**2+100000, '64 MiB reserve')
    OUT.mkdir()
    try:
        value = verify()
        value['review_script_sha256'] = sha(__file__)
        (OUT/'result.json').write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps(value,sort_keys=True))
    except BaseException as error:
        (OUT/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,error=str(error)))+'\n')
        raise
    finally:
        items = {p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'}
        (OUT/'artifact_hashes.json').write_text(json.dumps(items,sort_keys=True,indent=2)+'\n')


if __name__ == '__main__':
    main()
