#!/usr/bin/env python3
"""Separate, bounded thesis expansion; never writes legacy experiments or sources.

Configuration and source hashes are frozen before preparation and acquisition.
All declared cases are retained, including infeasible/failed cases; no retry.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'audit_results/thesis_expansion_20260928'
CONFIG = ROOT / 'configs/virtual3d/thesis_expansion_20260928.json'
SCENE = ROOT / 'configs/virtual3d/v33_direction_scene_r1_20260917.json'
GIB = 1024 ** 3
MODES = {'G': 'G', 'S': 'S', 'X': 'swapped', 'Xnf': 'swapped_no_feedback'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def payload(value):
    return (json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def used_bytes():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0


def guard(size=0):
    if used_bytes() + size > GIB:
        raise RuntimeError('new experiment would exceed the separate 1 GiB output cap')
    if shutil.disk_usage(ROOT).free - size < GIB:
        raise RuntimeError('new experiment would violate the 1 GiB disk reserve')


def write_bytes(path, data):
    guard(len(data)); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def write(path, value):
    write_bytes(path, payload(value))


def arrays(path, **values):
    import numpy as np
    buffer = io.BytesIO(); np.savez_compressed(buffer, **values)
    write_bytes(path, buffer.getvalue())


def freeze():
    if OUT.exists():
        raise FileExistsError('new batch already exists; cannot replace freeze')
    config = read(CONFIG)
    frozen_legacy = read(ROOT / 'audit_results/semantic_development_acquisition_20260923/method_correction.json')['execution_source_sha256']
    for rel, expected in frozen_legacy.items():
        if sha(ROOT / rel) != expected:
            raise ValueError('frozen 9/23 execution source changed: ' + rel)
    from nso.research_evidence_v31 import closure
    sources = closure([Path(__file__), CONFIG, SCENE])
    # The public saved-model loader uses dynamic file names, hence explicit inputs.
    inputs = [ROOT / 'audit_results/v33_direction_information_r1_20260917' / f'{p}_geometry.json'
              for p in ('P00', 'P01')]
    inputs += [ROOT / 'audit_results/v34_pixel_information_20260918' / f'{p}_h{h}_pixels.npz'
               for p in ('P00', 'P01') for h in (0, 1)]
    OUT.mkdir()
    write(OUT / 'protocol.json', config)
    write(OUT / 'source_freeze.json', dict(
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        protected_20260923_sources=frozen_legacy, config_sha256=sha(CONFIG),
        frozen_before_public_template_preparation_and_all_new_episodes=True,
        legacy_queue_resumed=False, legacy_10GiB_gate_unchanged=True))
    print(json.dumps({'frozen_cases': len(config['cases']), 'source_count': len(sources),
                      'free_GiB': shutil.disk_usage(ROOT).free / GIB}), flush=True)


def verify_sources():
    record = read(OUT / 'source_freeze.json')
    for group in ('source_sha256', 'input_sha256', 'protected_20260923_sources'):
        for rel, expected in record[group].items():
            if sha(ROOT / rel) != expected:
                raise ValueError('frozen input/source changed: ' + rel)
    if sha(CONFIG) != record['config_sha256']:
        raise ValueError('configuration changed after freeze')
    return read(OUT / 'protocol.json')


def parents():
    originals = {p['id']: p for p in read(SCENE)['parents']}
    result = {p: deepcopy(v) for p, v in originals.items()}
    for item in read(CONFIG)['variants']:
        p = deepcopy(originals[item['base_parent']])
        for h in p['hypotheses']:
            boxes = h['assets'][0]['boxes']
            if item['id'] == 'P00_rib_shift':
                boxes[4][2] += .25; boxes[4][3] += .25
            elif item['id'] == 'P01_lower_ribs':
                for b in boxes[3:]: b[5] = 1.35
            else:
                raise ValueError('unrecognized frozen variant')
        p['expansion_id'] = item['id']
        p['variant_definition'] = item
        result[item['id']] = p
    return result


def world_for(parent, hypothesis, episode, budget, seed, noise='iid_025px'):
    """Constructor-only adapter; inherited sensor/render/step/evaluator unchanged."""
    import numpy as np
    from dataclasses import replace
    from env.information_pixel_v34 import InformationPixelWorldV34
    from nso.box_union_geometry_v30 import BoxV30
    world = InformationPixelWorldV34(parent['id'], hypothesis, episode, noise_model=noise)
    world.parent = deepcopy(parent)
    world.config = replace(world.config, max_steps=int(budget))
    world.noise_seed = int(seed)
    world._reference_boxes, world.solid_boxes = [], []
    for asset in parent['hypotheses'][hypothesis]['assets']:
        members = [BoxV30(tuple((np.asarray(b).reshape(3, 2) + world.shift[:, None]).ravel()), asset['id'])
                   for b in asset['boxes']]
        world._reference_boxes.extend(members); world.solid_boxes.extend(members)
    for b in parent['background_boxes']:
        world.solid_boxes.append(BoxV30(tuple((np.asarray(b).reshape(3, 2) + world.shift[:, None]).ravel()), None))
    ground = BoxV30((0., world.config.width_m, 0., world.config.height_m, -.1, 0.), None)
    world.boxes = tuple(world.solid_boxes) + (ground,)
    bounds = np.asarray([b.bounds for b in world.boxes]).reshape(-1, 3, 2)
    world.primitives = np.concatenate((bounds[:, :, 0], bounds[:, :, 1] - bounds[:, :, 0]), axis=1)
    return world


def prepare():
    config = verify_sources()
    import numpy as np
    from nso.direction_information_v33 import DirectionGeometryV33
    from nso.online_planner_v35 import load_public_models_v35
    from nso.observation_belief_v35 import PublicTemplatesV35
    from nso.pixel_information_v34 import SavedPotentialModelV34
    from env.information_pixel_v34 import swept_clear_v34
    prepared = {}
    for name, parent in parents().items():
        started = time.monotonic(); folder = OUT / 'public_templates' / name
        if folder.exists(): raise FileExistsError('preparation cannot overwrite a scene')
        folder.mkdir(parents=True)
        if name in ('P00', 'P01'):
            models, prefix = load_public_models_v35(ROOT, name)
            templates = PublicTemplatesV35.from_saved(ROOT / 'audit_results/v34_pixel_information_20260918', name, models[0].poses)
            tables = read(ROOT / 'audit_results/v33_direction_information_r1_20260917' / f'{name}_geometry.json')['tables']
            preparation_queries = 0
        else:
            geometries = [DirectionGeometryV33(parent, h) for h in (0, 1)]
            tables = [m.tables() for m in geometries]
            models = tuple(SavedPotentialModelV34(t, ['unused'] * len(t['poses'])) for t in tables)
            prefix = tuple(parent['prefix_actions'])
            depths, scans = [], []
            for h in (0, 1):
                world = world_for(parent, h, 'public-template-only', 42, 0, 'clean')
                packets = [world.packet_at(pose, noise_model='clean') for pose in models[0].poses]
                depths.append(np.stack([p.frame.depth_m for p in packets]))
                scans.append(np.stack([p.scan.ranges_m for p in packets]))
            templates = PublicTemplatesV35(models[0].poses, np.stack(depths), np.stack(scans),
                provenance={'both_candidates_public': True, 'scene': name})
            preparation_queries = 2 * len(models[0].poses)
        if (models[0].poses, models[0].edges, models[0].anchor) != (models[1].poses, models[1].edges, models[1].anchor):
            raise ValueError('paired templates lack common navigation graph')
        for h in parent['hypotheses']:
            boxes = [b for a in h['assets'] for b in a['boxes']] + parent['background_boxes']
            for n, links in enumerate(models[0].edges):
                for _, nxt in links:
                    if not swept_clear_v34(models[0].poses[n][:2], models[0].poses[nxt][:2], boxes):
                        raise ValueError('unsafe common graph')
        informative = [n for n in range(len(models[0].poses)) if templates.template_information(n)['informative']]
        bounds = np.asarray([b for h in parent['hypotheses'] for a in h['assets'] for b in a['boxes']]).reshape(-1, 3, 2)
        cells = np.asarray(parent['nav_cells']); low = cells.min(axis=0) - .5
        shift = np.array([-low[0], -low[1], 0.])
        public_bounds = [bounds[:, :, 0].min(axis=0) + shift, bounds[:, :, 1].max(axis=0) + shift]
        nodes = models[0].prefix_nodes
        common_prefix = bool(np.array_equal(templates.depth[0, nodes], templates.depth[1, nodes])
                             and np.array_equal(templates.ranges[0, nodes], templates.ranges[1, nodes]))
        write(folder / 'geometry.json', {'tables': tables, 'parent': parent})
        arrays(folder / 'sensor_templates.npz', depth=templates.depth, ranges=templates.ranges)
        write(folder / 'metadata.json', dict(prefix=prefix, informative_nodes=informative,
              public_bounds=public_bounds, shift=shift, graph_nodes=len(models[0].poses),
              paired_graph_safe=True, prefix_geometrically_identical=common_prefix,
              prefix_returned=nodes[0] == nodes[-1] == models[0].anchor,
              prefix_ideal_coverage=[m.terminal(m.initial_mask)['coverage'] for m in models],
              pure_template_queries=preparation_queries, actual_episodes=0,
              elapsed_seconds=time.monotonic()-started))
        prepared[name] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        print(json.dumps({'prepared': name, 'seconds': time.monotonic()-started,
                          'common_prefix': common_prefix}), flush=True)
    write(OUT / 'preparation_seal.json', prepared)


def load_scene(name):
    import numpy as np
    from nso.pixel_information_v34 import SavedPotentialModelV34
    from nso.observation_belief_v35 import PublicTemplatesV35
    folder = OUT / 'public_templates' / name
    for rel, expected in read(OUT / 'preparation_seal.json')[name].items():
        if sha(folder / rel) != expected: raise ValueError('public preparation changed')
    raw = read(folder / 'geometry.json')
    models = tuple(SavedPotentialModelV34(t, ['unused'] * len(t['poses'])) for t in raw['tables'])
    with np.load(folder / 'sensor_templates.npz', allow_pickle=False) as data:
        templates = PublicTemplatesV35(models[0].poses, data['depth'], data['ranges'],
                                       provenance={'preparation_sha256': sha(OUT / 'preparation_seal.json')})
    return raw['parent'], models, templates, read(folder / 'metadata.json')


def run_case(index):
    config = verify_sources(); case = config['cases'][index]
    folder = OUT / 'cases' / case['id']
    if folder.exists(): raise FileExistsError('case already started; retry forbidden')
    guard(8 * 1024 ** 2); folder.mkdir(parents=True)
    started = time.monotonic()
    write(folder / 'started.json', dict(case=case, source_freeze_sha256=sha(OUT / 'source_freeze.json'),
          preparation_sha256=sha(OUT / 'preparation_seal.json'), time_unix=time.time(), pid=os.getpid()))
    def timeout(*_): raise TimeoutError('120 second complete-case cap')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(120)
    trace, history, actions, measurements = [], [], [], {}
    world = controller = mapper = None
    try:
        import numpy as np
        from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35, ObservationV35
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.surface_measurement_v34 import extract_observed_asset_mesh, SurfaceMeasurementV34
        from nso.decision_replay_v13 import save_packet, array_hash
        from nso.sensor_contract_v34 import validate_sensor_packet_v34
        from env.information_pixel_v34 import nonsemantic_rgb
        parent, models, templates, info = load_scene(case['scene'])
        controller = CPUFourModuleControllerV35(models, templates, info['prefix'],
            mode=MODES[case['mode']], total_budget=case['budget'], informative_nodes=info['informative_nodes'])
        world = world_for(parent, case['hypothesis'], case['id'], case['budget'], case['noise_seed'])
        mapper = ObservedRuntimeMapperV10(world.shape, world.config, truncation_m=.12)
        folder.joinpath('packets').mkdir()
        following = None; snapshots = {}; collisions = 0
        for paid in range(case['budget'] + 1):
            guard(2 * 1024 ** 2)
            packet = world.packet() if paid == 0 else world.step(following)
            save_packet(folder / 'packets' / f'{paid:03d}.npz', packet)
            validate_sensor_packet_v34(packet, world.transform, world.config)
            if packet.action_id != paid or packet.action != following or np.any(packet.frame.semantic):
                raise ValueError('paid observation/action/semantic boundary violated')
            xy = np.asarray(packet.frame.world_from_camera[:2, 3]) - world.shift[:2]
            if not np.allclose(xy, np.rint(xy), atol=1e-9, rtol=0): raise ValueError('off-grid odometry')
            pose = (*map(int, np.rint(xy)), int(packet.heading))
            collisions += int(packet.collision)
            mapper.update(packet.frame, packet.scan)
            controller.accept(ObservationV35(frame_id=f'paid-{paid}', step=paid, pose=pose,
                action=following, depth=packet.frame.depth_m, rgb=packet.frame.color_rgb,
                ranges=packet.scan.ranges_m, collision=bool(packet.collision), measured_map_sha256=array_hash(mapper.belief)))
            row = dict(paid=paid, action=following, pose=pose, packet_sha256=packet.sha256(),
                collision=bool(packet.collision), collisions=collisions,
                nonsemantic=dict(depth=array_hash(packet.frame.depth_m), scan=array_hash(packet.scan.ranges_m),
                                 rgb=array_hash(nonsemantic_rgb(packet.frame.color_rgb))))
            trace.append(row)
            next_action = controller.next_action()
            row['next_action'] = next_action
            history.append(dict(step=paid, posterior=controller.posterior_receipts[-1], state=deepcopy(controller.state), next_action=next_action))
            for stage, take in [('prefix', paid == 18), ('final', next_action is None)]:
                if not take: continue
                raw = mapper.mesh(); mesh, crop = extract_observed_asset_mesh(raw, info['public_bounds'])
                arrays(folder / f'{stage}_mesh.npz', vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles))
                arrays(folder / f'{stage}_map.npz', belief=mapper.belief)
                write(folder / f'{stage}_crop.json', crop)
                snapshots[stage] = dict(mesh=mesh, belief=mapper.belief.copy(), paid=paid, pose=pose, collisions=collisions)
            if next_action is None: break
            actions.append(next_action); following = next_action
        write(folder / 'trace.json', trace)
        write(folder / 'controller.json', dict(history=history, plans=controller.plans, calls=controller.calls,
              actions=actions, summary=controller.summary(), truth_or_selected_hypothesis_passed_to_controller=False))
        write(folder / 'prediction_freeze.json', {'before_first_reference_access': True,
              'artifacts': {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}})
        floor = world.evaluation_floor()
        arrays(folder / 'evaluation_floor.npz', reachable=floor['reachable'])
        evaluator = SurfaceMeasurementV34(world.instance_mesh(0), world.instance_mesh(0, vertical_only=True))
        for stage, snapshot in snapshots.items():
            coverage = float(np.mean(snapshot['belief'][floor['reachable']] != -1))
            measurements[stage] = evaluator.evaluate(snapshot['mesh'], coverage,
                returned=tuple(snapshot['pose']) == models[0].poses[models[0].anchor],
                collisions=snapshot['collisions'], failed=False, paid_actions=snapshot['paid'], budget=case['budget'])
        result = dict(case=case, status='completed', measurements=measurements, paid_actions=paid,
            controller_summary=controller.summary(), sensor_counts=dict(world.counts),
            trajectory_sha256=hashlib.sha256(payload(actions)).hexdigest(),
            prefix_nonsemantic_sha256=hashlib.sha256(payload([r['nonsemantic'] for r in trace[:19]])).hexdigest(),
            elapsed_seconds=time.monotonic()-started, real_geometry_used_for_terminal_evaluation=True,
            resource_usage_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()))
        write(folder / 'result.json', result)
    except BaseException as exc:
        signal.alarm(0)
        result = dict(case=case, status='failed', exception=repr(exc), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started, paid_actions=world.step_count if world else 0,
            sensor_counts=dict(world.counts) if world else {}, measurements=measurements,
            controller_summary=controller.summary() if controller else None,
            retry_allowed=False, retained_in_declared_denominator=True)
        write(folder / 'failure.json', result)
        if not (folder / 'trace.json').exists(): write(folder / 'partial_trace.json', trace)
        if controller and not (folder / 'controller.json').exists():
            write(folder / 'partial_controller.json', dict(history=history, plans=controller.plans,
                  calls=controller.calls, actions=actions, summary=controller.summary()))
    finally:
        signal.alarm(0)
    verify_sources()
    write(folder / 'seal.json', {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()})
    print(json.dumps({'case': case['id'], 'status': result['status'],
          'seconds': round(result['elapsed_seconds'], 2),
          'J5': measurements.get('final', {}).get('main_joint'),
          'eligible': measurements.get('final', {}).get('eligible')}), flush=True)


def run_all():
    config = verify_sources()
    for index, case in enumerate(config['cases']):
        folder = OUT / 'cases' / case['id']
        if folder.exists():
            if (folder / 'seal.json').exists(): continue
            raise RuntimeError('partial case exists; automatic retry prohibited: ' + case['id'])
        guard(8 * 1024 ** 2)
        completed = subprocess.run([sys.executable, '-B', str(Path(__file__)), 'case', '--index', str(index)], cwd=ROOT)
        if completed.returncode != 0:
            raise RuntimeError('case process exited without sealed result: ' + case['id'])
        if (index + 1) % 16 == 0:
            print(json.dumps({'completed_slots': index+1, 'declared_slots': len(config['cases']),
                  'bytes': used_bytes(), 'free_GiB': shutil.disk_usage(ROOT).free/GIB}), flush=True)


def analyze():
    config = verify_sources(); rows = []; results = []
    for case in config['cases']:
        folder = OUT / 'cases' / case['id']
        for rel, expected in read(folder / 'seal.json').items():
            if sha(folder / rel) != expected: raise ValueError('case artifact changed: ' + case['id'])
        result = read(folder / ('result.json' if (folder / 'result.json').exists() else 'failure.json'))
        results.append(result)
        measure = result.get('measurements', {}).get('final', {})
        row = dict(case, status=result['status'], elapsed_seconds=result['elapsed_seconds'],
                   paid_actions=result['paid_actions'], eligible=measure.get('eligible', False),
                   returned=measure.get('returned'), collisions=measure.get('collisions'),
                   C_map=measure.get('C_map'), failure=result.get('exception', ''))
        for threshold in ('02cm', '05cm', '10cm'):
            for name in ('precision', 'recall', 'f1', 'joint'):
                row[threshold + '_' + name] = measure.get(threshold, {}).get(name)
        rows.append(row)
    target = io.StringIO(); writer = csv.DictWriter(target, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows); write_bytes(OUT / 'episode_metrics.csv', target.getvalue().encode())
    grouped = defaultdict(dict)
    for row in rows: grouped[(row['scene'], row['hypothesis'], row['budget'], row['noise_seed'])][row['mode']] = row
    pairs = []
    for key, modes in grouped.items():
        scene, hypothesis, budget, noise_seed = key
        for treatment, control in [('S', 'G'), ('X', 'Xnf'), ('X', 'G')]:
            a, b = modes[treatment], modes[control]
            valid = bool(a['eligible'] and b['eligible'])
            diff = a['05cm_joint'] - b['05cm_joint'] if a['05cm_joint'] is not None and b['05cm_joint'] is not None else None
            pairs.append(dict(scene=scene, hypothesis=hypothesis, budget=budget, noise_seed=noise_seed,
                treatment=treatment, control=control, both_eligible=valid, delta_J5=diff,
                treatment_J5=a['05cm_joint'], control_J5=b['05cm_joint']))
    target = io.StringIO(); writer = csv.DictWriter(target, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    write_bytes(OUT / 'paired_metrics.csv', target.getvalue().encode())
    seed_groups = defaultdict(list)
    for row in rows:
        seed_groups[(row['scene'],row['hypothesis'],row['budget'],row['mode'])].append(row)
    seed_variation = []
    for key, group in seed_groups.items():
        values = [r['05cm_joint'] for r in group if r['05cm_joint'] is not None]
        mean = sum(values)/len(values) if values else None
        seed_variation.append(dict(scene=key[0],hypothesis=key[1],budget=key[2],mode=key[3],
            declared_replicates=len(group),measured_replicates=len(values),
            J5_mean=mean,J5_min=min(values) if values else None,J5_max=max(values) if values else None,
            J5_sample_sd=(sum((v-mean)**2 for v in values)/(len(values)-1))**.5 if len(values)>1 else None,
            interpretation='two noise realizations of the same geometry; not independent layouts'))
    target=io.StringIO(); writer=csv.DictWriter(target,fieldnames=list(seed_variation[0]));writer.writeheader();writer.writerows(seed_variation)
    write_bytes(OUT/'noise_seed_variation.csv',target.getvalue().encode())
    summaries = []
    for scene_group in [('originals', ('P00','P01')), ('variants', ('P00_rib_shift','P01_lower_ribs'))]:
        label, scenes = scene_group
        for budget in (30,42,54):
            for treatment, control in [('S','G'), ('X','Xnf'), ('X','G')]:
                group = [p for p in pairs if p['scene'] in scenes and p['budget'] == budget and p['treatment'] == treatment and p['control'] == control]
                if not group: continue
                all_measured = all(p['delta_J5'] is not None for p in group)
                result = dict(group=label, budget=budget, treatment=treatment, control=control,
                    declared_pairs=len(group), both_eligible=sum(p['both_eligible'] for p in group),
                    all_pairs_measured=all_measured, independent_layout_count=2,
                    two_noise_replicates_per_geometry_are_not_independent_layouts=True)
                if all_measured:
                    control_mean = sum(p['control_J5'] for p in group)/len(group)
                    delta = sum(p['delta_J5'] for p in group)/len(group)
                    result.update(control_mean=control_mean, treatment_mean=control_mean+delta,
                        mean_delta_J5=delta, relative_percent=100*delta/control_mean if control_mean else None,
                        wins=sum(p['delta_J5']>1e-12 for p in group), ties=sum(abs(p['delta_J5'])<=1e-12 for p in group),
                        losses=sum(p['delta_J5'] < -1e-12 for p in group))
                summaries.append(result)
    mixtures = []
    for key, modes in grouped.items():
        if any(modes[m]['05cm_joint'] is None for m in ('G','S','X')): continue
        for reliability in (0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.):
            mixed = reliability*modes['S']['05cm_joint']+(1-reliability)*modes['X']['05cm_joint']
            mixtures.append(dict(scene=key[0], hypothesis=key[1], budget=key[2], noise_seed=key[3],
                probability_correct_episode_category=reliability, mixture_J5=mixed,
                G_J5=modes['G']['05cm_joint'], delta_J5=mixed-modes['G']['05cm_joint'],
                analytic_mixture_not_new_reliability_experiment=True))
    target=io.StringIO(); writer=csv.DictWriter(target, fieldnames=list(mixtures[0]) if mixtures else
        ['scene','hypothesis','budget','noise_seed','probability_correct_episode_category','mixture_J5','G_J5','delta_J5','analytic_mixture_not_new_reliability_experiment']); writer.writeheader(); writer.writerows(mixtures)
    write_bytes(OUT/'analytic_category_mixture.csv',target.getvalue().encode())
    prefix_groups=defaultdict(list)
    for r in results:
        c=r['case']
        if 'prefix_nonsemantic_sha256' in r:
            prefix_groups[(c['scene'],c['hypothesis'],c['budget'],c['noise_seed'])].append(r['prefix_nonsemantic_sha256'])
    prefix_checks=[dict(condition=list(k),completed_arms=len(v),same_nonsemantic_prefix=len(set(v))==1) for k,v in prefix_groups.items()]
    write(OUT/'paired_prefix_checks.json',prefix_checks)
    write(OUT / 'summary.json', dict(declared_cases=len(rows), completed=sum(r['status']=='completed' for r in rows),
        failed=sum(r['status']=='failed' for r in rows), eligible=sum(r['eligible'] for r in rows),
        aggregate=summaries, total_episode_seconds=sum(r['elapsed_seconds'] for r in rows),
        all_cases_retained=True, all_available_paired_prefixes_identical=all(p['same_nonsemantic_prefix'] for p in prefix_checks),
        statistical_significance_claimed=False, unknown_layout_generalization_claimed=False,
        frozen_legacy_59_sources_unchanged=True, bytes_before_summary=used_bytes(), free_GiB=shutil.disk_usage(ROOT).free/GIB))
    print(json.dumps(read(OUT/'summary.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze','prepare','case','run','analyze'])
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command != 'freeze':
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            if os.environ.get(name) != '1': raise ValueError(name+' must equal 1')
    if args.command == 'case': run_case(args.index)
    else: {'freeze':freeze,'prepare':prepare,'run':run_all,'analyze':analyze}[args.command]()


if __name__ == '__main__':
    main()
