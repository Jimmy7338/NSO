#!/usr/bin/env python3
"""One class-prior intervention in the unchanged B11 simulated action loop."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from junction import observed_gate, summarize_ranking

HERE = Path(__file__).resolve().parent
SCENES = HERE.parent / 'scenes'
STUDY = HERE.parents[1]
B11 = STUDY.parent / 'paired_unseen_gain_20261120/execution'
sys.path.insert(0, str(B11))
import runtime_binding as binding


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


PROTOCOL = read(HERE / 'protocol.json')
ARMS = tuple(PROTOCOL['arms'])
LAYOUTS = tuple(PROTOCOL['layouts'])


def execution_protocol():
    p = read(B11 / 'protocol.json')
    p['policy']['version'] = 'B11_neutral_shared_sequence_one_facility_class_prior_decision'
    p['policy']['paid_action_budget'] = PROTOCOL['budget_including_return']
    p['arms'] = {arm: dict(score_mode='G', predictor='visibility') for arm in ARMS}
    p['layouts'] = {layout: dict(scene_id=f'controlled-mechanism-{layout}') for layout in LAYOUTS}
    return p


def prepared(layout):
    return HERE / 'layouts' / layout / 'prepared'


def freeze():
    assert not (HERE / 'freeze.json').exists(), 'This study has already been frozen'
    assert not list((HERE / 'layouts').glob('*/runs/*/*')), 'Existing results found before freeze'
    parent = read(B11 / 'freeze.json')
    for path, expected in parent['files_sha256'].items():
        assert binding.sha(path) == expected, path
    files = [HERE / name for name in ('protocol.json', 'run.py', 'junction.py', 'supervisor.py', 'audit_matrix.py')]
    files += [SCENES / 'mechanism_world.py', SCENES / 'freeze.json', STUDY / 'PROTOCOL.json', B11 / 'freeze.json']
    for layout in LAYOUTS:
        p = prepared(layout)
        assert read(p / 'initial_state.json')['paid_actions'] == 0
        files += sorted(p.glob('*'))
    freeze_data = dict(status='frozen_controlled_mechanism_actor', frozen_unix_s=time.time(),
                       trial_schedule=[dict(layout=l, arm=a) for l in LAYOUTS for a in ARMS],
                       files_sha256={str(p): binding.sha(p) for p in files})
    binding.write(HERE / 'freeze.json', freeze_data)


def verify(layout=None):
    for name, check in read(HERE / 'freeze.json')['files_sha256'].items():
        assert binding.sha(name) == check, name
    for name, check in read(B11 / 'freeze.json')['files_sha256'].items():
        assert binding.sha(name) == check, name
    if layout is not None:
        return read(prepared(layout) / 'initial_state.json')


class SingleJunction:
    """Observe the original candidates without owning generation or execution."""

    def __init__(self, old, policy_module):
        self.old = old
        self.policy_module = policy_module
        self.active_backend = None
        self.active_state = None
        self.current_gate = None
        self.events = []
        self.trigger = None

    def install(self):
        backend_class = self.policy_module.ExplorationJointCPUFourModules
        original_generate = self.policy_module.generate_options
        original_select = backend_class.select_target
        original_score = backend_class._scores_v10_1

        def generate_options(*args, **kwargs):
            routes, audit = original_generate(*args, **kwargs)
            if self.active_state is None or self.trigger is not None:
                return routes, audit
            state = self.active_state
            graph = state['graph']
            # Same region legality predicate that select_target will apply to the
            # untouched returned list. No route is removed by this hook.
            legal = [r for r in routes if graph['current_region'] > 0 and
                     graph['current_region'] == graph['anchor_region'] ==
                     int(graph['regions'][tuple(r['pose'][:2])])]
            packet = state['packet']
            gate = observed_gate(packet, state['return_anchor'], legal)
            gate['packet_sha256'] = packet.sha256()
            gate['remaining_budget'] = int(state['ledger'].remaining_budget)
            gate['belief_sha256'] = hashlib.sha256(np.ascontiguousarray(state['mapper'].belief).tobytes()).hexdigest()
            if gate['qualified']:
                self.current_gate = gate
                self.active_backend.args.cpu_score_mode = self.active_backend.args.dual_template_arm
            else:
                self.events.append(gate)
            return routes, audit

        def score(backend, state, routes, attempted):
            mode = backend.args.cpu_score_mode
            backend.args.cpu_score_mode = 'G'
            try:
                scores, rows = original_score(backend, state, routes, attempted)
            finally:
                backend.args.cpu_score_mode = mode
            if self.current_gate is None:
                return scores, rows
            gate = self.current_gate
            assert [r['candidate_id'] for r in routes] == [r['candidate_id'] for r in gate['candidates']]
            cache = state['mapper']._dual_template_state['cache']
            original = cache['models']['S']
            swapped = copy.copy(original)
            swapped.audit = dict(original.audit)
            swapped.objects = copy.deepcopy(original.objects)
            shape_priors = []
            for src, dst in zip(original.objects, swapped.objects):
                dst['shape_prior'] = list(reversed(src['shape_prior']))
                shape_priors.append(dict(track_id=src['support_track'], S=list(src['shape_prior']),
                                         X=list(dst['shape_prior']), G=[.5, .5]))
            binding.revision.reweight(swapped, 16., original.audit['calibration'])
            np.testing.assert_array_equal(original.points, swapped.points)
            np.testing.assert_array_equal(original.history_keep, swapped.history_keep)
            assert swapped.bank is original.bank
            x_cache = {}
            scores['X'] = []
            for route, row, common in zip(routes, rows, scores['N']):
                value, _ = self.old.method.route_union_area(swapped, route['outbound_states'][1:], cache['ray'], x_cache)
                row['template_expected_visible_area_m2']['X'] = value
                row['prior_potentials']['X'] = value
                scores['X'].append(common + value)
            gate['shape_priors'] = shape_priors
            gate['candidate_summaries'] = summarize_ranking(gate['candidates'], scores, rows)
            return scores, rows

        def select_target(backend, scene_index):
            assert self.active_state is None
            self.active_backend = backend
            self.active_state = backend.scenes[scene_index]
            backend.args.cpu_score_mode = 'G'
            try:
                result = original_select(backend, scene_index)
                if self.current_gate is not None:
                    plan = self.active_state['last_selection']
                    assert plan['mode'] == backend.args.dual_template_arm
                    selected = plan['selected']
                    chosen = None if selected is None else int(selected['candidate_id'])
                    audit = self.current_gate
                    audit.update(mode=plan['mode'], chosen_candidate_id=chosen, chosen_native_option=selected,
                                 candidate_set_sha256=canonical(plan['candidates']),
                                 score_set_sha256=canonical(plan['scores']), native_plan_number=plan['plan_number'])
                    chosen_row = next((r for r in audit['candidate_summaries'] if r['candidate_id'] == chosen), None)
                    audit['selected_side'] = None if chosen_row is None else chosen_row['side']
                    self.trigger = audit
                return result
            finally:
                backend.args.cpu_score_mode = 'G'
                self.active_state = None
                self.active_backend = None
                self.current_gate = None

        self.policy_module.generate_options = generate_options
        backend_class._scores_v10_1 = score
        backend_class.select_target = select_target

    def receipt(self):
        return dict(status='triggered' if self.trigger is not None else 'nontrigger',
                    intervention=self.trigger, prior_failed_gate_events=self.events,
                    outside_intervention_score_mode='G', simulator_hidden_geometry_in_gate_or_scoring=False)


def install():
    old = binding.install()
    path = SCENES / 'mechanism_world.py'
    spec = importlib.util.spec_from_file_location('NSO_20261121_controlled_mechanism_world', path)
    scenes = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scenes
    spec.loader.exec_module(scenes)
    assert set(scenes.LAYOUTS) == set(LAYOUTS)
    from nso.cpu_sensor_contract_v10 import GridTransform
    import nso_bridge.exploration_joint_policy as policy_module

    def world_packet(layout):
        world = scenes.make_world(layout, seed=PROTOCOL['seed'])
        assert world.config.max_steps >= PROTOCOL['budget_including_return']
        transform = GridTransform(tuple(world.shape), world.config.resolution_m)
        return world, transform, old.packet(world, transform, world.sense(), layout)

    args = execution_protocol()
    binding.ARMS = args['arms']
    old.method.ARMS = ARMS
    old.method.MODE = {arm: 'G' for arm in ARMS}
    old.HERE, old.prepared, old.verify = HERE, prepared, verify
    old.protocol, old.world_and_packet = execution_protocol, world_packet
    gate = SingleJunction(old, policy_module)
    gate.install()
    return old, gate


def prepare(layout):
    old, _ = install()
    world, tr, packet, comp, runtime, effective, owner = old.bootstrap(layout, 'G')
    assert world.step_count == 0
    mapper = runtime.states[0]['mapper']
    assert not mapper.current_footprint_conflict
    dest = prepared(layout)
    dest.mkdir(parents=True, exist_ok=False)
    packet.frame.save(dest / 'initial_frame.npz')
    packet.scan.save(dest / 'initial_scan.npz')
    binding.write(dest / 'initial_state.json', dict(layout=layout, scene_id=packet.scene_id,
        scene_role='four_placement_two_parent_semantic_mechanism', paid_actions=0,
        initial_packet_sha256=packet.sha256(), initial_owner_sha256=owner['owner_sha256'],
        anchor=[*packet.position, packet.heading], grid_resolution_m=world.config.resolution_m,
        shape=list(world.shape), owner_counts=owner['owner_counts'], stop_budget=PROTOCOL['budget_including_return']))
    binding.write(dest / 'freeze.json', dict(status='first_packet_only_before_any_actor_choice',
        files_sha256={p.name: binding.sha(p) for p in sorted(dest.iterdir())}))


def run(layout, arm):
    verify(layout)
    old, gate = install()
    parent = HERE / 'layouts' / layout / 'runs' / arm
    existing = set(parent.glob('attempt_*'))
    try:
        old.run(layout, arm)
    finally:
        attempts = set(parent.glob('attempt_*')) - existing
        if len(attempts) == 1:
            out = attempts.pop()
            binding.write(out / 'intervention.json', gate.receipt())
            manifest = out / 'artifact_manifest.json'
            if (out / 'result.json').is_file():
                binding.write(manifest, dict(status='completed_execution',
                    files_sha256={p.name: binding.sha(p) for p in out.iterdir()
                                  if p.is_file() and p != manifest},
                    controlled_protocol_sha256=binding.sha(HERE / 'protocol.json'),
                    freeze_sha256=binding.sha(HERE / 'freeze.json'),
                    S_G_X_action_and_score_contract='One original selector mode change in an otherwise G_visibility trial.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'freeze', 'verify', 'run'))
    parser.add_argument('--layout', choices=LAYOUTS)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.layout)
    elif args.command == 'freeze':
        freeze()
    elif args.command == 'verify':
        verify(args.layout)
    elif args.layout is not None and args.arm is not None:
        run(args.layout, args.arm)
    else:
        parser.error('run needs a layout and arm')


if __name__ == '__main__':
    main()
