#!/usr/bin/env python3
"""Frozen finite geometric value-of-information screen, r0 or one justified r1."""
import argparse
from pathlib import Path
import signal
import sys
from time import perf_counter
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import ROOT, read, write, freeze, verify_sources, seal
from nso.direction_information_v33 import (DirectionGeometryV33, pair_geometry, RADAR_RAYS,
    RADAR_HEIGHT, RADAR_RANGE, VERTICAL_FOV, VERTICAL_SPACING, HORIZONTAL_SPACING,
    CAMERA_HEIGHT, CAMERA_RANGE, HORIZONTAL_FOV, ROBOT_RADIUS)
from nso.finite_belief_solver_v33 import ExactCoverageBeliefSolverV33, SolverLimitV33, NEG

PROTOCOL = ROOT/'docs/research/V33_DIRECTION_INFORMATION_PROTOCOL_20260917.md'


def run(revision):
    suffix = '' if revision == 'r0' else '_r1'
    config_path = ROOT/f'configs/virtual3d/v33_direction_scene{suffix}_20260917.json'
    out = ROOT/f'audit_results/v33_direction_information_{revision}_20260917'
    if out.exists(): raise FileExistsError('frozen run already exists; no implicit retry')
    cfg = read(config_path)
    assert cfg['hypothesis_prior'] == [.5,.5]
    obs, radar = cfg['geometric_observation'], cfg['radar_observation']
    assert abs(obs['vertical_fov_deg']-VERTICAL_FOV) < 1e-10
    assert obs['surface_vertical_spacing_max_m'] == VERTICAL_SPACING
    assert obs['surface_horizontal_spacing_max_m'] == HORIZONTAL_SPACING
    assert (obs['camera_height_m'],obs['range_m'],obs['horizontal_fov_deg']) == (CAMERA_HEIGHT,CAMERA_RANGE,HORIZONTAL_FOV)
    assert (radar['rays'],radar['height_m'],radar['range_m']) == (RADAR_RAYS,RADAR_HEIGHT,RADAR_RANGE)
    assert all(p['robot_radius_m'] == ROBOT_RADIUS for p in cfg['parents'])
    out.mkdir(); started = perf_counter(); models_all = []; solvers = []; parents = []
    counts = dict(analytic_geometry_models=0, exact_solver_instances=0, memo_states=0,
        worlds=0, sensor_packets=0, mapper_updates=0, TSDF_integrations=0, Q_evaluations=0, new_main_tasks=0)
    def remaining_state_cap():
        left = cfg['solver_limit']['maximum_memo_states']-sum(s.states for s in solvers)
        if left <= 0: raise SolverLimitV33('cumulative memo-state cap exhausted')
        return left
    def timeout(*_): raise SolverLimitV33('fixed300-second wall cap; optimum not certified')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(cfg['solver_limit']['wall_seconds'])
    try:
        sources = [Path(__file__), config_path, PROTOCOL,
            ROOT/'docs/research/V33_DIRECTION_SCENE_GEOMETRY_20260917.md',
            ROOT/'docs/research/V33_FINITE_SOLVER_REVIEW_20260917.md',
            ROOT/'tests/virtual3d/test_finite_belief_solver_v33.py',
            ROOT/'tests/virtual3d/test_direction_scene_v33.py',
            ROOT/'nso/direction_scene_v33.py']
        if revision == 'r1': sources.append(ROOT/'docs/research/V33_R1_GEOMETRY_CORRECTION_20260917.md')
        manifest = freeze(out, sources, status='frozen_before_geometry_and_policy_calculation',
            revision=revision, main_tasks_used=16, physical_execution_forbidden=True,
            thresholds_cm={'2':None,'5':None,'10':None}, precision_threshold_status='untested_not_applicable_to_ideal_visibility',
            versions=dict(numpy=np.__version__), independent_abstract_solver_tests=17)
        for parent in cfg['parents']:
            models = []
            for h in (0,1):
                counts['analytic_geometry_models'] += 1
                models.append(DirectionGeometryV33(parent, h))
            models_all.append(models)
            tables = [m.tables() for m in models]; paired = pair_geometry(models)
            geometry_gates = dict(shared_action_graph=paired['identical_legal_graph'],
                connected=paired['all_safe_connected'], paired_prefix=paired['identical_prefix_geometry'],
                all_targets_observable=all(t['all_target_representatives_observable'] for t in tables),
                equal_mirror_area=all(abs(models[0].areas[k]-models[1].areas[k]) < 1e-9 for k in models[0].areas))
            row = dict(parent_id=parent['id'], geometry_gates=geometry_gates, geometry_passed=all(geometry_gates.values()),
                pair=paired, tables=tables)
            parents.append(row); write(out, out/(parent['id']+'_geometry.json'), row)
            print(dict(stage='geometry_complete',parent=parent['id'],gates=geometry_gates,
                missing=[len(t['unobservable_targets']) for t in tables], poses=len(models[0].poses)), flush=True)
        all_geometry = all(p['geometry_passed'] for p in parents)
        policy_rows = []
        if all_geometry:
            for parent, models in zip(cfg['parents'], models_all):
                remaining = parent['total_action_budget']-len(parent['prefix_actions'])
                if remaining != 24: raise ValueError('frozen remaining budget changed')
                initial = [m.prefix()[1] for m in models]
                solver = ExactCoverageBeliefSolverV33(models, initial, remaining_state_cap())
                solvers.append(solver); counts['exact_solver_instances'] += 1
                g = solver.uncertain(solver.anchor, remaining, *initial)
                known = [solver.known(h, solver.anchor, remaining, initial[h]) for h in (0,1)]
                feasible = min(g,*known) > NEG/2
                witnesses = []
                if feasible:
                    for h in (0,1):
                        for policy in ('G','class_oracle','swapped_class_with_correction'):
                            w = solver.witness(h, remaining, policy, hint=1-h if policy.startswith('swapped') else None)
                            w['total_paid_actions'] = len(parent['prefix_actions'])+w['suffix_paid_actions']
                            witnesses.append(w)
                    for policy, expected in [('G',g),('class_oracle',sum(known)/2)]:
                        actual = sum(w['terminal']['joint'] for w in witnesses if w['policy'] == policy)/2
                        if abs(actual-expected)>1e-10: raise AssertionError('policy witness expectation differs')
                # Identical-structure control has the same two formal IDs,
                # but no information can change their common physical model.
                ordinary = ExactCoverageBeliefSolverV33([models[0],models[0]], [initial[0],initial[0]],
                    remaining_state_cap())
                solvers.append(ordinary); counts['exact_solver_instances'] += 1
                ordinary_g = ordinary.uncertain(ordinary.anchor, remaining, initial[0],initial[0])
                ordinary_k = ordinary.known(0, ordinary.anchor, remaining, initial[0])
                control_pass = ordinary_g>NEG/2 and abs(ordinary_g-ordinary_k)<=1e-10
                mean = sum(known)/2 if feasible else None
                gain = mean-g if feasible else None
                relative = gain/g if feasible and g>0 else None
                row = dict(parent_id=parent['id'], exact_search_completed=True, feasible=feasible,
                    G_optimal=g if feasible else None, class_optimal_by_hypothesis=known if feasible else None,
                    class_optimal_mean=mean, information_absolute=gain, information_relative=relative,
                    screening_passed=bool(control_pass and relative is not None and relative>.05),
                    witnesses=witnesses, main_memo_states=solver.states, ordinary_memo_states=ordinary.states,
                    controls=dict(identical_structure=dict(G=ordinary_g,known=ordinary_k,passed=control_pass),
                        early_geometry_reveal=dict(value=mean,scope='algebraic_singleton_belief_not_new_measurement'),
                        independent_class=dict(value=g if feasible else None,scope='posterior_unchanged_optimal_cue_ignored'),
                        swapped_with_geometric_correction=dict(values=[w['terminal'] for w in witnesses if w['policy'].startswith('swapped')])),
                    optimum_scope='all legal finite-grid history policies; not continuous-motion optimality')
                policy_rows.append(row); write(out, out/(parent['id']+'_policies.json'), row)
                print(dict(stage='policies_complete',parent=parent['id'],G=row['G_optimal'],oracle=mean,
                           relative=relative,passed=row['screening_passed'],states=sum(s.states for s in solvers)),flush=True)
                solver.clear(); ordinary.clear()
                for m in models: m.terminal.cache_clear()
        counts['memo_states'] = sum(s.states for s in solvers)
        forbidden = [name for name in sys.modules if name in ('env.virtual3d','nso.mapping3d','nso.observed_runtime_mapper_v10')]
        if forbidden: raise AssertionError('physical module imported: '+str(forbidden))
        verify_sources(out)
        result = dict(status='complete' if all_geometry else 'geometry_gate_failed',
            revision=revision, all_geometry_gates_passed=all_geometry,
            all_finite_information_gates_passed=bool(all_geometry and len(policy_rows)==2 and all(r['screening_passed'] for r in policy_rows)),
            parents=[dict(parent_id=r['parent_id'],geometry_gates=r['geometry_gates']) for r in parents],
            policy_summaries=[{k:v for k,v in r.items() if k not in ('witnesses',)} for r in policy_rows],
            counts=counts, main_tasks_used=16, source_count=len(manifest['source_sha256']),
            precision_thresholds_cm={'2':None,'5':None,'10':None}, precision_threshold_status='untested',
            actual_C80_verified=False, physical_experiment_ready=False, full_architecture_advantage_proven=False,
            semantic_efficacy_proven=False, elapsed_seconds=perf_counter()-started)
        write(out, out/'result.json', result); seal(out); print(result, flush=True)
    except BaseException as error:
        counts['memo_states'] = sum(s.states for s in solvers)
        write(out, out/'failure.json', dict(status='censored' if isinstance(error,SolverLimitV33) else 'failed',
            error=traceback.format_exc(), counts=counts, elapsed_seconds=perf_counter()-started,
            exact_optimum_certified=False, main_tasks_used=16))
        if not (out/'artifact_hashes.json').exists(): seal(out)
        raise
    finally:
        signal.alarm(0)
        for solver in solvers: solver.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--revision', choices=('r0','r1'), default='r0')
    args = parser.parse_args()
    if args.run: run(args.revision)
    else: parser.print_help()
