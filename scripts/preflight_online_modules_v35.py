#!/usr/bin/env python3
"""One frozen cached-observation V35 gate; zero new physical or TSDF trials."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import (ROOT, read, sha, write, write_bytes,
    freeze, verify_sources, verify_inventory, seal, json_value)
from nso.online_planner_v35 import load_public_models_v35, ForecastBeliefPlannerV35
from nso.observation_belief_v35 import PublicTemplatesV35, CONFIG_V35
from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35, ObservationV35

OUT = ROOT/'audit_results/v35_saved_observation_gate_20260918'
PIXELS = ROOT/'audit_results/v34_pixel_information_20260918'
PROTOCOL = ROOT/'docs/research/V35_SAVED_OBSERVATION_GATE_PROTOCOL_20260918.md'
MODES = ('G', 'S', 'swapped', 'swapped_no_feedback')


def run():
    if OUT.exists():
        raise FileExistsError('single preregistered batch; no overwrite or implicit retry')
    if not sys.dont_write_bytecode or any(os.environ.get(k) != '1' for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('single-thread -B required')
    verify_sources(PIXELS); verify_inventory(PIXELS)
    prior=ROOT/'audit_results/v33_direction_information_r1_20260917'
    verify_sources(prior); verify_inventory(prior)
    gates = [ROOT/'audit_results'/name for name in
        ('v35_online_planner_toy_tests_20260918',
         'v35_observation_belief_tests_20260918',
         'v35_four_module_controller_tests_20260918')]
    # Independent receipts use small per-tool schemas. Require passing tests
    # and freeze all receipt bytes, without assuming a metadata schema.
    for gate in gates:
        verify_inventory(gate)
        result = read(gate/'result.json')
        if (result.get('status') != 'passed' or result.get('exit_code') != 0
                or result.get('tests_run') != 11):
            raise ValueError('unpassed implementation receipt: '+str(gate))
        for rel, want in result['source_sha256'].items():
            if sha(ROOT/rel) != want:
                raise ValueError('tested implementation changed: '+rel)
    inputs = {str(p.relative_to(ROOT)): sha(p) for folder in [PIXELS, *gates,
        ROOT/'audit_results/v33_direction_information_r1_20260917']
        for p in folder.rglob('*') if p.is_file()}
    OUT.mkdir()
    started = time.monotonic()
    counts = dict(controllers=0, cached_observations=0, nominal_actions=0,
        online_planning_calls=0, online_planning_attempts=0,
        correction_intervention_selects=0, correction_intervention_attempts=0,
        memo_states=0, new_worlds=0,
        renderer_queries=0, sensor_packets=0, mapper_updates=0,
        TSDF_integrations=0, Q_evaluations=0, new_main_tasks=0)
    completed = []
    correction_interventions = []
    current = None
    controller = None
    def deadline(*_):
        raise TimeoutError('600-second cached-observation gate limit')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(600)
    try:
        manifest = freeze(OUT, [Path(__file__), PROTOCOL,
            ROOT/'tests/virtual3d/test_online_planner_v35.py',
            ROOT/'tests/virtual3d/test_observation_belief_v35.py',
            ROOT/'tests/virtual3d/test_cpu_four_modules_v35.py'],
            input_sha256=inputs, source_status='frozen_before_first_formal_controller',
            parent='P00', modes=MODES, total_budget=42, forced_prefix=18,
            likelihood_config=CONFIG_V35, main_tasks_used=19,
            scope='cached clean arrays and public templates; not physical acquisition')
        models, prefix = load_public_models_v35(ROOT, 'P00')
        templates = PublicTemplatesV35.from_saved(PIXELS, 'P00', models[0].poses)
        forecast = [templates.template_information(n) for n in range(len(models[0].poses))]
        informative = [row['node'] for row in forecast if row['informative']]
        write(OUT, OUT/'public_forecast.json', forecast)
        # Full actual cached arrays belong only to this test fixture.
        raw = []
        for h in (0, 1):
            with np.load(PIXELS/f'P00_h{h}_pixels.npz', allow_pickle=False) as archive:
                raw.append({k: np.array(archive[k], copy=True) for k in ('depth', 'rgb', 'ranges')})

        def consume(controller, h, node, step, action, erase=False):
            rgb = raw[h]['rgb'][node].copy()
            if erase:
                for color in CONFIG_V35['marker_colors'].values():
                    rgb[np.all(rgb == np.asarray(color, np.uint8), axis=-1)] = [127, 127, 127]
            obs = ObservationV35(frame_id=f'opaque-{step}', step=step,
                pose=tuple(models[0].poses[node]), action=action,
                depth=raw[h]['depth'][node], rgb=rgb, ranges=raw[h]['ranges'][node])
            controller.accept(obs)
            counts['cached_observations'] += 1
            if step:
                counts['nominal_actions'] += 1

        def next_action(controller):
            old = len(controller.plans)
            if controller.terminal_reason is None and controller.state['step'] >= 18:
                counts['online_planning_attempts'] += 1
            action = controller.next_action()
            for plan in controller.plans[old:]:
                if plan['planning']['phase'] == 'online_belief_global_planning':
                    counts['online_planning_calls'] += 1
                    counts['memo_states'] += plan['planning']['memo_states']
            return action

        for h in (0, 1):
            for mode in MODES:
                controller = CPUFourModuleControllerV35(models, templates, prefix,
                    mode=mode, informative_nodes=informative)
                counts['controllers'] += 1
                current = dict(hypothesis_fixture_only=h, mode=mode, history=[])
                node = models[0].anchor
                action = None
                actions = []
                for step in range(43):
                    consume(controller, h, node, step, action)
                    following_action = next_action(controller)
                    receipt=controller.posterior_receipts[-1]
                    # First corrected state only, prospectively fixed. Holding
                    # future observation capability constant isolates the
                    # actual posterior's immediate effect from anticipation.
                    if (mode=='swapped' and receipt['geometry_applied_log_odds']!=0
                            and receipt['probabilities'][h]>.5 and step>=18
                            and not any(r['hypothesis_fixture_only']==h for r in correction_interventions)):
                        counts['correction_intervention_attempts']+=1
                        semantic_only=1/(1+math.exp(-receipt['semantic_log_odds']))
                        alternative=ForecastBeliefPlannerV35(models,informative).select(
                            node,42-step,controller.masks,semantic_only,
                            geometry_feedback=True,excluded_information_nodes=controller.visited_nodes)
                        counts['correction_intervention_selects']+=1
                        counts['memo_states']+=alternative['memo_states']
                        correction_interventions.append(dict(hypothesis_fixture_only=h,step=step,
                            node=node,masks_hex=[hex(m) for m in controller.masks],
                            corrected_probability0=receipt['probabilities'][0],
                            semantic_only_probability0=semantic_only,
                            actual_action=following_action,counterfactual_action=alternative['action'],
                            action_changed=following_action!=alternative['action'],
                            shared_future_geometry_feedback=True,
                            forecast_intervention=alternative))
                    current['history'].append(dict(state=controller.state,
                        posterior=controller.posterior_receipts[-1], next_action=following_action))
                    if following_action is None:
                        break
                    actions.append(following_action)
                    node = dict(models[0].edges[node])[following_action]
                    action = following_action
                result = dict(hypothesis_fixture_only=h, mode=mode, actions=actions,
                    summary=controller.summary(), returned=node == models[0].anchor,
                    nominal_actions=step, final_potential=[model.terminal(mask)
                        for model, mask in zip(models, controller.masks)],
                    actual_C_map=None, actual_F1=None, measured_efficacy=False)
                result['qualified'] = result['returned'] and step <= 42 and all(
                    p['feasible'] for p in result['final_potential'])
                result['all_four_called'] = set(result['summary']['modules_called']) == {
                    'OV-SDF', 'IGCR', 'STGHP', 'RPN-UQ'}
                current.update(result=result, calls=controller.calls, plans=controller.plans)
                payload=json.dumps(json_value(current), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
                filename=f'h{h}_{mode}.json.gz'
                write_bytes(OUT, OUT/filename, gzip.compress(payload, mtime=0))
                result['trace_file']=filename
                completed.append(result)
                current=None
                print(json.dumps(dict(hypothesis=h, mode=mode, returned=result['returned'],
                    nominal_actions=step, qualified=result['qualified'],
                    first_suffix_action=actions[18] if len(actions)>18 else None,
                    p0=result['summary']['probabilities'][0],
                    elapsed_s=time.monotonic()-started)), flush=True)

        no_class = []
        for h in (0, 1):
            controller=CPUFourModuleControllerV35(models, templates, prefix,
                mode='S', informative_nodes=informative)
            counts['controllers'] += 1
            node=models[0].anchor; action=None
            for step in range(19):
                consume(controller,h,node,step,action,erase=True)
                action=next_action(controller)
                if step<18:
                    if action!=prefix[step]: raise AssertionError('prefix changed')
                    node=dict(models[0].edges[node])[action]
            g=next(r for r in completed if r['hypothesis_fixture_only']==h and r['mode']=='G')
            trace=json.loads(gzip.decompress((OUT/g['trace_file']).read_bytes()))['history'][18]
            no_class.append(dict(hypothesis_fixture_only=h,
                probabilities=controller.state['probabilities'], action=action,
                masks_equal=controller.state['masks_hex']==trace['state']['masks_hex'],
                probabilities_equal=controller.state['probabilities']==trace['state']['probabilities'],
                action_equal=action==trace['next_action']))
        contrasts=[]
        for h in (0,1):
            rows={r['mode']:r for r in completed if r['hypothesis_fixture_only']==h}
            def first(a,b):
                aa,bb=rows[a]['actions'],rows[b]['actions']
                return next((i+1 for i in range(max(len(aa),len(bb)))
                    if (aa[i] if i<len(aa) else None)!=(bb[i] if i<len(bb) else None)),None)
            contrasts.append(dict(hypothesis_fixture_only=h,
                S_vs_G_first_action=first('S','G'),
                swapped_vs_S_first_action=first('swapped','S'),
                correction_vs_no_feedback_first_action=first('swapped','swapped_no_feedback')))
        saved_noisy=read(gates[1]/'result.json')['diagnostic_summary']
        checks=dict(all_cache_routes_qualified=all(r['qualified'] for r in completed),
            all_four_modules_called=all(r['all_four_called'] for r in completed),
            no_class_matches_geometry=all(all(r[k] for k in ('masks_equal','probabilities_equal','action_equal')) for r in no_class),
            observed_class_changes_actions=any(r['S_vs_G_first_action'] is not None for r in contrasts),
            feedback_arms_change_actions=any(r['correction_vs_no_feedback_first_action'] is not None for r in contrasts),
            actual_posterior_correction_changes_action=any(r['action_changed'] for r in correction_interventions),
            saved_noisy_wrong_prior_corrected=saved_noisy['wrong_prior_true_class_probability_after_first_geometry']>.5
                and saved_noisy['feedback_disabled_true_class_probability']<.5)
        for rel,want in inputs.items():
            if sha(ROOT/rel)!=want: raise ValueError('frozen input changed: '+rel)
        verify_sources(OUT)
        write(OUT,OUT/'result.json',dict(status='complete', implementation_gate_passed=all(checks.values()),
            checks=checks, cached_rollouts=completed, no_class_controls=no_class,
            contrasts=contrasts, correction_interventions=correction_interventions,
            counts=counts, main_tasks_used=19,
            new_physical_tasks=0, measured_online_semantic_advantage=False,
            full_architecture_advantage_proven=False, source_count=len(manifest['source_sha256']),
            elapsed_s=time.monotonic()-started))
        seal(OUT)
        print(json.dumps(dict(status='complete',checks=checks,counts=counts)),flush=True)
    except BaseException:
        if current is not None:
            if controller is not None:
                current.update(calls=controller.calls, plans=controller.plans,
                    state=controller.state, posterior_receipts=controller.posterior_receipts)
            write_bytes(OUT,OUT/'partial_trace.json.gz',gzip.compress(
                json.dumps(json_value(current),allow_nan=False).encode(),mtime=0))
        write(OUT,OUT/'failure.json',dict(status='failed',error=traceback.format_exc(),
            completed=completed,counts=counts,elapsed_s=time.monotonic()-started,
            main_tasks_used=19,failed_planner_attempts_may_have_partial_unreported_memo_states=True))
        seal(OUT)
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    else:parser.print_help()
