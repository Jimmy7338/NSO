#!/usr/bin/env python3
"""Read-only four-method AISLE causal-prefix diagnosis; no policy/quality replay."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'audit_results/article_stage_20260928'
METHODS = ('G', 'B', 'S', 'NBV')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


class Inputs:
    def __init__(self):
        self.pins = {}

    def read(self, path, pin=None):
        payload = path.read_bytes()
        record = dict(sha256=digest(payload), bytes=len(payload))
        require(pin is None or record['sha256'] == pin, 'input SHA ' + str(path))
        name = str(path.relative_to(ROOT))
        require(name not in self.pins or self.pins[name] == record, 'input changed ' + name)
        self.pins[name] = record
        return payload

    def json(self, path, pin=None):
        return json.loads(self.read(path, pin))


def planning_state(evidence):
    """Recover state after this paid frame's feedback, not its earlier association."""
    states = {x['instance_id']: deepcopy(x) for x in evidence['association']['instances']}
    for feedback in evidence['geometry_feedback']:
        if feedback['applied']:
            key = feedback['instance_id']
            require(feedback['paid_step'] == evidence['paid_step']
                    and feedback['observation_sha256'] == evidence['observation_sha256'], 'current feedback')
            require(feedback['instance']['geometric_feedback_frames'] == states[key]['geometric_feedback_frames']+1,
                    'one applied feedback')
            states[key] = deepcopy(feedback['instance'])
    for belief in evidence['structure_belief']['instances']:
        state = states[belief['instance_id']]
        require(state['geometry_log_scores'] == belief['geometry_log_evidence'], 'post-feedback evidence binding')
        state.update({k: deepcopy(belief[k]) for k in (
            'active_structure_prior', 'structure_probabilities', 'semantic_conditioning_used')})
        state['belief_observed_class'] = belief['observed_class']
        state['peer_instance_ids'] = belief['peer_instance_ids']
        state['rho'] = belief['rho']
    return states


def compact_state(state):
    return {k: state[k] for k in ('instance_id', 'observed_class', 'belief_observed_class',
        'association_uncertain', 'semantic_conditioning_used', 'active_structure_prior',
        'structure_probabilities', 'geometry_log_scores', 'geometric_feedback_frames',
        'peer_instance_ids', 'rho')}


def physical_packet_signature(inputs, root, manifest, step):
    result = {}
    for kind in ('rgbd', 'scan'):
        name = f'packets/{step:03d}_{kind}.npz'
        with zipfile.ZipFile(io.BytesIO(inputs.read(root/name, manifest['files'][name]['sha256']))) as archive:
            require(len(archive.namelist()) == len(set(archive.namelist())), 'unique packet arrays')
            result[kind] = {n: digest(archive.read(n)) for n in archive.namelist()
                            if not (kind == 'rgbd' and n == 'frame_id.npy')}
    return result


def run(output):
    require(not output.exists(), 'exclusive output directory')
    inputs = Inputs()
    snapshot = STAGE/'analysis_v1/exposure_AISLE_complete'
    sealed = inputs.json(snapshot/'manifest.json')
    for name, record in sealed['files'].items():
        inputs.read(snapshot/name, record['sha256'])
    require(inputs.json(snapshot/'findings.json') == [], 'released snapshot without findings')
    slots = list(csv.DictReader(io.StringIO(inputs.read(snapshot/'slots.csv').decode())))
    slots = {r['method']: r for r in slots if r['arm'] == 'ExposureV3' and r['scene_id'] == 'ART1_AISLE_DEV'}
    require(set(slots) == set(METHODS) and all(r['status'] == 'reviewed_qualified' for r in slots.values()),
            'all four fixed AISLE methods reviewed, without result selection')
    protocol_path = ROOT/'configs/virtual3d/article_exposure_ablation_v3_20260928.json'
    protocol = inputs.json(protocol_path, next(iter(slots.values()))['input_protocol_sha256'])
    for name in ('nso/controller_semantic_mechanism.py', 'nso/controller_article_v1.py',
                 'nso/controller_article_exposure_v3.py', 'nso/view_quality_article_exposure_v3.py'):
        inputs.read(ROOT/name, protocol['source_sha256'][name])
    episodes = {}; summaries = {}; selected = []; observes = []; timelines = []
    for method in METHODS:
        slot = slots[method]; run_id = slot['run_id']
        folder = STAGE/'exposure_ablation_v3/episodes'/run_id
        review_folder = STAGE/'episode_reviews_exposure_v3'/run_id
        rm = inputs.json(review_folder/'manifest.json', slot['input_review_manifest_sha256'])
        review = inputs.json(review_folder/'review.json', rm['files']['review.json']['sha256'])
        require(review['qualified'] and review['all_checks_passed']
                and review['run_id'] == run_id, 'independent qualified terminal review')
        manifest = inputs.json(folder/'artifact_manifest.json', slot['input_artifact_manifest_sha256'])
        require(manifest['protocol_sha256'] == slot['input_protocol_sha256'], 'episode protocol')
        def read(name):
            payload = inputs.read(folder/name, manifest['files'][name]['sha256'])
            return json.loads(gzip.decompress(payload) if name.endswith('.gz') else payload)
        result = read('result.json'); final = read('controller_final.json'); graph = read('public_graph.json')
        evaluation = read('evaluation.json')
        require(evaluation['metrics'] == review['metrics'], 'saved final evaluation binding')
        logs = {}; packets = {}; firsts = {}; class_events = []; previous_class = {}; return_step = None
        feedback_steps = {}; peer_frames = 0; max_peer_count = 0
        for step in range(result['acquired_and_saved_packets']):
            log = read(f'steps/{step:03d}.json.gz'); packet = read(f'packets/{step:03d}_receipt.json')
            logs[step] = log; packets[step] = packet
            evidence = log['controller_evidence']; decision = log['decision']
            states = planning_state(evidence)
            for key, state in states.items():
                first = firsts.setdefault(key, dict(observed=step))
                if state['semantic_conditioning_used']:
                    first.setdefault('semantic_used', step)
                if state['observed_class'] is not None and not state['association_uncertain']:
                    first.setdefault('qualified_observed_class', step)
                category = (state['observed_class'], state['belief_observed_class'], state['association_uncertain'])
                if previous_class.get(key) != category:
                    class_events.append(dict(paid_step=step, instance_id=key, observed_class=category[0],
                        belief_class=category[1], association_uncertain=category[2]))
                    previous_class[key] = category
                if state['peer_instance_ids']:
                    peer_frames += 1
                max_peer_count = max(max_peer_count, len(state['peer_instance_ids']))
            for key in evidence['first_actual_reliable_planes']:
                firsts[key].setdefault('reliable_plane', step)
            for feedback in evidence['geometry_feedback']:
                if feedback['applied']:
                    feedback_steps.setdefault(feedback['instance_id'], []).append(step)
            if decision.get('reason') == 'return' and return_step is None:
                return_step = step
            if packet['execution']['action'] == 'observe':
                observes.append(dict(method=method, paid_step=step, pose_xyyaw_rad=packet['execution']['pose_xyyaw_rad'],
                    completed_macro_id=evidence['completed_macro_id'],
                    applied_feedback=[f['instance_id'] for f in evidence['geometry_feedback'] if f['applied']]))
            if method in ('B', 'G') and step <= 31:
                timelines.append(dict(method=method, paid_step=step, states=[compact_state(s) for s in states.values()]))
            selection = decision.get('global_selection')
            if not decision['global_replanned'] or not selection or not selection.get('selected'):
                continue
            chosen = selection['selected']; target = chosen['target']; view = target['node']+':'+str(target['heading'])
            components = []
            for forecast in selection.get('forecasts', []):
                for candidate in forecast['candidates']:
                    if candidate['view_id'] == view:
                        probabilities = states[forecast['instance_id']]['structure_probabilities']
                        inspection = protocol['controller']['inspection_weight']*sum(p*g for p,g in zip(
                            probabilities, candidate['structure_new_surface_area_m2']))
                        components.append(dict(instance_id=forecast['instance_id'], inspection_gain=inspection,
                            fallback_reason=candidate.get('fallback_reason')))
            selected.append(dict(method=method, paid_step=step, macro_id=decision['macro_id'],
                target_xy_m=graph['nodes'][target['node']], remaining=selection['remaining'],
                choice=chosen if chosen['kind'] != 'diagnose_then_observe' else {
                    k:chosen[k] for k in ('kind', 'target', 'score', 'instance_id')},
                inspection_components=components,
                discovery_gain=(chosen['expected_gain']-sum(c['inspection_gain'] for c in components))
                    if chosen['kind'] == 'direct' else None))
        summaries[method] = dict(run_id=run_id, original_qualified=True, metrics={k:float(slot[k])
            for k in ('C_nav','P','R','F1','J_nav')}, per_instance_metrics=evaluation['metrics']['per_instance'],
            action_counts=dict(Counter(p['execution']['action'] for p in packets.values())),
            paid_actions=int(slot['paid_actions']),path_length_m=float(slot['path_length_m']),
            first_observation_events=firsts, class_events=class_events, applied_feedback_steps=feedback_steps,
            first_return_decision_step=return_step, final_budget_remaining=int(slot['final_budget_remaining']),
            actual_peer_instance_frames=peer_frames, maximum_peer_count=max_peer_count,
            initialization_actions_spent=final['initialization_actions_spent'],
            final_observed_instance_count=len(final['observed_instances']['instances']),
            planning_s=float(slot['planning_s']), evidence_s=float(slot['evidence_s']))
        episodes[method] = dict(folder=folder, manifest=manifest, logs=logs, packets=packets, graph=graph)

    # Independent physical comparison: all four terminal cohorts, with only the
    # run-namespaced frame ID removed. No scene, motion or sensor is replayed.
    signatures = {method: {step:physical_packet_signature(inputs,e['folder'],e['manifest'],step)
                          for step in e['logs']} for method,e in episodes.items()}
    pairs = {}
    for left,right in (('B','G'),('S','B'),('G','NBV')):
        common = sorted(set(signatures[left]) & set(signatures[right]))
        changed = [s for s in common if signatures[left][s] != signatures[right][s]]
        first = min(changed) if changed else None
        states = [s for s in common if first is None or s < first]
        posterior_changed = []
        geometry_differences = []
        for step in states:
            lhs=planning_state(episodes[left]['logs'][step]['controller_evidence'])
            rhs=planning_state(episodes[right]['logs'][step]['controller_evidence'])
            require(lhs.keys()==rhs.keys(), 'same-prefix observed identities')
            for key in lhs:
                require(lhs[key]['anchor_world_m']==rhs[key]['anchor_world_m'], 'same observed anchors')
                if lhs[key]['geometry_log_scores']!=rhs[key]['geometry_log_scores']:
                    geometry_differences.append([step,key])
                if lhs[key]['structure_probabilities']!=rhs[key]['structure_probabilities']:
                    posterior_changed.append([step,key])
                for state in (lhs[key],rhs[key]):
                    mass=[p*math.exp(l) for p,l in zip(state['active_structure_prior'],state['geometry_log_scores'])]
                    z=sum(mass)
                    require(max(abs(m/z-p) for m,p in zip(mass,state['structure_probabilities']))<1e-12,
                            'post-feedback posterior equals sealed prior times same measured evidence')
        pairs[left+'-'+right]=dict(common_actual_prefix_frames=len(states),
            first_physical_observation_divergence_paid_step=first,
            complete_physical_trajectory_equal=not changed and len(signatures[left])==len(signatures[right]),
            changed_posteriors_in_same_prefix=posterior_changed, changed_geometry_evidence_in_same_prefix=geometry_differences)
    require(pairs['B-G']['first_physical_observation_divergence_paid_step']==32,'published B/G witness')
    require(pairs['S-B']['complete_physical_trajectory_equal'] and not pairs['S-B']['changed_posteriors_in_same_prefix'],
            'exact S/B null remains visible')
    require(pairs['G-NBV']['complete_physical_trajectory_equal'],'G/NBV equality measured, not inferred from final score')

    choices={}; candidates=[]; structure_names=None
    for method in ('B','G'):
        episode=episodes[method]; log=episode['logs'][27]; selection=log['decision']['global_selection']
        state=planning_state(log['controller_evidence'])
        other=episodes['G' if method=='B' else 'B']['logs'][27]['decision']['global_selection']
        def physical_pool(pool, owner):
            pool=deepcopy(pool)
            for record in pool['measurement_records']:
                step=record['paid_step']; source=episodes[owner]['packets'][step]
                require(record['observation_sha256']==source['observation_sha256']
                    and record['frame_id']==episodes[owner]['logs'][step]['controller_evidence']['frame_id'], 'pool observation provenance')
                del record['observation_sha256']; del record['frame_id']
                record['verified_physical_packet']=signatures[owner][step]
            return pool
        require(physical_pool(selection['candidate_pool'],method)==physical_pool(other['candidate_pool'],'G' if method=='B' else 'B')
            and selection['instance_candidate_allocations']==other['instance_candidate_allocations'], 'same candidate availability')
        require(selection['remaining']==133,'same actual budget at first target divergence')
        for lf,rf in zip(selection['forecasts'],other['forecasts']):
            require(lf['instance_id']==rf['instance_id'],'same observed forecast instance')
            require(len(lf['candidates'])==len(rf['candidates']) and all(
                {k:v for k,v in l.items() if k not in ('expected_new_surface_area_m2','unexcluded_expected_new_surface_area_m2')}
                =={k:v for k,v in r.items() if k not in ('expected_new_surface_area_m2','unexcluded_expected_new_surface_area_m2')}
                for l,r in zip(lf['candidates'],rf['candidates'])), 'same physical component areas and gates')
        choices[method]=dict(selected=selection['selected'],state_at_choice=[compact_state(s) for s in state.values()],
            highest_diagnostic_score=max(x['score'] for x in selection['diagnostic_options']),
            direct_options_count=len(selection['direct_options']), diagnostic_options_count=len(selection['diagnostic_options']))
        require(selection['selected']['kind']=='direct','actual first choice is direct, not diagnostic')
        require(selection['selected']['score']==max(r['score'] for r in selection['direct_options']+selection['diagnostic_options']),
            'chosen numeric maximum')
        for option in selection['direct_options']:
            inspection=0.; parts=[]; view=option['target']['node']+':'+str(option['target']['heading'])
            for forecast in selection['forecasts']:
                structure_names=forecast['structure_names']
                for candidate in forecast['candidates']:
                    if candidate['view_id']!=view:continue
                    p=state[forecast['instance_id']]['structure_probabilities']
                    gain=protocol['controller']['inspection_weight']*sum(a*b for a,b in zip(p,candidate['structure_new_surface_area_m2']))
                    inspection+=gain;parts.append(dict(instance_id=forecast['instance_id'],posterior=p,
                        area_m2=candidate['structure_new_surface_area_m2'],inspection_gain=gain))
            require(abs(option['score']*option['total_cost']-option['expected_gain'])<1e-12,'score is gain per reserved cost')
            candidates.append(dict(method=method,view_id=view,**option,inspection_gain=inspection,
                discovery_gain=option['expected_gain']-inspection,parts=parts))
        macro=log['decision']['macro_id']
        finish=next(step for step,entry in episode['logs'].items() if step>27
                    and entry['controller_evidence']['completed_macro_id']==macro)
        receipt=episode['packets'][finish]['execution']
        require(receipt['action']=='observe','committed target finished with paid observation')
        choices[method]['actual_macro']=dict(selection_paid_step=27,completion_paid_step=finish,
            executed_actions_to_completion=finish-27,completion_pose_xyyaw_rad=receipt['pose_xyyaw_rad'],
            predicted_out_observe_return_cost=selection['selected']['total_cost'],
            reserved_return_cost=selection['selected']['total_cost']-(finish-27),
            feedback_before={k:s['geometric_feedback_frames'] for k,s in state.items()},
            feedback_after={k:s['geometric_feedback_frames'] for k,s in planning_state(
                episode['logs'][finish]['controller_evidence']).items()})
    for target in ('home:2','n_000_002:1'):
        rows=[r for r in candidates if r['view_id']==target]
        require(len(rows)==2 and abs(rows[0]['discovery_gain']-rows[1]['discovery_gain'])<1e-12,'same residual discovery utility')
    witness = {method:[dict(paid_step=step, execution=episodes[method]['packets'][step]['execution'],
                    next_action=episodes[method]['logs'][step]['decision']['action'],
                    routing=episodes[method]['logs'][step]['decision']['routing'],
                    safety_guard=episodes[method]['logs'][step]['decision']['safety_guard']) for step in (27,31,32,34,35)]
               for method in ('B','G')}
    summary=dict(schema='article.exposure_aisle_mechanism.v1',cohort='all four reviewed AISLE DEV methods; budget160 seed92801',
        scope='post-feedback saved state and exact physical packets; causal attribution restricted to same actual prefix',
        metric_version='article.common_numeric_face_evaluation.v1',structure_names=structure_names,methods=summaries,
        physical_pairs=pairs,first_choice=choices,first_action_witness=witness,
        new_worlds=0,new_controller_reexecutions=0,new_tsdf_integrations=0,new_quality_evaluations=0)
    output.mkdir(parents=True)
    for name,value in [('summary.json',summary),('step27_direct_candidate_arithmetic.json',candidates),
                       ('common_prefix_post_feedback.json',timelines),('all_selected_macros.json',selected),
                       ('all_paid_observations.json',observes)]:
        (output/name).write_bytes(canonical(value))
    source=Path(__file__).resolve();inputs.read(source)
    (output/'source.py').write_bytes(source.read_bytes())
    files={str(f.relative_to(output)):dict(sha256=digest(f.read_bytes()),bytes=f.stat().st_size)
           for f in sorted(output.iterdir()) if f.is_file()}
    (output/'manifest.json').write_bytes(canonical(dict(schema='article.exposure_aisle_mechanism_manifest.v1',
        files=files,inputs=inputs.pins,input_equality_rule='exact uncompressed NPY array bytes; only RGBD frame_id excluded',
        cap_bytes=1024**2)))
    size=sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
    require(size<=1024**2,'compact one MiB output cap')
    print(json.dumps(dict(output=str(output.relative_to(ROOT)),bytes=size,physical_pairs=pairs)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
