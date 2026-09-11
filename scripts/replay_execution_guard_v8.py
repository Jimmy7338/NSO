#!/usr/bin/env python3
"""Independent geometry/BFS and raw-sensor replay of the four guard regressions."""
import argparse
from collections import deque
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np
from scipy.ndimage import distance_transform_edt

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
def require(ok, text):
    if not ok: raise ValueError(text)
def ah(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(json.dumps([array.dtype.str, array.shape]).encode()+array.tobytes()).hexdigest()

def successor(state, action):
    r, c, h = state
    if action == 'forward':
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[h]
        return (r+dr, c+dc, h)
    require(action in ('left', 'right'), 'unpaid/unknown motion')
    return (r, c, (h+(1 if action == 'right' else -1)) % 4)

def inside(safe, cell):
    return 0 <= cell[0] < safe.shape[0] and 0 <= cell[1] < safe.shape[1]

def bfs_distance(safe, start, goal):
    todo = deque([(start, 0)]); seen = {start}
    while todo:
        state, cost = todo.popleft()
        if state == goal: return cost
        for action in ('left', 'right', 'forward'):
            nxt = successor(state, action)
            if nxt not in seen and inside(safe, nxt[:2]) and safe[nxt[:2]]:
                seen.add(nxt); todo.append((nxt, cost+1))
    return None

def check_plan(record, safe, state, anchor, budget):
    if not inside(safe, state[:2]) or not safe[state[:2]]:
        reason, cost = 'current_footprint_not_known_safe', 0
    elif not inside(safe, anchor[:2]) or not safe[anchor[:2]]:
        reason, cost = 'anchor_footprint_not_known_safe', 0
    else:
        distance = bfs_distance(safe, state, anchor)
        if distance is None: reason, cost = 'anchor_disconnected_in_latest_map', 0
        elif distance > budget: reason, cost = 'known_return_exceeds_remaining_budget', distance
        else: reason, cost = 'known_safe_return_within_budget', distance
    available = reason == 'known_safe_return_within_budget'
    require(record['reason'] == reason and record['available'] == available and record['paid_cost'] == cost,
            'return plan feasibility/independent BFS cost differs')
    if available:
        require(len(record['actions']) == cost, 'unpaid return rotations')
        for action in record['actions']:
            state = successor(state, action)
            require(inside(safe, state[:2]) and safe[state[:2]], 'return crosses unsafe footprint')
        require(state == anchor, 'return omits anchor heading')
    else:
        require(not record['actions'], 'unavailable plan contains actions')

def worker(args):
    sys.path.insert(0, str(args.snapshot))
    from env.virtual3d_response_v7 import ResponseConfigV7, ResponseContextV7, ResponseWorldV7
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from nso.counterfactual_view_scoring import _mapper_snapshot
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    # The production guard, route graph and execution loop are never called.
    reports = []; result = read(args.run / 'results.json')
    for context_id, family in (('T1','storage_shelves'),('T1','ventilation_baffles'),
                                ('T4','storage_shelves'),('T4','ventilation_baffles')):
        folder = args.run / context_id / family
        fixture = read(folder / 'fixture.json'); raw_context = fixture['context']
        context = ResponseContextV7(**(raw_context | {'offset_xy_m': tuple(raw_context['offset_xy_m'])}))
        config = ResponseConfigV7(**fixture['config']); world = ResponseWorldV7(context, family, config)
        mapper = SemanticHistoryMapperV3(world.shape, config, config.truncation_m)
        def sensors(frame, scan, frame_path, scan_path):
            for actual, expected in ((frame, RGBDFrame.load(frame_path)),(scan, PlanarScan.load(scan_path))):
                for field in fields(type(actual)):
                    np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))
        prefix = folder / 'prefix'; records = read(prefix / 'records.json')
        require(len(records) == 21, 'prefix length')
        for j, row in enumerate(records):
            if j:
                require(row['action'] == world.prefix_actions[j-1], 'prefix action')
                frame, collision, done = world.step(row['action'])
            else: frame, collision, done = world.sense(), False, False
            scan = world.scan()
            sensors(frame, scan, prefix/'frames'/f'{j:04d}.npz', prefix/'scans'/f'{j:04d}.npz')
            require(not collision and not done and (*world.position, world.heading) == (*row['position'], row['heading']), 'prefix state')
            mapper.update(frame, scan)
        branch = folder / 'guarded_candidate_005'; route = read(folder / 'route.json')
        anchor = (*world.position, world.heading)
        require(anchor == tuple(route['states'][0]) == tuple(route['states'][-1]), 'anchor')
        require(_mapper_snapshot(mapper)['hashes'] == read(branch/'initial_mapper_hashes.json'), 'prefix mapper hashes')
        actions = read(branch/'actions.json'); decisions = read(branch/'decisions.json')
        require([json.loads(line) for line in (branch/'actions.jsonl').read_text().splitlines()] == actions, 'action stream')
        require([json.loads(line) for line in (branch/'decisions.jsonl').read_text().splitlines()] == decisions, 'decision stream')
        paid = route_paid = return_paid = 0; states = [list(anchor)]; denials = []; terminal = None; mode = 'route'
        for index, d in enumerate(decisions):
            state = (*world.position, world.heading); remaining = 48-paid
            require(d['decision_id'] == index and d['paid_actions_before'] == paid and
                    d['absolute_step'] == world.step_count and tuple(d['position']) == world.position and
                    d['heading'] == world.heading and d['remaining_actions'] == remaining and
                    d['belief_sha256'] == ah(mapper.belief), 'decision did not use latest recorded observation')
            safe = distance_transform_edt(np.pad(mapper.belief == 0, 1, constant_values=False))[1:-1,1:-1] > config.robot_radius_m/config.resolution_m + np.sqrt(2)/2
            if d['kind'] == 'mode_switch':
                require(mode == 'route' and denials and d['previous_mode'] == 'route', 'invalid switch')
                mode = 'return'
            require(d['mode'] == mode, 'mode record')
            if d['kind'] == 'assessment':
                target = successor(state, d['action'])[:2]
                reason = ('no_action_budget' if remaining < 1 else
                          'current_footprint_not_known_safe' if not safe[state[:2]] else
                          'next_footprint_not_known_safe' if not inside(safe, target) or not safe[target] else
                          'latest_observed_grid_allows_action')
                allowed = reason == 'latest_observed_grid_allows_action'
                require(d['allowed'] == allowed and d['reason'] == reason and tuple(d['target']) == target, 'independent footprint gate differs')
                if mode == 'route': require(d['action'] == route['actions'][route_paid], 'route followed a different candidate')
                if not allowed:
                    denials.append({'decision_id': index, 'reason': reason, 'paid_actions_before': paid, 'action': d['action']})
                    continue
                a = actions[paid]
                require(a['assessment_decision_id'] == index and a['action'] == d['action'] and a['mode'] == mode, 'action lacks preceding gate')
                frame, collision, done = world.step(a['action']); scan = world.scan(); paid += 1
                sensors(frame, scan, branch/'frames'/f'{paid:04d}.npz', branch/'scans'/f'{paid:04d}.npz')
                expected = successor(state, a['action'])
                require(a['action_index'] == paid and a['absolute_step'] == world.step_count and a['collision'] == collision and a['done'] == done, 'actual action accounting')
                require(list(expected) == a['expected_state'] and (*world.position, world.heading) == (*a['position'], a['heading']), 'actual motion state')
                mapper.update(frame, scan)
                if collision:
                    blocked = successor((*world.position, world.heading), 'forward')[:2]
                    if inside(safe, blocked): mapper.belief[blocked] = 1
                require(ah(mapper.belief) == a['belief_after_sha256'], 'updated map differs')
                states.append([*world.position, world.heading])
                if mode == 'route': route_paid += 1
                else: return_paid += 1
            elif d['kind'].startswith('return_plan') or d['kind'] == 'return_replan_after_denial':
                check_plan(d, safe, state, anchor, remaining)
            elif d['kind'] == 'terminal_stop':
                terminal = d['reason']
                require(index == len(decisions)-1, 'actions after stop')
        require(paid == len(actions) and paid <= 48 and states == read(branch/'states.json'), 'final paid/state accounting')
        require(_mapper_snapshot(mapper)['hashes'] == read(branch/'final_mapper_hashes.json'), 'final mapper hashes')
        with np.load(branch/'final_map.npz') as saved:
            for key in saved.files: np.testing.assert_array_equal(saved[key], getattr(mapper, key))
        mesh = mapper.mesh()
        with np.load(branch/'final_mesh.npz') as saved:
            np.testing.assert_array_equal(saved['vertices'], np.asarray(mesh.vertices))
            np.testing.assert_array_equal(saved['triangles'], np.asarray(mesh.triangles))
        r = read(branch/'result.json')
        require(r in result['cases'] and r['paid_actions'] == paid and r['route_paid_actions'] == route_paid and
                r['return_paid_actions'] == return_paid and r['collisions'] == sum(a['collision'] for a in actions) and
                r['terminal_reason'] == terminal and r['first_denial'] == denials[0] and
                r['final_state'] == states[-1] and r['at_anchor_including_heading'] == (tuple(states[-1]) == anchor), 'result summary')
        require(terminal == 'returned_to_prefix_anchor' and not r['collisions'] and tuple(states[-1]) == anchor, 'guard regression did not return cleanly')
        reports.append({'context': context_id, 'family': family, 'paid_actions': paid, 'decisions': len(decisions), 'independent_BFS_and_gate_passed': True, 'raw_sensors_and_mesh_exact': True})
    require(result['paid_actions'] == sum(r['paid_actions'] for r in reports), 'total actions')
    dump(args.result, {'cases': reports, 'paid_actions': result['paid_actions'], 'cases_checked': 4})

def main(args):
    started = time.time(); own_sha = sha(Path(__file__)); report = {'status':'failed','passed_full':False}
    try:
        manifest = read(args.run/'artifact_hashes.json')
        def check():
            for name, digest in manifest.items(): require(sha(args.run/name) == digest, f'asset changed: {name}')
        check(); require(read(args.run/'metadata.json')['status']=='complete_execution_pending_independent_replay', 'incomplete diagnostic run')
        seal = read(args.run/'pre_execution_seal.json')
        with tempfile.TemporaryDirectory(prefix='independent-guard-replay-') as tmp:
            temp = Path(tmp); snapshot = temp/'sources'; snapshot.mkdir()
            with zipfile.ZipFile(args.run/'sources.zip') as archive:
                for name,digest in seal['source_sha256'].items():
                    target = snapshot/name
                    require(target.resolve().is_relative_to(snapshot.resolve()), 'unsafe archive member')
                    data = archive.read(name);require(hashlib.sha256(data).hexdigest()==digest,'archived source differs')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            result = temp/'result.json'
            command = [sys.executable,str(Path(__file__).resolve()),'--worker','--run',str(args.run),'--snapshot',str(snapshot),'--result',str(result)]
            completed = subprocess.run(command,cwd=temp,env=os.environ|{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
            require(completed.returncode==0, 'isolated worker failed')
            report.update(read(result))
        check(); require(sha(Path(__file__))==own_sha,'reviewer changed during replay')
        report.update(status='passed_full',passed_full=True,artifact_manifest_sha256=sha(args.run/'artifact_hashes.json'),
                      source_archive_sha256=sha(args.run/'sources.zip'),independent_confirmation=False,
                      production_guard_or_route_planner_called=False,raw_artifacts_modified=False,
                      scope='four known failure regressions only; independent gate/BFS, reused archived mapper/physical simulator')
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    report.update(elapsed_s=time.time()-started,verifier_sha256=own_sha)
    dump(args.run/'verification.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report['passed_full'] else 1

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=lambda x:Path(x).resolve(),required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--snapshot',type=Path);p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.worker: worker(a)
    else: raise SystemExit(main(a))
