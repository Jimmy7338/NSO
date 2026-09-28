#!/usr/bin/env python3
"""Pinned complete TARE node transfer; replay reexecutes saved actions only."""
import argparse
from collections import deque
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import select
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_external_routes_v39 import ExperimentV39, BUDGET, PREFIX, prior
from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35

OUTPUT = ROOT/'audit_results/v39_tare_adapter_20260920'
PROTOCOL = ROOT/'docs/research/V39_TARE_TRANSFER_PROTOCOL_20260920.md'
OVERRIDES = ROOT/'configs/virtual3d/v39_tare_transfer_overrides_20260920.json'
BRIDGE = ROOT/'scripts/tare_ros_bridge_v39.py'
RUNTIME = Path('/dev/shm/nso_v39_tare')
PINNED_NODE_SHA256 = '59fd9b818f4cc6a2edc3e5bf4a8641e2dad92b19ba1d824f1427624aecfd6f03'
PROJECTION_LIMIT = math.sqrt(.5)+1e-9
PHYSICS_TRACE_FIELDS = ('paid','action','pose_v33','packet_sha256','nonsemantic',
    'collision','collisions_so_far','measured_map_sha256','cue','next_action')


class NativeClientV39:
    """Private NDJSON process; never fabricates native output or sensor input."""
    def __init__(self, *, runtime=RUNTIME, port=11339):
        self.runtime = Path(runtime)
        self._closed = False
        self.diagnostics = []
        self.responses = 0
        self.stderr_path = self.runtime/'logs'/f'adapter-{os.getpid()}.stderr'
        self.stderr = self.stderr_path.open('w')
        self.process = subprocess.Popen([sys.executable,'-B',str(BRIDGE),
            '--runtime',str(self.runtime),'--port',str(port),
            '--overrides-json',str(OVERRIDES),'--planning-wait-s','5'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            text=True,bufsize=1,start_new_session=True)
        try:
            self.ready = self._receive(60.)
            if self.ready.get('status') != 'ready' or self.ready.get('node_sha256') != PINNED_NODE_SHA256:
                raise ValueError('native ready/binary contract mismatch')
            if not self.ready.get('native_initial_timer_observed'):
                raise ValueError('native initial timer handshake required')
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if not select.select([self.process.stdout],[],[],max(0.,deadline-time.monotonic()))[0]:
                break
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('native bridge exited: '+self.stderr_path.read_text()[-4000:])
            try:
                response = json.loads(line)
            except ValueError:
                self.diagnostics.append(line.rstrip()[:1000]); self.diagnostics=self.diagnostics[-20:]
                continue
            if isinstance(response, dict) and 'status' in response:
                return response
            self.diagnostics.append(line.rstrip()[:1000])
        raise TimeoutError('native bridge response timeout; no substitute waypoint')

    def request(self, message):
        if self._closed or self.process.poll() is not None:
            raise RuntimeError('native bridge unavailable')
        self.process.stdin.write(json.dumps(message,allow_nan=False)+'\n')
        self.process.stdin.flush()
        response = self._receive(15.)
        self.responses += 1
        return response

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n'); self.process.stdin.flush()
                self.process.wait(timeout=12.)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL); self.process.wait()
        self.stderr.close()
        for stream in (self.process.stdin,self.process.stdout):
            if stream is not None: stream.close()


class TareWaypointPlannerV39:
    """Only native waypoints choose exploration goals; BFS is local execution."""
    def __init__(self, models, translation, client=None, recorded_actions=None):
        self.models = models
        a = models[0]
        self.poses, self.edges, self.anchor = a.poses, a.edges, a.anchor
        self.translation = tuple(float(v) for v in translation[:2])
        self.paths = tuple(self._bfs(n) for n in range(len(self.poses)))
        self.client = client
        self.recorded_actions = None if recorded_actions is None else tuple(recorded_actions)
        self.step = -1
        self.response = None
        self.responses = []
        self.last_waypoint = None
        self.last_output_counter = None
        self.rejected_waypoints = []
        self.selections = []
        self.closed = False
        self.shutdown = None
        self.wait_started_step = None
        self.wait_turns = 0

    def _bfs(self, start):
        paths = {start: ()}; queue = deque([start])
        while queue:
            node = queue.popleft()
            for action, following in self.edges[node]:
                if following not in paths:
                    paths[following] = paths[node]+((action,following),)
                    queue.append(following)
        return paths

    def observe(self, observation, packet_path=None):
        if observation.step != self.step+1:
            raise ValueError('one consecutive paid observation per adapter update')
        self.step = observation.step
        if self.recorded_actions is not None:
            return dict(step=self.step,policy_reexecuted=False,
                scope='saved-action sensor/mapping replay; no native ROS publication')
        if self.client is None or packet_path is None:
            raise ValueError('a live original node and actual paid packet required')
        response = self.client.request(dict(op='observe',packet_path=str(packet_path),action_id=self.step))
        if response.get('status') != 'observed' or response.get('action_id') != self.step:
            raise ValueError('native observation acknowledgement mismatch')
        if response.get('unique_paid_observations') != self.step+1:
            raise ValueError('native bridge duplicated or skipped paid input')
        self.response = deepcopy(response)
        waypoint = response.get('waypoint')
        if waypoint is not None and waypoint.get('output_counter') != self.last_output_counter:
            self.last_output_counter = waypoint.get('output_counter')
            xyz = waypoint.get('xyz')
            valid = isinstance(xyz,list) and len(xyz)==3 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in xyz)
            if waypoint.get('frame_id') == 'map' and valid:
                self.last_waypoint = deepcopy(waypoint)
            else:
                self.last_waypoint = None
                self.rejected_waypoints.append(dict(step=self.step,reason='invalid_frame_or_nonfinite_waypoint',waypoint=waypoint))
        # Paths are native diagnostics, never substituted for a missing waypoint.
        compact = {k:v for k,v in response.items() if k!='paths'}
        compact['native_path_sizes'] = {k:len(v.get('xyz',[])) for k,v in response.get('paths',{}).items()}
        compact['last_accepted_waypoint'] = deepcopy(self.last_waypoint)
        self.responses.append(compact)
        return compact

    def select(self, node, remaining, masks, probability0, **ignored):
        if self.step < 0:
            raise ValueError('paid observation required')
        if self.recorded_actions is not None:
            action = self.recorded_actions[self.step] if self.step<len(self.recorded_actions) else None
            return dict(action=action,phase='recorded_action_replay',policy_reexecuted=False,
                recorded_execution_step=self.step+1 if action else None,posterior_used_to_choose_action=False)
        reason, target, projection = 'native_waypoint', None, None
        finished = bool(self.response and self.response.get('exploration_finished'))
        if finished:
            reason = 'native_finished_return'
        elif self.last_waypoint is None:
            reason = 'no_native_waypoint_return'
        else:
            wx,wy,wz = self.last_waypoint['xyz']
            xy = (wx-self.translation[0],wy-self.translation[1])
            candidates = [(math.hypot(p[0]-xy[0],p[1]-xy[1]),len(self.paths[node][i]),i)
                for i,p in enumerate(self.poses) if i in self.paths[node]]
            error,cost,target = min(candidates)
            projection = dict(native_xyz=list(self.last_waypoint['xyz']),native_frame='map',
                graph_node=target,graph_pose=list(self.poses[target]),xy_projection_error_m=error,
                maximum_projection_error_m=PROJECTION_LIMIT,z_used_to_plan=False)
            if error > PROJECTION_LIMIT:
                reason, target = 'native_waypoint_projection_rejected_return', None
            elif target == node:
                reason, target = 'native_goal_at_current_pose_no_progress_return', None
            elif cost+len(self.paths[target][self.anchor]) > remaining:
                reason, target = 'native_goal_not_return_affordable', None
        # One bounded native keypose cadence: genuine new, paid directions only.
        # This is explicitly adapter scheduling, not a TARE exploration goal.
        waiting_reason = reason in ('no_native_waypoint_return', 'native_goal_at_current_pose_no_progress_return')
        if waiting_reason:
            count = int((self.response or {}).get('published_registered_scans',0))
            until_due = 5-count%5
            due_after_wait = self.wait_started_step is not None and bool((self.response or {}).get('expected_native_planning_due'))
            next_right = dict(self.edges[node]).get('right')
            can_return = next_right is not None and 1+len(self.paths[next_right][self.anchor]) <= remaining
            if not due_after_wait and until_due<=4 and self.wait_turns<4 and can_return:
                if self.wait_started_step is None: self.wait_started_step=self.step
                self.wait_turns += 1
                result=dict(action='right',phase='adapter_paid_native_cadence_wait',
                    decision_reason=reason,projection=projection,adapter_wait_turn=self.wait_turns,
                    scans_until_next_native_keypose=until_due,native_finished=False,
                    reserved_return_actions=len(self.paths[next_right][self.anchor]),
                    posterior_used_to_choose_action=False,coverage_objective_replacement=False,
                    policy_reexecuted=True,native_goal_generated=False)
                self.selections.append(deepcopy(result))
                return result
            reason += '_cadence_exhausted_or_unaffordable'
        else:
            self.wait_started_step=None; self.wait_turns=0
        if target is None:
            target = self.anchor
        route = self.paths[node].get(target)
        if route is None or len(route)+len(self.paths[target][self.anchor]) > remaining:
            raise ValueError('no affordable safe-graph return; no alternative exploration planner')
        action = route[0][0] if route else None
        result = dict(action=action,phase='native_tare_waypoint_projection',decision_reason=reason,
            native_goal_node=target,projection=projection,native_finished=finished,
            native_waypoint_reused=bool(self.response and not self.response.get('fresh_waypoint')),
            projected_path_actions=[a for a,_ in route],reserved_return_actions=len(self.paths[target][self.anchor]),
            received_probability0_ignored=float(probability0),posterior_used_to_choose_action=False,
            coverage_objective_replacement=False,policy_reexecuted=True)
        self.selections.append(deepcopy(result))
        return result

    def close(self):
        if self.closed: return
        self.closed = True
        if self.client is not None:
            self.client.close()
            native_log=self.client.runtime/'logs'/f"node-{self.client.process.pid}.log"
            self.shutdown=dict(bridge_exit_code=self.client.process.returncode,
                bridge_stderr_tail=self.client.stderr_path.read_text()[-16000:],
                native_log_tail=native_log.read_text()[-16000:] if native_log.exists() else None,
                diagnostics=self.client.diagnostics)


class TareControllerV39(CPUFourModuleControllerV35):
    def select_target(self):
        if self.state is None or self.pending is not None or self.terminal_reason is not None:
            raise ValueError('active paid state without pending action required')
        step,node=self.state['step'],self.state['node']
        if self.selected is not None and self.selected['step']==step:
            return deepcopy(self.selected)
        if step<len(self.prefix_actions):
            result=dict(action=self.prefix_actions[step],phase='common_forced_prefix',posterior_used_to_choose_action=False)
        else:
            result=self.planner.select(node,self.budget-step,self.masks,self.belief.probabilities[0])
        action=result['action']; destination=self.links[node].get(action) if action is not None else None
        selected=dict(step=step,from_node=node,action=action,node=destination,
            pose=None if destination is None else list(self.poses[destination]),
            probabilities=list(self.belief.probabilities),planning=result)
        self.selected=deepcopy(selected); self.plans.append(deepcopy(selected))
        self._record('STGHP','execute_native_waypoint_on_public_graph',**selected)
        return deepcopy(selected)

    def summary(self):
        # Main evidence is sealed after this call. Stop writers before copying logs.
        self.planner.close()
        out=super().summary()
        out.update(external_method='TARE ground/RGB-D transfer',structural_posterior_used=False,
            original_native_core_executed=self.planner.recorded_actions is None,
            replay_scope='saved actions, fresh sensors/TSDF/metrics; policy not rerun',
            native_visibility_degrees=[360,24],camera_visibility_model_adapted=False,
            online_global_plans=sum(p['step']>=PREFIX for p in self.plans),
            native_ready=None if self.planner.client is None else self.planner.client.ready,
            native_publication_count=len(self.planner.responses),rejected_waypoints=self.planner.rejected_waypoints,
            adapter_paid_cadence_turns=sum(s['phase']=='adapter_paid_native_cadence_wait' for s in self.planner.selections),
            adapter_shutdown=self.planner.shutdown,
            scope='complete original native core with disclosed sensor/waypoint transfer; not narrow-FOV fair ranking')
        return out


class TareExperimentV39(ExperimentV39):
    def __init__(self,output=OUTPUT):
        super().__init__(output)

    def prepare(self):
        sources=(Path(__file__),BRIDGE,PROTOCOL,OVERRIDES,
            ROOT/'docs/research/V39_TARE_RUNTIME_PREFLIGHT_20260920.md',
            ROOT/'third_party/official_baselines/tare_official/src/tare_planner/config/garage.yaml',
            *(ROOT/'audit_results/v39_tare_runtime_preflight_20260920'/name for name in
              ('installed_packages.json','environment.json','verification.json')))
        super().prepare(methods=('TARE',),extra_sources=sources,scope='L2-TARE-transfer',
            config_overrides=dict(native_node_sha256=PINNED_NODE_SHA256,
                native_parameter_overrides=prior.read(OVERRIDES),native_visibility_degrees=[360,24],
                native_scan_messages_per_keypose=5,camera_visibility_model_adapted=False,
                adapter_maximum_paid_right_turns_per_wait=4,ros_master_port=11339,
                xy_projection_maximum_error_m=PROJECTION_LIMIT,
                native_replay_policy='recorded main actions only; no ROS policy rerun',
                early_stop_at_anchor_allowed=True,not_fair_narrow_FOV_ranking=True))

    def create_controller(self,case,acquisition):
        from nso.online_planner_v35 import load_public_models_v35
        from nso.observation_belief_v35 import PublicTemplatesV35
        models,prefix=load_public_models_v35(ROOT,case['parent'])
        templates=PublicTemplatesV35.from_saved(prior.INFORMATION,case['parent'],models[0].poses)
        replay=getattr(self,'replay',False)
        actions=prior.read(self.case_folder/'result.json')['actions'] if replay else None
        client=None if replay else NativeClientV39()
        try:
            if client is not None:
                h,w=acquisition['raster_shape']; scale=acquisition['raster_resolution_m']
                client.request(dict(op='boundary',xy=[[0.,0.],[w*scale,0.],[w*scale,h*scale],[0.,h*scale]]))
            planner=TareWaypointPlannerV39(models,acquisition['translation'],client,actions)
            controller=TareControllerV39(models,templates,prefix,mode='G',total_budget=BUDGET,
                informative_nodes=acquisition['informative_nodes'],planner=planner)
            return controller,models
        except BaseException:
            if client is not None: client.close()
            raise

    def after_observation(self,controller,observation,packet):
        path=None if self.replay else self.case_folder/'packets'/f'{observation.step:03d}.npz'
        return controller.planner.observe(observation,path)

    def close_controller(self,controller):
        controller.planner.close()

    def verify_replay_policy(self,trace,evidence,folder):
        saved=prior.read(folder/'trace.json')
        if len(trace)!=len(saved) or any(any(a[k]!=b[k] for k in PHYSICS_TRACE_FIELDS) for a,b in zip(trace,saved)):
            raise AssertionError('fixed-action fresh sensor/map trace differs')
        if evidence['actions']!=prior.read(folder/'controller.json')['actions']:
            raise AssertionError('recorded action execution differs')
        self.write(folder/'replay'/'policy_scope.json',dict(policy_reexecuted=False,
            compared_trace_fields=list(PHYSICS_TRACE_FIELDS),fresh_native_ROS_publications=0,
            all_saved_actions_executed=True,claim='sensor/fusion/measurement repeatability only'))

    def decorate_result(self,result):
        result.update(original_TARE_native_core_executed=not self.replay,
            native_node_sha256=PINNED_NODE_SHA256,native_visibility_degrees=[360,24],
            camera_visibility_model_adapted=False,native_policy_reexecuted_in_replay=False,
            comparison_scope='disclosed ground/RGB-D transfer; not narrow-FOV fair ranking')
        return result

    def finish_replay(self,result,expected,counts,folder,work):
        fields=('physical_case','stages','counts','paid_actions','returned','collisions','actions')
        if any(prior.digest(result[k])!=prior.digest(expected[k]) for k in fields):
            self.write(work/'mismatch_result.json',result)
            raise AssertionError('saved-action physical reconstruction replay differs')
        self.write(work/'result.json',dict(passed=True,main_result_sha256=prior.sha(folder/'result.json'),
            fresh_process=True,all_saved_actions_sensor_packets_maps_meshes_scores_equal=True,
            all_controller_decisions_reexecuted=False,policy_reexecuted=False,
            native_ROS_publications=0,counts=counts,compared_result_fields=list(fields),
            scope='fixed-main-trajectory sensor/fusion/metric repeatability; not independent policy rerun'))


def prefix_smoke(output):
    """Actual ROS calls on the 19 existing common-prefix packets; no new World."""
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    if shutil.disk_usage(ROOT).free<66*1024**2: raise RuntimeError('preserve 64 MiB disk reserve')
    output.mkdir()
    acquisition=prior.read(prior.OUTPUT/'config.json')['parents']['P00']
    experiment=TareExperimentV39()
    experiment.replay=False
    experiment.case_folder=prior.OUTPUT/'case00'
    controller=None; receipts=[]; sources={}
    try:
        import numpy as np
        from nso.cpu_four_modules_v35 import ObservationV35
        from nso.decision_replay_v13 import load_packet
        controller,_=experiment.create_controller(dict(parent='P00'),acquisition)
        for step in range(PREFIX+1):
            path=experiment.case_folder/'packets'/f'{step:03d}.npz'
            sources[str(path.relative_to(ROOT))]=prior.sha(path)
            packet=load_packet(path)
            obs=ObservationV35(frame_id=f'opaque-{step}',step=step,
                pose=prior.observed_pose(packet,acquisition['translation']),action=packet.action,
                depth=packet.frame.depth_m,rgb=packet.frame.color_rgb,ranges=packet.scan.ranges_m,
                collision=packet.collision,measured_map_sha256=None)
            controller.accept(obs)
            receipt=experiment.after_observation(controller,obs,packet)
            action=controller.next_action()
            if step<PREFIX and action!=controller.prefix_actions[step]:
                raise AssertionError('common prefix changed')
            receipts.append(dict(observation=step,bridge=receipt,next_action=action))
        result=dict(passed=True,actual_original_TARE_node=True,saved_prefix_frames=19,
            new_worlds=0,new_sensor_queries=0,new_TSDF_integrations=0,new_quality_evaluations=0,
            next_action_after_prefix=receipts[-1]['next_action'],summary=controller.summary(),
            receipts=receipts,input_sha256=sources,
            source_sha256={str(p.relative_to(ROOT)):prior.sha(p) for p in (Path(__file__),BRIDGE,OVERRIDES,PROTOCOL)},
            scope='interface smoke only; no physical performance outcome')
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(passed=True,saved_frames=19,next_action=result['next_action_after_prefix'])),flush=True)
    except BaseException:
        import traceback
        (output/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),receipts=receipts),indent=2)+'\n')
        raise
    finally:
        if controller is not None: experiment.close_controller(controller)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','main','replay','analyze','prefix-smoke'))
    parser.add_argument('--index',type=int)
    parser.add_argument('--smoke-output',type=Path,default=ROOT/'audit_results/v39_tare_prefix_smoke_20260920')
    args=parser.parse_args(); experiment=TareExperimentV39()
    if args.command=='prepare': experiment.prepare()
    elif args.command=='analyze': experiment.analyze()
    elif args.command=='prefix-smoke': prefix_smoke(args.smoke_output)
    else:
        if args.index is None: parser.error('--index required')
        experiment.run_case(args.index,replay=args.command=='replay')


if __name__=='__main__': main()
