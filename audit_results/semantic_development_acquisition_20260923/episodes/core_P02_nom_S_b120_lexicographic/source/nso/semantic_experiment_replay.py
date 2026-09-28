"""Replay saved paid packets through the same driver and component factory.

SavedPacketSensor never constructs a World. ComparingWriter validates artifacts
in memory and cannot write into the immutable episode directory.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import numpy as np

from env.development_sensor_v41 import SensorStepV41,return_pose_matches,runtime_counts_v41
from nso.episode_driver_v43 import canonical_bytes,execute_episode_v43
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.saved_replay_v44 import _arrays,_canonical_mesh,_first_difference
from nso.semantic_experiment import inspect_experiment,make_components,read_json,evaluate_endpoint
from utils.rgbd_contract import PlanarScan


class SavedPacketSensor:
    def __init__(self,episode):
        self.episode=episode;self.root=episode['root'];self.index=0;self.closed=False;self.frame_ids=set()
        self.count=episode['result']['acquired_and_saved_packets']
        files=episode['manifest']['files']
        expected={f'packets/{i:03d}_{suffix}' for i in range(self.count)
                  for suffix in ('rgbd.npz','scan.npz','receipt.json')}
        if {name for name in files if name.startswith('packets/')}!=expected:
            raise ValueError('exact consecutive saved packet inventory required')
        first=read_json(self.root/'packets/000_receipt.json')['execution']['pose_xyyaw_rad']
        last=read_json(self.root/f'packets/{self.count-1:03d}_receipt.json')['execution']['pose_xyyaw_rad']
        returned=return_pose_matches(last,first)
        if episode['result']['sensor_status'].get('returned_xy_and_yaw')!=returned:
            raise ValueError('terminal returned flag differs from actual saved initial/final poses')

    def _capture(self,action):
        if self.closed or self.index>=self.count:raise ValueError('saved sequence exhausted or closed')
        prefix=f'packets/{self.index:03d}'
        receipt=read_json(self.root/(prefix+'_receipt.json'))
        if receipt['execution']['action']!=action:raise ValueError('replayed action differs from recorded paid action')
        values=_arrays(self.root/(prefix+'_rgbd.npz'),PaidRGBDObservationV40.__dataclass_fields__)
        for name in ('frame_id','paid_step'):values[name]=values[name].item()
        rgbd=PaidRGBDObservationV40.from_mapping(values)
        scan_values=_arrays(self.root/(prefix+'_scan.npz'),PlanarScan.__dataclass_fields__)
        for name in set(scan_values)-{'ranges_m','world_from_laser'}:scan_values[name]=scan_values[name].item()
        scan=PlanarScan(**scan_values)
        files=self.episode['manifest']['files']
        if (rgbd.frame_id in self.frame_ids or rgbd.paid_step!=self.index
                or receipt['observation_sha256']!=rgbd.sha256()
                or receipt['rgbd_artifact']!=files[prefix+'_rgbd.npz']
                or receipt['scan_artifact']!=files[prefix+'_scan.npz']):
            raise ValueError('saved observation identity or array receipt mismatch')
        self.frame_ids.add(rgbd.frame_id);self.index+=1
        return SensorStepV41(rgbd,scan,deepcopy(receipt['execution']))

    def initial_observation(self):
        if self.index:raise ValueError('one saved initial observation only')
        return self._capture('initial_observation')

    def step(self,action):
        if not self.index:raise ValueError('saved initial grant required before paid action')
        return self._capture(action)

    def close(self):
        self.closed=True
        if self.index!=self.count:raise ValueError('replay ended before complete saved trajectory')
        return deepcopy(self.episode['result']['sensor_status'])


class ComparingWriter:
    def __init__(self,episode):
        self.episode=episode;self.root=episode['root'];self.seen=set();self.first_difference=None
        count=episode['result']['acquired_and_saved_packets']
        encoding=read_json(self.root/'encoding.json')
        if encoding.get('schema')!='v44.step_encoding.v1' or len(encoding.get('steps',[]))!=count:
            raise ValueError('complete compressed-step ledger required')
        self.encoding={row['artifact']:row for row in encoding['steps']}
        expected={f'steps/{i:03d}.json.gz' for i in range(count)}
        if set(self.encoding)!=expected or {n for n in episode['manifest']['files'] if n.startswith('steps/')}!=expected:
            raise ValueError('exact saved step inventory required')

    def _fail(self,message):
        self.first_difference=self.first_difference or message
        raise ValueError(message)

    def arrays(self,name,**arrays):
        if name in self.seen:self._fail('duplicate replay artifact: '+name)
        expected=_arrays(self.root/name,tuple(arrays))
        if name=='prediction/mesh.npz':
            left,right=_canonical_mesh(expected),_canonical_mesh(arrays)
            if any(a.shape!=b.shape or not np.allclose(a,b,atol=1e-9,rtol=0) for a,b in zip(left,right)):
                self._fail('order-invariant complete mesh differs')
        elif any(not np.array_equal(expected[key],np.asarray(value)) for key,value in arrays.items()):
            self._fail('saved arrays differ: '+name)
        self.seen.add(name)
        return deepcopy(self.episode['manifest']['files'][name])

    def json(self,name,value,*,terminal=False):
        name=str(name)
        if name.startswith('steps/'):
            name+='.gz';encoding=self.encoding[name]
            if (encoding.get('stored_bytes')!=self.episode['manifest']['files'][name]['bytes']
                    or encoding.get('encoding')!='gzip-json-v1'
                    or not 0<encoding.get('uncompressed_bytes',0)<=self.episode['protocol']['maximum_file_bytes']):
                self._fail('saved compression metadata differs')
            with gzip.open(self.root/name,'rb') as stream:
                content=stream.read(self.episode['protocol']['maximum_file_bytes']+1)
            if len(content)!=encoding['uncompressed_bytes']:self._fail('saved step decompression length differs')
            expected=json.loads(content)
        else:expected=read_json(self.root/name)
        if name in self.seen:self._fail('duplicate replay artifact: '+name)
        if name=='result.json':
            # CPU time differs; every status/action/pose/fusion/failure field remains checked.
            expected={key:val for key,val in expected.items() if key not in ('timings','elapsed_s')}
            value={key:val for key,val in value.items() if key not in ('timings','elapsed_s')}
        difference=_first_difference(expected,value,name)
        if difference:self._fail(difference)
        self.seen.add(name)
        return deepcopy(self.episode['manifest']['files'][name])

    def complete(self):
        required={name for name in self.episode['manifest']['files']
                  if name.startswith(('packets/','steps/','prediction/')) or name=='result.json'}
        if self.first_difference is not None or self.seen!=required:
            raise ValueError(self.first_difference or 'incomplete replay artifact comparison')


def replay_saved_episode(episode):
    if not episode.get('current_sources_match'):
        raise ValueError('actual policy replay requires matching executed sources or an isolated archived environment')
    before=runtime_counts_v41();controller,mapper=make_components(episode['protocol'],episode['slot'],episode['bundle'])
    sensor=SavedPacketSensor(episode);writer=ComparingWriter(episode)
    began=time.monotonic()
    result=execute_episode_v43(sensor,controller,mapper,writer,budget=episode['slot']['budget'],
                              maximum_elapsed_s=episode['protocol']['maximum_elapsed_s'])
    writer.complete()
    after=runtime_counts_v41()
    if before!=after:raise ValueError('saved replay unexpectedly queried a live sensor')
    return dict(status='verified',verification_kind='same_driver_saved_packet_policy_and_TSDF_replay',
        frames_verified=sensor.index,prediction_verified=True,occupancy_exact=True,
        mesh_order_invariant_tolerance_m=1e-9,terminal_status_verified=result['status'],
        new_worlds=0,new_sensor_packets=0,physical_actions=0,
        replayed_tsdf_integrations=mapper.snapshot()['tsdf_integration_count'],
        runtime=dict(before=before,after=after),counterfactual_trajectory=False,
        elapsed_s=time.monotonic()-began)


def review_experiment(root,*,expected_manifest_sha256,output,replay_only=False):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.is_relative_to(root) or output.exists():raise ValueError('new review outside immutable episode required')
    episode=inspect_experiment(root,expected_manifest_sha256)
    result=dict(schema='semantic.experiment.review.v1',status='experiment_review_failed',
        run_id=episode['started']['run_id'],phase_id=episode['protocol']['phase_id'],
        episode_manifest_sha256=expected_manifest_sha256,
        source_sha256=episode['manifest']['source_sha256'],automatic_retry=False,
        current_sources_match=episode['current_sources_match'],
        current_source_differences=episode['current_source_differences'],
        primary_experiment_started=episode['protocol'].get('primary_experiment_started',False))
    began=time.monotonic();before=runtime_counts_v41()
    try:
        result['replay']=replay_saved_episode(episode)
        if not replay_only:result['evaluation']=evaluate_endpoint(episode)
        final_inspection=inspect_experiment(root,expected_manifest_sha256)
        if not final_inspection['current_sources_match']:
            raise ValueError('executed sources changed during actual review')
        result['status']='experiment_replay_verified' if replay_only else 'experiment_reviewed'
    except Exception as exc:
        result['error']=dict(type=type(exc).__name__,message=str(exc))
    after=runtime_counts_v41()
    result['runtime']=dict(before=before,after=after,no_new_world_or_sensor_action=before==after)
    if before!=after:result['status']='experiment_review_failed'
    result['elapsed_s']=time.monotonic()-began
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(result))
    return result
