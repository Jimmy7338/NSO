"""Finite already-declared packet fixture; no World, rendering, or live actions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from env.development_sensor_v41 import SensorStepV41
from nso.analytic_fixture_v42 import analytic_forward_sequence_v42
from nso.episode_driver_v43 import BoundedRunWriterV43, execute_episode_v43, file_sha256
from nso.observed_mapper_v42 import _array_sha256
from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41
from nso.saved_replay_v44 import SOURCE_NAMES, PROTOCOL_NAME
from utils.rgbd_contract import PlanarScan

ROOT = Path(__file__).resolve().parents[1]


def fixed_packet_steps_v44():
    result = []
    for index, packet in enumerate(analytic_forward_sequence_v42()):
        laser = np.eye(4); laser[:3, 3] = [packet.world_from_camera[0, 3], .75, .3]
        scan = PlanarScan(float(index), np.zeros(4), -np.pi, np.pi/2, 8., laser)
        execution = dict(action='initial_observation' if index == 0 else 'forward', paid_step=index,
            action_cost=int(index > 0), collision=False, pose_xyyaw_rad=[float(packet.world_from_camera[0, 3]), .75, 0.])
        if index:
            execution['pose_before_xyyaw_rad'] = [.75, .75, 0.]
        else:
            execution['initial_frames'] = 1
        result.append(SensorStepV41(packet, scan, execution))
    return tuple(result)


class FiniteMapperV44:
    def __init__(self):
        self.receipts = []
    def update(self, observation, scan):
        receipt = dict(frame_id=observation.frame_id, paid_step=observation.paid_step,
                       observation_sha256=observation.sha256())
        self.receipts.append(receipt)
        return deepcopy(receipt)
    def occupancy_arrays(self):
        return np.full((20, 20), -1, np.int8), np.zeros((20, 20), bool)
    def mesh_arrays(self):
        return dict(vertices=np.empty((0, 3)), triangles=np.empty((0, 3), np.int32), vertex_colors=np.empty((0, 3)))
    def snapshot(self):
        return dict(frames=len(self.receipts), tsdf_integration_count=0, receipts=deepcopy(self.receipts),
                    occupancy_sha256=_array_sha256(self.occupancy_arrays()[0]))


class FiniteControllerV44:
    def __init__(self):
        self.index = -1
    def accept(self, observation, mapper, *, execution_outcome):
        self.index += 1
        return dict(observation_sha256=observation.sha256(), finite_fixture=True)
    def choose(self):
        return dict(action='forward' if self.index == 0 else 'blocked', reason='finite predeclared fixture', paid_step=self.index)


class FiniteSensorV44:
    def __init__(self, packets, returned=False):
        self.packets = packets; self.returned = returned; self.index = 0
    def initial_observation(self):
        return self.packets[0]
    def step(self, action):
        self.index += 1
        if self.index >= len(self.packets):
            raise RuntimeError('finite fixture exhausted; no new observations may be requested')
        return self.packets[self.index]
    def close(self):
        return dict(returned_xy_and_yaw=self.returned, finite_fixture=True)


def create_finite_episode_v44(directory, *, real_controller=False, compressed=False):
    """Create V43-format saved finite fixtures, explicitly not study episodes.

    Real controller uses one initial grant, home-only graph and budget one,
    which selects stop while respecting the return reserve. No action executes.
    Doubles use two precomputed packets and predeclared forward/blocked commands.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT/PROTOCOL_NAME).read_text())
    sources = set(SOURCE_NAMES)
    if compressed:
        from nso.evidence_writer_v44 import CompressedStepWriterV44
        sources |= {'scripts/run_development_v44.py', 'nso/evidence_writer_v44.py'}
        writer = CompressedStepWriterV44(directory)
    else:
        writer = BoundedRunWriterV43(directory)
    source_sha = {name: file_sha256(ROOT/name) for name in sorted(sources)}
    for name in sorted(sources):
        writer._write('source/'+name, (ROOT/name).read_bytes())
    packets = fixed_packet_steps_v44()
    graph_spec = dict(schema_version='v41.public_navigation.v1', source_kind='provided_navigation_prior',
                      nodes={'home': [.75, .75]}, edges=[])
    if not real_controller:
        graph_spec['nodes']['advance'] = [1., .75]; graph_spec['edges'] = [['home', 'advance']]
    graph = PublicPrimitiveGraphV41(graph_spec)
    asset = ROOT/'audit_results/v40_p1_development_geometry_20260920/DEV_A_00'
    public = json.loads((asset/'public_planner_spec.json').read_text())
    workspace = json.loads((asset/'public_workspace.json').read_text())
    public['task']['max_actions'] = 1
    workspace['bounds_xy_m'] = [[0., 0.], [2., 2.]]
    writer.json('started.json', dict(run_id='R3_A_S', slot=protocol['slots']['R3_A_S'],
        source_sha256=source_sha, protocol_sha256=source_sha[PROTOCOL_NAME], public_graph_sha256=graph.input_sha256,
        finite_fixture=True, fixture_scope='precomputed packet wiring only; no study World or physical trajectory'))
    writer.json('public_graph.json', graph_spec); writer.json('public_spec.json', public)
    writer.json('public_workspace.json', workspace)
    if real_controller:
        from nso.controller_v43 import ANSControllerV43
        from nso.observed_mapper_v42 import ObservedMapperV42
        prior = public['structure_prior']
        controller = ANSControllerV43(graph, home=PrimitiveStateV41('home', 0), budget=1,
            palette=workspace['marker_palette'], structure_names=prior['abstract_structures'],
            class_structure_prior=prior['probability_by_category'], mode='S', **protocol['controller'])
        mapper = ObservedMapperV42(shape=(20, 20), origin_xy_m=(0., 0.), **protocol['mapper'])
        packets = packets[:1]
    else:
        controller, mapper = FiniteControllerV44(), FiniteMapperV44()
    result = execute_episode_v43(FiniteSensorV44(packets, returned=real_controller), controller, mapper, writer, budget=1)
    writer.json('runtime.json', dict(world_created=False, source_unchanged=True, evaluation_executed=False,
                                    finite_fixture=True), terminal=True)
    if compressed:
        writer.json('encoding.json', dict(schema='v44.step_encoding.v1', steps=writer.step_encoding), terminal=True)
    writer.json('artifact_manifest.json', dict(scope='finite saved-packet fixture; not actual study run',
        files=writer.files, source_sha256=source_sha), terminal=True)
    return dict(root=directory, result=result, real_controller=real_controller, source_sha256=source_sha)
