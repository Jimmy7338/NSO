"""Replay frozen external scans with the unchanged shared RGB-D/scan/action interface."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import ANALYSES, HERE, SOURCES


MODEL = ANALYSES / "semantic_scene_redesign_20261119"
sys.path[0:0] = [str(MODEL / "scene_adapter"), str(MODEL / "assets")]
from replicacad_world import ReplicaCADConfig, ReplicaCADWorld
from textured_raycast import TexturedRayRenderer


BUDGET = 192
SOURCE = Path(__file__).resolve()
REGISTRATION = HERE / "STARTS.json"
REGISTERED = json.loads(REGISTRATION.read_text()) if REGISTRATION.exists() else {"layouts": {}}
SPECS = REGISTERED["layouts"]
LAYOUTS = tuple(SPECS)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def source_scene(floor):
    path = SOURCES[floor]["path"]
    result = trimesh.load(path, force="scene", process=False)
    if result.is_empty or not len(result.graph.nodes_geometry):
        raise ValueError(f"Source is not a renderable triangle scene: {floor}")
    return result


class ScannedWorld(ReplicaCADWorld):
    """Only use the inherited depth renderer, raycast scan and swept-edge step."""

    def __init__(self, layout):
        spec = SPECS[layout]
        self.config = ReplicaCADConfig(max_steps=BUDGET)
        c = self.config
        self.spec = spec
        self.layout, self.seed = layout, spec["seed"]
        self.shape = tuple(spec["shape"])
        self.height, self.width = np.asarray(self.shape) * c.resolution_m
        self.start = tuple(spec["start"])
        self.position, self.heading = self.start, spec["heading"]
        self.step_count = self.collisions = self.moves = 0
        self.world_from_source = np.asarray(spec["world_from_source"])
        self.source_from_world = np.linalg.inv(self.world_from_source)
        cache = HERE / "precomputed_assets" / f'{spec["source_floor"]}.npz'
        with np.load(cache, allow_pickle=False) as saved:
            self._occupancy = saved["occupancy"].copy()
            self._robot_valid = saved["robot_valid"].copy()
            self._edge_valid = saved["edge_valid"].copy()
            self._rooms = saved["room_labels"].copy()
        self._reachable = self._flood(self.start)
        if not self._robot_valid[self.start] or not self._reachable.any():
            raise ValueError("Registered 0.20 m footprint fails precomputed swept mesh graph")
        self._source_scene = source_scene(spec["source_floor"])
        self._renderer = TexturedRayRenderer(self._source_scene)
        self._ray = self._renderer.scene
        self._frame_cache, self._open_gids, self._closed_regions = {}, [], []
        f = c.width_px / (2 * np.tan(np.deg2rad(c.fov_deg) / 2))
        self.intrinsic = np.array([[f, 0, (c.width_px - 1) / 2], [0, f, (c.height_px - 1) / 2], [0, 0, 1.]])
        self.geometry_receipt = dict(spec, capsule_cache_sha256=sha(cache), rendered_geometry_from_scan=True,
                                     eligible_source_triangles=sum(x["triangles"] for x in self._renderer.stats),
                                     semantic_input="zero/current-observation-only", reference_never_in_actor=True)

    def sense_visible_instances(self):
        return np.zeros((self.config.height_px, self.config.width_px), dtype=np.uint16)

    def evaluator_truth(self):
        return {**super().evaluator_truth(), "room_labels": self._rooms.copy()}


def make_world(layout):
    return ScannedWorld(layout)


def build_scene(layout):
    spec = SPECS[layout]
    source = source_scene(spec["source_floor"])
    scene = trimesh.Scene()
    transform = np.asarray(spec["world_from_source"])
    metadata = {}
    for name in source.graph.nodes_geometry:
        local, geo = source.graph[name]
        scene.add_geometry(source.geometry[geo], geom_name=name, node_name=name, transform=transform @ local)
        metadata[name] = dict(instance_id=0, source_instance_id=0, semantic_class=0, owner=0)
    return scene, [], metadata
