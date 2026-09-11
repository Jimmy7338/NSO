"""Prospectively sealed V11.1 confirmation contexts.

This module reuses the audited V9 physical asset renderer while admitting only
the independently reserved E00--E11 manifest. It contains no reward, planner,
candidate, or learned-model input.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from env.virtual3d_competition_v9 import (CompetitionContextV9, CompetitionWorldV9,
    COMPETITION_ARRANGEMENTS, PREFIX_ACTIONS_P0)


DEFAULT_MANIFEST = (Path(__file__).resolve().parents[1]
                    / "configs/virtual3d/semantic_gain_v11_1_confirmation_contexts.json")


def sensor_seed(outer_seed, context_id):
    token = f"nso-semantic-gain-v11-confirmation/{outer_seed}/{context_id}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


@dataclass(frozen=True)
class ConfirmationContextV11(CompetitionContextV9):
    stratum: str = "efficacy"
    marker_width_m: float = .48

    def __post_init__(self):
        if self.role != "independent_confirmation" or self.stratum not in (
                "efficacy", "coverage_stress"):
            raise ValueError("invalid independent confirmation role or stratum")
        if not self.context_id.startswith("E") or self.outer_seed not in range(401, 413):
            raise ValueError("confirmation context is outside the reserved E seed block")
        if self.sensor_seed != sensor_seed(self.outer_seed, self.context_id):
            raise ValueError("invalid independently derived sensor seed")
        if self.prefix_step not in (80, 150) or self.depth_sigma_m not in (.01, .02):
            raise ValueError("invalid frozen prefix or noise condition")
        if not (.15 <= self.marker_width_m <= .6):
            raise ValueError("invalid marker visibility condition")
        if self.initial_heading != 0 or not np.allclose(self.initial_xy_m, (4.1, .7)):
            raise ValueError("confirmation start changed")


def load_confirmation_contexts(path=None):
    manifest = json.loads(Path(path or DEFAULT_MANIFEST).read_text())
    if (manifest["schema_version"] != "semantic_gain_v11_1_confirmation_contexts/1"
            or manifest["status"] != "sealed_before_world_construction"
            or manifest["arrangement_order"] != list(COMPETITION_ARRANGEMENTS)):
        raise ValueError("unknown or unsealed confirmation manifest")
    expected_ids = [f"E{i:02d}" for i in range(12)]
    if [row["context_id"] for row in manifest["contexts"]] != expected_ids:
        raise ValueError("confirmation context inventory changed")
    return manifest


def get_confirmation_context(context_id, path=None):
    row = next((r for r in load_confirmation_contexts(path)["contexts"]
                if r["context_id"] == context_id), None)
    if row is None:
        raise ValueError("undeclared confirmation context")
    step = int(row["prefix_step"])
    endpoint = {80: ((7.3, 3.7), 1), 150: ((4.1, 4.7), 0)}[step]
    if row["mirror_x"]:
        endpoint = {150: ((4.1, 4.7), 0)}.get(step)
        if endpoint is None:
            raise ValueError("mirrored short prefix was not prospectively declared")
    return ConfirmationContextV11(
        context_id=row["context_id"], role="independent_confirmation",
        outer_seed=int(row["outer_seed"]), sensor_seed=sensor_seed(
            int(row["outer_seed"]), row["context_id"]), mirror_x=bool(row["mirror_x"]),
        object_front_centers_xy_m=tuple(tuple(x) for x in row["object_front_centers_xy_m"]),
        initial_xy_m=(4.1, .7), initial_heading=0,
        final_prefix_xy_m=endpoint[0], final_prefix_heading=endpoint[1], prefix_step=step,
        depth_sigma_m=float(row["depth_sigma_m"]), object_width_m=float(row["object_width_m"]),
        object_depth_m=float(row["object_depth_m"]), stratum=row["stratum"],
        marker_width_m=float(row["marker_width_m"]))


class ConfirmationWorldV11(CompetitionWorldV9):
    def __init__(self, context, arrangement="shelf_west", config=None,
                 semantic_condition="aligned"):
        super().__init__(context, arrangement, config=config,
                         semantic_condition=semantic_condition)
        for marker in self._physical_markers:
            marker["width_m"] = context.marker_width_m
        self.prefix_actions = self.prefix_actions[:context.prefix_step]
        self.competition_truth["generator"] = "semantic_gain_v11_1_independent_confirmation"
        self.competition_truth["stratum"] = context.stratum


def create_confirmation_world(context_id, arrangement, config=None,
                              semantic_condition="aligned", manifest_path=None):
    return ConfirmationWorldV11(get_confirmation_context(context_id, manifest_path),
                                arrangement, config=config,
                                semantic_condition=semantic_condition)


def collect_confirmation_prefix(world):
    if world.step_count != 0:
        raise ValueError("confirmation prefix must start from a fresh world")
    rows = [{"step": 0, "action": None, "frame": world.sense(), "scan": world.scan(),
             "collision": False, "done": False}]
    for action in world.prefix_actions:
        frame, collision, done = world.step(action)
        rows.append({"step": world.step_count, "action": action, "frame": frame,
                     "scan": world.scan(), "collision": collision, "done": done})
        if collision or done:
            raise RuntimeError(f"sealed confirmation prefix failed at {world.step_count}")
    return rows
