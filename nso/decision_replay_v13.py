"""Measured history persistence for V13 development, without simulator access."""
from dataclasses import fields
import hashlib
import json

import numpy as np

from nso.cpu_sensor_contract_v10 import SensorPacket, digest, json_value
from utils.rgbd_contract import RGBDFrame, PlanarScan


def save_packet(path, packet):
    """Persist all sensor bytes and provenance; never serialize Python objects."""
    arrays = {}
    metadata = {}
    for field in fields(packet):
        value = getattr(packet, field.name)
        if field.name in ("frame", "scan"):
            for inner in fields(value):
                arrays[field.name + "__" + inner.name] = np.asarray(getattr(value, inner.name))
        else:
            metadata[field.name] = json_value(value)
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_packet(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        for name, cls in (("frame", RGBDFrame), ("scan", PlanarScan)):
            values = {}
            for field in fields(cls):
                array = data[name + "__" + field.name]
                values[field.name] = array.item() if array.ndim == 0 else array.copy()
            metadata[name] = cls(**values)
    metadata["position"] = tuple(metadata["position"])
    return SensorPacket(**metadata)


def array_hash(array):
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}:{array.shape}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def decision_state(runtime):
    """Auditable state identity, including mesh, feedback and execution phase.

    This is an equality certificate, not a restorable TSDF snapshot. Restore
    through every authorized action and observation from frame zero instead.
    Wall-clock profiling is deliberately excluded from deterministic identity.
    """
    state = runtime.states[0]
    backend = runtime.components._cpu_backend.scenes[0]
    mapper = state["mapper"]
    mesh = mapper.mesh()
    value = {
        "packet_hashes": state["packet_hashes"],
        "map_arrays": {name: array_hash(value) for name, value in vars(mapper).items()
                       if isinstance(value, np.ndarray)},
        "mesh": {name: array_hash(np.asarray(getattr(mesh, name))) for name in
                 ("vertices", "triangles", "vertex_colors")},
        "runtime": {key: state[key] for key in ("phase", "active_actions", "option",
                    "pending", "closed", "arrived_count")},
        "modules": runtime.components._cpu_backend.summary(0),
        "conditional_pending": {str(key): value for key, value in
                                backend["conditional_pending"].items()},
    }
    value = json_value(value)
    return {"sha256": digest(value), "evidence": value}


def replay_history(runtime, packets):
    """Replay paid history into a fresh, frame-zero-initialized runtime.

    Bootstrapping only the final map loses intermediate gain predictions and
    feedback. Reissuing the policy actions restores those state transitions.
    No future packet may be included when restoring a decision checkpoint.
    """
    if runtime.states[0]["packet"].sha256() != packets[0].sha256():
        raise ValueError("initial packet mismatch")
    for packet in packets[1:]:
        action = runtime.next_local_action(0)
        if action != packet.action:
            raise ValueError(f"history policy diverged at action {packet.action_id}")
        runtime.observe(0, packet.action_id, None, None, None, sensor_packet=packet)
    return runtime
