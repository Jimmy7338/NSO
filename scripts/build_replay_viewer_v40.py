#!/usr/bin/env python3
"""Export a bounded, offline viewer of four SAVED V39 runs; no experiment imports.

Only explicit whitelists cross into the display payload. Stored endpoint meshes
are rendered with Pillow, with one camera shared by all methods. No World,
sensor, planner, TSDF integrator, or surface evaluator is instantiated.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION

ROOT = Path(__file__).resolve().parents[1]
COHORT = "audit_results/v39_external_cpu_20260920"
DEFAULT_OUT = ROOT / "docs/thesis/demos/v40_replay"
MAX_BYTES = 20_000_000
RESERVE = 64 * 1024 * 1024
CASES = (("case00", "G", "h0"), ("case01", "S", "h0"),
         ("case04", "G", "h1"), ("case05", "S", "h1"))
COLORS = {"G": (115, 121, 132), "S": (0, 114, 178)}
CAMERAS = (("oblique", "共享斜视", 24, 55),
           ("reverse", "共享反向", 24, -55),
           ("top", "共享俯视", 76, 55))
FRAME_KEYS = ("step", "action", "next_action", "pose", "timestamp_s", "rgb", "depth",
              "depth_points_xy", "valid_depth_fraction", "posterior", "cue",
              "planning", "source")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


class Sources:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.records = {}

    def read(self, relative: str) -> bytes:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("source outside project")
        data = path.read_bytes()
        record = {"sha256": digest(data), "bytes": len(data)}
        previous = self.records.get(relative)
        if previous is not None and previous != record:
            raise ValueError("source changed during export: " + relative)
        self.records[relative] = record
        return data

    def json(self, relative: str):
        return json.loads(self.read(relative))

    def verify(self):
        for relative, record in list(self.records.items()):
            if digest((self.root / relative).read_bytes()) != record["sha256"]:
                raise ValueError("source changed during export: " + relative)


class Assets:
    def __init__(self):
        self.data = {}
        self.bytes = 0

    def image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        raw = buffer.getvalue()
        key = digest(raw)
        if key not in self.data:
            encoded = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
            self.bytes += len(encoded)
            if self.bytes > MAX_BYTES - 1_000_000:
                raise ValueError("embedded image budget exceeded")
            self.data[key] = encoded
        return key


def depth_rgb(depth: np.ndarray) -> np.ndarray:
    """Shared 0--4 m display; invalid/absent values are grey, never zero error."""
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 4)
    stops = np.array([[32, 63, 94], [36, 111, 141], [82, 164, 158],
                      [166, 200, 153], [246, 224, 124]], dtype=float)
    scaled = np.clip(np.nan_to_num(depth, nan=0, posinf=4, neginf=0), 0, 4)
    lo = np.minimum(scaled.astype(int), 3)
    fraction = scaled - lo
    color = stops[lo] * (1 - fraction[..., None]) + stops[lo + 1] * fraction[..., None]
    color[~valid] = (215, 220, 226)
    return np.rint(color).astype(np.uint8)


def valid_number(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) and result > -1e20 else None


def sanitize_planning(plan: dict | None) -> dict:
    if plan is None:
        return {"phase": "finished", "uses_belief": False, "candidates": [],
                "objective": "public-template proxy; not measured F1 or Q"}
    raw = plan.get("planning", {})
    candidates = []
    for item in raw.get("candidates", []):
        feasible = bool(item["return_feasible"])
        candidates.append({"action": str(item["action"]),
                           "expected_proxy": valid_number(item.get("expected_proxy")) if feasible else None,
                           "return_feasible": feasible,
                           "forecast_information": bool(item.get("forecast_information", False))})
    return {"phase": str(raw.get("phase", "unknown")),
            "uses_belief": bool(raw.get("posterior_used_to_choose_action", False)),
            "candidates": candidates,
            "objective": "public-template proxy; not measured F1 or Q"}


def sanitize_posterior(history: dict) -> dict:
    raw = history["posterior"]
    probabilities = [float(value) for value in raw["probabilities"]]
    if len(probabilities) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities):
        raise ValueError("invalid two-template weights")
    if abs(sum(probabilities) - 1) > 1e-8:
        raise ValueError("weights do not sum to one")
    return {"weights": probabilities,
            "semantic_log_odds": float(raw["semantic_log_odds"]),
            "geometry_log_odds": float(raw["geometry_log_odds"]),
            "geometry_applied_log_odds": float(raw["geometry_applied_log_odds"]),
            "new_geometry_pose": bool(raw["new_geometry_pose"]),
            "class_conflict": bool(raw["class_conflict"]),
            "calibrated": False, "scope": "global two-template structural weights"}


def checkpoint_at(step: int) -> str | None:
    if not 0 <= step <= 42:
        raise ValueError("paid step outside saved run")
    return "final" if step == 42 else "prefix" if step >= 18 else None


def validate_alignment(trace, history, plans):
    if len(trace) != 43 or len(history) != 43 or len(plans) != 42:
        raise ValueError("expected the complete 42-action saved trajectory")
    for step, (row, observation) in enumerate(zip(trace, history)):
        if row["paid"] != step or observation["step"] != step:
            raise ValueError("observation / paid-step misalignment")
        if row["pose_v33"] != observation["state"]["pose"]:
            raise ValueError("trace and belief refer to different poses")
        if row["next_action"] != observation["next_action"]:
            raise ValueError("trace and controller disagree on next action")
        if step < 42:
            plan = plans[step]
            if plan["step"] != step or plan["action"] != row["next_action"]:
                raise ValueError("decision is not aligned with its observation")
            if trace[step + 1]["action"] != plan["action"]:
                raise ValueError("decision does not match the next paid action")


def project_depth(depth, intrinsic, transform, shift):
    v, u = np.mgrid[0:72:4, 0:96:4]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0) & (z <= 4)
    u, v, z = u[valid], v[valid], z[valid]
    xyz = np.column_stack(((u - intrinsic[0, 2]) * z / intrinsic[0, 0],
                           (v - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
    world = xyz @ transform[:3, :3].T + transform[:3, 3] - shift
    # Actual current-frame depth only, at a declared 4-pixel stride. No GT/map reads.
    return np.round(world[:, :2], 4).tolist()


def export_packet(raw: bytes, shift, assets: Assets):
    with np.load(io.BytesIO(raw), allow_pickle=False) as packet:
        depth = packet["frame__depth_m"].copy()
        rgb = packet["frame__color_rgb"].copy()
        intrinsic = packet["frame__intrinsic"].copy()
        transform = packet["frame__world_from_camera"].copy()
        stamp = float(packet["frame__timestamp_s"])
        # Deliberately never access packet metadata, hidden labels or semantic array.
    if depth.shape != (72, 96) or rgb.shape != (72, 96, 3) or rgb.dtype != np.uint8:
        raise ValueError("unexpected native sensor shape / dtype")
    if intrinsic.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("invalid observed calibration")
    if not np.isfinite(intrinsic).all() or not np.isfinite(transform).all():
        raise ValueError("nonfinite observed calibration")
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 4)
    return {"timestamp_s": stamp, "rgb": assets.image(Image.fromarray(rgb)),
            "depth": assets.image(Image.fromarray(depth_rgb(depth))),
            "depth_points_xy": project_depth(depth, intrinsic, transform, shift),
            "valid_depth_fraction": float(valid.mean())}, transform[:2, 3] - shift[:2]


def render_mesh(vertices, triangles, color, elevation, azimuth):
    """Pillow painter rendering of EVERY saved triangle; common orthographic view."""
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("invalid saved vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.min(initial=0) < 0:
        raise ValueError("invalid saved triangles")
    if triangles.max(initial=-1) >= len(vertices):
        raise ValueError("saved triangle index out of bounds")
    elev, azim = np.deg2rad([elevation, azimuth])
    camera = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0.])
    up = np.cross(camera, right)
    basis = np.stack((right, up, camera), axis=1)
    corners = np.array([[x, y, z] for x in (-3.1, 3.1) for y in (.85, 3.7) for z in (-.1, 1.95)])
    projected = vertices @ basis
    bounds = corners @ basis
    low, high = bounds[:, :2].min(axis=0), bounds[:, :2].max(axis=0)
    width, height, aa = 600, 350, 2
    scale = min((width - 48) / (high[0] - low[0]), (height - 42) / (high[1] - low[1]))
    center = (low + high) / 2
    xy = (projected[:, :2] - center) * scale
    xy = np.column_stack((xy[:, 0] + width / 2, height / 2 - xy[:, 1])) * aa
    facets = vertices[triangles]
    normals = np.cross(facets[:, 1] - facets[:, 0], facets[:, 2] - facets[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, norms[:, None], out=np.zeros_like(normals), where=norms[:, None] > 0)
    light = np.array([.45, .6, 1.]); light /= np.linalg.norm(light)
    shade = .55 + .45 * np.abs(normals @ light)
    colors = np.clip(np.asarray(color)[None, :] * shade[:, None] + 25.5, 0, 255).astype(np.uint8)
    order = np.argsort(projected[triangles, 2].mean(axis=1), kind="stable")
    image = Image.new("RGB", (width * aa, height * aa), "#f4f7fa")
    draw = ImageDraw.Draw(image)
    for index in order:
        draw.polygon([tuple(point) for point in xy[triangles[index]]], fill=tuple(colors[index]))
    # No axes, GT geometry, inferred caps, decimation or texture substitution.
    return image.resize((width, height), Image.Resampling.LANCZOS)


def metrics_whitelist(measurement):
    metric = measurement["05cm"]
    return {"step": int(measurement["paid_actions"]),
            "C_map": float(measurement["C_map"]),
            "F1_5cm": float(metric["f1"]), "J5": float(metric["joint"]),
            "precision_5cm": float(metric["precision"]), "recall_5cm": float(metric["recall"]),
            "eligible": bool(measurement["eligible"]), "returned": bool(measurement["returned"]),
            "scope": "saved offline complete-exterior evaluation; not a planner input"}


def events_for(left, right):
    first_divergence = next((t for t in range(18, 42)
                             if left[t]["next_action"] != right[t]["next_action"]), None)
    events = [{"step": 0, "label": "初始观测"}, {"step": 18, "label": "共同前缀完成"}]
    semantic = next((f["step"] for f in right if abs(f["posterior"]["semantic_log_odds"]) > 1e-9), None)
    geometry = next((f["step"] for f in right if abs(f["posterior"]["geometry_applied_log_odds"]) > 1e-9), None)
    if semantic is not None:
        events.append({"step": semantic, "label": "首次语义权重更新"})
    if first_divergence is not None:
        events.append({"step": first_divergence, "label": "首次后继动作分歧"})
    if geometry is not None:
        events.append({"step": geometry, "label": "首次几何反馈更新"})
    events.append({"step": 42, "label": "真实终点与最终重建"})
    return sorted(events, key=lambda event: (event["step"], event["label"])), first_divergence


def build_payload(root=ROOT):
    sources, assets = Sources(root), Assets()
    config = sources.json(COHORT + "/config.json")
    public = config["parents"]["P00"]
    if public["budget"] != 42 or public["forced_prefix"] != 18:
        raise ValueError("unexpected frozen budget")
    shift = np.asarray(public["translation"], dtype=float)
    payload = {"schema": "nso-v40-saved-replay-1", "budget": 42, "prefix": 18,
               "native_sensor_shape": [72, 96], "depth_range_m": [0, 4],
               "public_roi_local": (np.asarray(public["public_bounds"]) - shift).tolist(),
               "cameras": [{"id": c[0], "label": c[1], "elevation": c[2], "azimuth": c[3]}
                           for c in CAMERAS], "runs": {}, "evaluation": {}, "conditions": []}
    for case, method, condition in CASES:
        folder = COHORT + "/" + case
        trace = sources.json(folder + "/trace.json")
        controller = sources.json(folder + "/controller.json")
        result = sources.json(folder + "/result.json")
        physical = result["physical_case"]
        if physical["method"] != method or physical["parent"] != "P00" or physical["hypothesis"] != int(condition[1]):
            raise ValueError("fixed illustrative case identity changed")
        if result["paid_actions"] != 42 or not result["returned"] or result["collisions"] != 0:
            raise ValueError("saved run status differs from declared scope")
        validate_alignment(trace, controller["history"], controller["plans"])
        frames = []
        for step, (row, history) in enumerate(zip(trace, controller["history"])):
            packet_path = folder + f"/packets/{step:03d}.npz"
            packet, camera_xy = export_packet(sources.read(packet_path), shift, assets)
            if not np.allclose(camera_xy, row["pose_v33"][:2], atol=1e-8, rtol=0):
                raise ValueError("observed camera and saved trajectory are not aligned")
            cue = row["cue"]
            frame = {"step": step, "action": row["action"], "next_action": row["next_action"],
                     "pose": [float(v) for v in row["pose_v33"]], **packet,
                     "posterior": sanitize_posterior(history),
                     "cue": {"observed_class": cue["class_id"], "pixel_count": int(cue["pixel_count"]),
                             "minimum_pixels": int(cue["minimum_pixels"]),
                             "source": "public artificial RGB colors; not a natural semantic network"},
                     "planning": sanitize_planning(controller["plans"][step] if step < 42 else None),
                     "source": {"packet_file": packet_path,
                                "packet_file_sha256": sources.records[packet_path]["sha256"],
                                "recorded_packet_digest": row["packet_sha256"]}}
            if set(frame) != set(FRAME_KEYS):
                raise ValueError("unexpected display frame key")
            frames.append(frame)
        checkpoints = {}
        payload["evaluation"][case] = {}
        for stage, step in (("prefix", 18), ("final", 42)):
            path = folder + f"/{stage}_extracted.npz"
            crop = sources.json(folder + f"/{stage}_crop.json")
            with np.load(io.BytesIO(sources.read(path)), allow_pickle=False) as mesh:
                vertices = mesh["vertices"].copy() - shift
                triangles = mesh["triangles"].copy()
            if len(triangles) != crop["kept_triangles"]:
                raise ValueError("saved crop / triangle count mismatch")
            views = {name: assets.image(render_mesh(vertices, triangles, COLORS[method], elevation, azimuth))
                     for name, _, elevation, azimuth in CAMERAS}
            checkpoints[stage] = {"step": step, "views": views, "triangles": len(triangles),
                                  "mesh_file": path, "mesh_file_sha256": sources.records[path]["sha256"]}
            payload["evaluation"][case][stage] = metrics_whitelist(result["stages"][stage]["measurement"])
        payload["runs"][case] = {"method": method, "frames": frames, "checkpoints": checkpoints,
                                 "trace_sha256": sources.records[folder + "/trace.json"]["sha256"],
                                 "controller_sha256": sources.records[folder + "/controller.json"]["sha256"]}
    for condition, cases, label in (("h0", ["case00", "case01"], "P00 / h0 · 零增益控制"),
                                     ("h1", ["case04", "case05"], "P00 / h1 · 已知正例")):
        events, first = events_for(*(payload["runs"][case]["frames"] for case in cases))
        payload["conditions"].append({"id": condition, "cases": cases, "offline_label": label,
                                      "events": events, "first_divergence_step": first})
    payload["assets"] = assets.data
    payload["scope"] = {"controlled_simulation": True, "artificial_rgb_cues": True,
                         "exact_simulated_poses": True, "shared_paid_prefix": 18,
                         "uncalibrated_two_template_weights": True,
                         "new_worlds": 0, "new_sensor_packets": 0, "new_planner_calls": 0,
                         "new_TSDF_integrations": 0, "new_quality_evaluations": 0,
                         "continuous_meshes_available": False, "GT_geometry_loaded": False}
    sources.read("scripts/build_replay_viewer_v40.py")
    sources.verify()
    return payload, sources


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSO · 语义决策同步回放</title>
<style>
:root{--ink:#172c3d;--muted:#647484;--line:#dce4eb;--paper:#fff;--bg:#eff3f7;--blue:#0072b2;--gray:#737984;--teal:#268a83;--amber:#a96a12}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC",sans-serif}button,select,input{font:inherit}button,select{border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 12px;color:var(--ink)}button{cursor:pointer}button:hover{background:#edf5fb;border-color:#8eb5cd}button:disabled{opacity:.45;cursor:not-allowed}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #eeae42;outline-offset:3px}.selected{background:#e6f2fa!important;border-color:#6da8cf!important;color:#00629a!important}button.primary{background:var(--ink);color:white;border-color:var(--ink)}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.tiny{font-size:11px}.muted{color:var(--muted)}.nav{background:#162c3e;color:#d7e2ec;padding:13px max(24px,calc((100vw - 1360px)/2));display:flex;justify-content:space-between;gap:12px;letter-spacing:.08em;font-size:11px}.nav strong{color:white;letter-spacing:.18em}main{max-width:1400px;margin:auto;padding:28px 24px 36px}.hero{display:flex;justify-content:space-between;align-items:flex-start;gap:25px}.eyebrow{font-size:11px;color:var(--blue);font-weight:700;letter-spacing:.14em}.hero h1{font-size:30px;line-height:1.3;margin:7px 0 10px;letter-spacing:-.02em}.hero p{margin:0;color:var(--muted);max-width:830px}.heroaside{text-align:right;padding-top:6px;min-width:140px}.heroaside strong{font-size:25px;line-height:1.2}.tags{display:flex;flex-wrap:wrap;gap:7px;margin:17px 0 22px}.tag{font-size:11px;border:1px solid #d9e2e8;border-radius:20px;padding:3px 10px;background:#fff;color:#506474}.tag.warn{background:#fff8eb;border-color:#f0dbb6;color:#805815}.toolbar{position:sticky;top:0;z-index:3;background:rgba(255,255,255,.97);border:1px solid var(--line);border-radius:12px;padding:14px 18px;box-shadow:0 4px 18px #19395008}.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.controls .spacer{flex:1}.control-label{font-size:11px;color:var(--muted);margin-right:3px}.timebar{display:grid;grid-template-columns:116px 1fr 170px;align-items:center;gap:15px;margin-top:13px}.timebar input{width:100%;accent-color:var(--blue)}.stepbig{font-size:23px;font-weight:700}.stepbig small{font-size:13px;font-weight:400;color:var(--muted)}.timecaption{text-align:right;font-size:11px;color:var(--muted)}.ticks{display:flex;justify-content:space-between;font-size:10px;color:#82909e;margin-top:-1px}.stage{display:flex;align-items:center;gap:10px;padding:15px 3px 12px}.stage .dot{width:7px;height:7px;border-radius:50%;background:var(--teal);flex-shrink:0}.stage strong{font-size:13px}.stage span{font-size:12px;color:var(--muted)}.compare{display:grid;grid-template-columns:1fr 1fr;gap:18px}.method{background:white;border:1px solid var(--line);border-top:4px solid var(--gray);border-radius:12px;overflow:hidden}.method.S{border-top-color:var(--blue)}.methodheader{padding:17px 20px 13px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e8edf2}.methodtitle{display:flex;align-items:center;gap:10px}.letter{width:32px;height:32px;display:grid;place-items:center;background:#edf0f3;color:var(--gray);border-radius:8px;font-size:19px;font-weight:700}.S .letter{background:#e8f3fa;color:var(--blue)}.method h2{font-size:17px;line-height:1.35;margin:0}.methodsubtitle{font-size:11px;color:var(--muted)}.next{text-align:right;font-size:11px;color:var(--muted)}.next strong{display:block;font-size:16px;color:var(--ink)}.section{padding:16px 20px;border-bottom:1px solid #e8edf2}.section:last-child{border-bottom:none}.sectionhead{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}.sectionhead h3{font-size:12px;letter-spacing:.04em;font-weight:700;margin:0}.sectionhead span{font-size:10px;color:var(--muted)}.sensors{display:grid;grid-template-columns:1fr 1fr;gap:12px}.sensorbox{border-radius:8px;background:#f3f6f8;padding:9px;text-align:center}.sensorbox img{display:block;width:192px;max-width:100%;height:auto;aspect-ratio:4/3;image-rendering:pixelated;margin:0 auto 7px}.sensorlabel{font-size:10px;color:var(--muted)}.depthkey{height:6px;border-radius:3px;background:linear-gradient(to right,#203f5e,#246f8d,#52a49e,#a6c899,#f6e07c);max-width:150px;margin:5px auto 2px}.depthticks{max-width:150px;margin:auto;display:flex;justify-content:space-between;font-size:9px;color:#72828f}.observed{margin-top:10px;font-size:11px;color:var(--muted)}.observed b{color:var(--ink);font-weight:500}.trajectory{display:block;width:100%;height:255px;background:#fafcfd;border:1px solid #e6edf2;border-radius:8px}.maplegend{display:flex;gap:13px;flex-wrap:wrap;margin-top:7px;font-size:10px;color:var(--muted)}.key{display:inline-block;width:13px;height:3px;vertical-align:middle;margin-right:5px;background:var(--gray)}.S .key.path{background:var(--blue)}.key.cloud{background:#6dafa9}.key.roi{height:8px;background:none;border:1px dashed #bdc8d2}.beliefrow{display:flex;align-items:center;gap:10px;margin-top:5px;font-size:11px}.beliefbar{height:12px;flex:1;background:#dfe8ef;border-radius:4px;overflow:hidden;display:flex}.beliefbar i{display:block;height:100%;background:#93a8b9}.beliefbar i+ i{background:#d8ad5b}.beliefnote{font-size:10px;color:var(--muted);margin-top:8px}.spark{width:100%;height:61px;display:block;margin-top:8px}.candidates{min-height:97px}.candidate{display:grid;grid-template-columns:51px 1fr 52px 75px;gap:8px;align-items:center;margin:8px 0;font-size:11px}.candidate .bar{height:7px;background:#edf1f4;border-radius:2px;overflow:hidden}.candidate .bar i{display:block;height:7px;background:#a4b2bd}.candidate.chosen{font-weight:700;color:var(--blue)}.candidate.chosen .bar i{background:var(--blue)}.candidate.invalid{color:#95a0aa}.candidate .status{font-size:10px;font-weight:400;text-align:right}.placeholder{display:grid;place-items:center;text-align:center;background:#f4f7fa;border:1px dashed #d5dfe7;border-radius:8px;color:#6a7b8a;font-size:12px;min-height:98px;padding:20px}.meshplaceholder{min-height:270px}.mesh{display:block;width:100%;height:auto;border-radius:8px;background:#f4f7fa}.meshstate{font-size:11px;color:var(--muted);margin-top:7px;min-height:36px}.checkpointbadge{font-size:10px;padding:3px 7px;border-radius:4px;background:#e9f0f5;color:#506b80}.scoregrid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}.score{background:#fff8ec;border:1px solid #ecddc3;border-radius:7px;padding:9px 10px}.score small{display:block;font-size:10px;color:#8b734d}.score strong{font-size:21px;font-weight:600;color:#6c522d}.offlinenote{font-size:10px;color:#8b734d;margin-top:6px}.foot{background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px 20px;margin-top:19px;font-size:12px;color:var(--muted)}.foot strong{color:var(--ink)}details{margin-top:12px}summary{cursor:pointer;color:#546d82;font-size:11px}.sourcegrid{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:10px}.source{font-size:10px;overflow-wrap:anywhere;line-height:1.7;background:#f6f8fa;border-radius:6px;padding:10px}.source b{font-family:inherit;color:var(--ink)}.runtime{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-size:10px;margin-top:12px}.viewtools{display:flex;gap:5px}.viewtools button{font-size:10px;padding:4px 7px}.evaluationlabel{font-size:11px;color:#856321;display:flex;align-items:center;gap:5px}noscript{display:block;padding:30px;background:#fff1db}
@media(max-width:850px){main{padding:20px 13px}.compare{grid-template-columns:1fr}.hero h1{font-size:25px}.heroaside{display:none}.toolbar{padding:11px;position:relative}.timebar{grid-template-columns:85px 1fr}.timecaption{grid-column:1/-1;text-align:left}.controls .spacer{display:none}.sourcegrid{grid-template-columns:1fr}.section{padding:14px}.nav{padding:12px 15px}.nav span{letter-spacing:0}.stage{align-items:flex-start}.stage span{display:block}}
</style></head><body>
<div class="nav"><strong>NSO / REPLAY LAB</strong><span>V40 P0 · READ-ONLY EVIDENCE</span></div>
<main><div class="hero"><div><div class="eyebrow">OBSERVATION → BELIEF → ACTION</div><h1>从语义观测到补看决策</h1><p>并排查看几何基线 G 与语义条件策略 S：相同付费动作时刻，实际看见什么、怎样判断、接着往哪里走。</p></div><div class="heroaside"><strong>4 <span class="muted">/</span> 172</strong><div class="tiny muted">保存轨迹 / 输入帧<br>新实验 0 · 新重建 0</div></div></div>
<div class="tags"><span class="tag warn">Controlled simulation · 受控虚拟实验</span><span class="tag warn">Artificial RGB cues · 人工类别提示</span><span class="tag">Exact simulated poses · 精确模拟位姿</span><span class="tag">18-step paid prefix · 共同付费前缀</span><span class="tag">96 × 72 原生输入 · 无 GPU / 无联网依赖</span></div>
<div class="toolbar"><div class="controls"><label for="condition" class="control-label">离线条件</label><select id="condition" aria-label="选择保存条件"></select><button id="play" class="primary">▶ 播放</button><button id="start">回到 0</button><button id="divergence">首处分歧</button><select id="event" aria-label="跳转关键事件"></select><div class="spacer"></div><select id="speed" aria-label="回放速度"><option value="1">1 步 / 秒</option><option value="2" selected>2 步 / 秒</option><option value="4">4 步 / 秒</option></select><label class="evaluationlabel"><input type="checkbox" id="evaluation">离线评分</label></div>
<div class="timebar"><div class="stepbig"><span id="step">18</span><small> / 42 动作</small></div><div><input id="slider" type="range" min="0" max="42" step="1" value="18" aria-label="已完成付费动作数"><div class="ticks"><span>0 · 初始帧</span><span>18 · 共同前缀结束</span><span>42 · 返回终点</span></div></div><div class="timecaption">以付费动作同步<br>播放速度 ≠ 运行耗时</div></div></div>
<div class="stage" aria-live="polite"><i class="dot"></i><div><strong id="stageTitle"></strong> <span id="stageDetail"></span></div></div>
<div id="compare" class="compare"></div>
<div class="foot"><strong>这是一份保存数据的同步回放。</strong> 轨迹、传感与控制器输出来自 V39；结构权重是未校准的全局两模板权重，不是设施实例概率。候选条形表示规划代理分数，不能解释为实际 F1 增量。在线轨迹图只含已执行路径和当前帧深度投影，没有 GT 场景几何。<br>三维视图只显示已到达的真实保存快照：第 18 步与第 42 步；两者之间没有逐步 TSDF，也没有插值补面。方法颜色与光照不表示误差。离线评分使用保存的完整外部垂直参考面结果，公开 ROI 外误差未计入，评分从不参与回放决策。h0 是零增益控制，h1 是此前公布的正例；两个条件来自同一父布局，不构成独立泛化验证。
<details><summary>核对当前帧与保存来源 / SHA256</summary><div id="sources" class="sourcegrid"></div></details><div class="runtime"><span>CPU 正交投影 · 3 个全方法共享相机 · 全部保存三角形 · 无纹理替代 / 配准 / 补洞</span><span id="assetInfo"></span></div></div>
</main><noscript>请启用 JavaScript 查看本地回放。本页面完全离线运行，不调用外部资源。</noscript>
<script id="replay-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('replay-data').textContent);
const $=id=>document.getElementById(id), act={forward:'前进',left:'左转',right:'右转',null:'完成'};
let current=DATA.conditions.find(x=>x.id==='h1'), step=18, camera='oblique', timer=null;
const fmt=v=>Number(v).toFixed(3), action=v=>v==null?'完成':(act[v]||v);
function checkpointAt(t){if(!Number.isInteger(t)||t<0||t>42)throw new Error('invalid paid step');return t===42?'final':t>=18?'prefix':null;}
function imageAsset(id){return DATA.assets[id];}
function card(method){return `<article class="method ${method}"><div class="methodheader"><div class="methodtitle"><div class="letter">${method}</div><div><h2>${method==='G'?'主动几何基线':'语义条件策略'}</h2><div class="methodsubtitle">${method==='G'?'使用几何反馈 · 不以类别条件选向':'类别条件 + 几何反馈 · 同预算与安全约束'}</div></div></div><div class="next">决策 ${step} → 动作 ${step+1}<strong id="next-${method}"></strong></div></div>
<div class="section"><div class="sectionhead"><h3>01 / 实际付费观测</h3><span>原生 96 × 72 · 仅显示放大</span></div><div class="sensors"><div class="sensorbox"><img id="rgb-${method}" alt="${method} 当前保存 RGB 帧"><div class="sensorlabel">RGB · 人工色码可见</div></div><div class="sensorbox"><img id="depth-${method}" alt="${method} 当前保存深度图"><div class="sensorlabel">轴向深度 · 无效/缺失为灰色</div><div class="depthkey"></div><div class="depthticks"><span>0</span><span>2</span><span>4 m</span></div></div></div><div id="observed-${method}" class="observed"></div></div>
<div class="section"><div class="sectionhead"><h3>02 / 已执行轨迹与当前观察</h3><span>公共局部坐标 · m</span></div><canvas id="map-${method}" class="trajectory" width="1000" height="510" aria-label="${method} 已执行轨迹"></canvas><div class="maplegend"><span><i class="key path"></i>已执行路径</span><span><i class="key cloud"></i>当前帧深度投影</span><span><i class="key roi"></i>共享公开任务范围</span></div></div>
<div class="section"><div class="sectionhead"><h3>03 / 结构判断</h3><span>全局两模板 · 未校准权重</span></div><div class="beliefrow"><span id="w0-${method}" class="mono"></span><div class="beliefbar"><i id="bar0-${method}"></i><i id="bar1-${method}"></i></div><span id="w1-${method}" class="mono"></span></div><canvas id="spark-${method}" class="spark" width="1000" height="122" aria-label="${method} 截至当前步的模板一权重"></canvas><div id="beliefnote-${method}" class="beliefnote"></div></div>
<div class="section"><div class="sectionhead"><h3>04 / 下一动作的候选</h3><span>预测代理 ≠ 实际重建收益</span></div><div id="candidates-${method}" class="candidates"></div></div>
<div class="section"><div class="sectionhead"><h3>05 / 真实保存的 TSDF</h3><div class="viewtools">${DATA.cameras.map(c=>`<button data-camera="${c.id}" class="${c.id===camera?'selected':''}">${c.label}</button>`).join('')}</div></div><div id="meshbox-${method}"></div><div id="meshstate-${method}" class="meshstate"></div><div id="score-${method}"></div></div></article>`;}
function configure(){
 $('compare').innerHTML=current.cases.map(c=>card(DATA.runs[c].method)).join('');
 $('event').innerHTML='<option value="">关键事件…</option>'+current.events.map(e=>`<option value="${e.step}">${e.step} · ${e.label}</option>`).join('');
 $('divergence').disabled=current.first_divergence_step===null;
 $('divergence').textContent=current.first_divergence_step===null?'无动作分歧':'首处分歧';
 document.querySelectorAll('[data-camera]').forEach(button=>button.onclick=()=>{camera=button.dataset.camera;document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('selected',b.dataset.camera===camera));render();});
 render();
}
function drawMap(canvas,run){
 const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);
 const b={x0:-3.65,x1:3.65,y0:-.6,y1:4.65},s=Math.min((W-110)/(b.x1-b.x0),(H-62)/(b.y1-b.y0)),ox=(W-(b.x1-b.x0)*s)/2,oy=(H-(b.y1-b.y0)*s)/2;
 const p=(x,y)=>[ox+(x-b.x0)*s,H-oy-(y-b.y0)*s];
 ctx.font='18px sans-serif';ctx.fillStyle='#8897a5';ctx.strokeStyle='#e8edf1';ctx.lineWidth=1.5;
 for(let x=-3;x<=3;x++){let a=p(x,0),z=p(x,4);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...z);ctx.stroke();ctx.fillText(String(x),a[0]-5,H-8);}
 for(let y=0;y<=4;y++){let a=p(-3.45,y),z=p(3.45,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...z);ctx.stroke();ctx.fillText(String(y),a[0]-35,a[1]+6);}
 const lo=DATA.public_roi_local[0],hi=DATA.public_roi_local[1],a=p(lo[0],hi[1]),z=p(hi[0],lo[1]);ctx.setLineDash([8,7]);ctx.strokeStyle='#bcc9d4';ctx.strokeRect(a[0],a[1],z[0]-a[0],z[1]-a[1]);ctx.setLineDash([]);
 const f=run.frames[step];ctx.save();ctx.beginPath();ctx.rect(ox,oy,W-2*ox,H-2*oy);ctx.clip();ctx.fillStyle='#70aaa78c';
 f.depth_points_xy.forEach(q=>{let v=p(q[0],q[1]);ctx.fillRect(v[0]-2,v[1]-2,4,4);});
 ctx.strokeStyle=run.method==='S'?'#0072b2':'#737984';ctx.lineWidth=5;ctx.lineCap='round';ctx.lineJoin='round';
 ctx.beginPath();run.frames.slice(0,step+1).forEach((r,i)=>{let v=p(r.pose[0],r.pose[1]);i?ctx.lineTo(...v):ctx.moveTo(...v);});ctx.stroke();
 const pos=p(f.pose[0],f.pose[1]),heading=f.pose[2]*Math.PI/2,dx=Math.sin(heading),dy=Math.cos(heading);
 ctx.fillStyle=run.method==='S'?'#0072b2':'#737984';ctx.beginPath();ctx.arc(...pos,8,0,2*Math.PI);ctx.fill();
 const tip=p(f.pose[0]+dx*.5,f.pose[1]+dy*.5),l=p(f.pose[0]+dx*.18-dy*.15,f.pose[1]+dy*.18+dx*.15),r=p(f.pose[0]+dx*.18+dy*.15,f.pose[1]+dy*.18-dx*.15);ctx.beginPath();ctx.moveTo(...tip);ctx.lineTo(...l);ctx.lineTo(...r);ctx.closePath();ctx.fill();ctx.restore();
 const anchor=p(0,0);ctx.fillStyle='#23384b';ctx.fillRect(anchor[0]-5,anchor[1]-5,10,10);ctx.font='17px sans-serif';ctx.fillText('Anchor',anchor[0]+12,anchor[1]+25);ctx.fillText('x / m',W-66,H-8);ctx.fillText('y / m',14,24);
}
function drawSpark(canvas,run){
 const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);const px=t=>30+t*(W-50)/42,py=v=>H-24-v*(H-38);
 ctx.font='17px sans-serif';ctx.fillStyle='#8c9aa6';ctx.fillText('w(h1)',30,17);ctx.strokeStyle='#e5eaf0';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(px(0),py(.5));ctx.lineTo(px(42),py(.5));ctx.stroke();ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(px(18),15);ctx.lineTo(px(18),H-20);ctx.stroke();ctx.setLineDash([]);
 ctx.strokeStyle=run.method==='S'?'#0072b2':'#737984';ctx.lineWidth=3;ctx.beginPath();run.frames.slice(0,step+1).forEach((f,i)=>{let x=px(i),y=py(f.posterior.weights[1]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();const f=run.frames[step];ctx.beginPath();ctx.arc(px(step),py(f.posterior.weights[1]),4,0,2*Math.PI);ctx.fillStyle=ctx.strokeStyle;ctx.fill();ctx.fillStyle='#8c9aa6';ctx.fillText('0',px(0),H-3);ctx.fillText('18',px(18)-10,H-3);ctx.fillText('42',px(42)-16,H-3);
}
function render(){
 $('slider').value=step;$('step').textContent=step;
 const rows=current.cases.map(c=>DATA.runs[c].frames[step]), different=rows[0].next_action!==rows[1].next_action;
 $('stageTitle').textContent=step<18?'共同付费前缀':step===42?'已到达保存终点':different?'同一付费时刻 · 后继动作不同':'在线决策 · 后继动作相同';
 $('stageDetail').textContent=step<18?'前 18 个动作预先固定，信念不参与动作选择。':step===42?'两方法均真实返航；可开启离线评分查看保存的终点结果。':`决策 ${step}：G ${action(rows[0].next_action)} / S ${action(rows[1].next_action)}。这是已记录决策，不是重新运行。`;
 const sourceBoxes=[];
 current.cases.forEach(caseId=>{
  const run=DATA.runs[caseId],m=run.method,f=run.frames[step],weights=f.posterior.weights,ck=checkpointAt(step);
  $('next-'+m).textContent=step===42?'已返航':action(f.next_action);
  $('next-'+m).parentElement.firstChild.textContent=step===42?'42 个付费动作完成':`决策 ${step} → 动作 ${step+1}`;
  $('rgb-'+m).src=imageAsset(f.rgb);$('depth-'+m).src=imageAsset(f.depth);
  $('observed-'+m).innerHTML=`刚执行 <b>${step===0?'初始观测':action(f.action)}</b> · 类别提示 <b>${f.cue.observed_class==null?'未检测到':f.cue.observed_class===2?'A':'B'}</b>（${f.cue.pixel_count} 像素） · 有效深度 <b>${(f.valid_depth_fraction*100).toFixed(1)}%</b>`;
  drawMap($('map-'+m),run);drawSpark($('spark-'+m),run);
  $('w0-'+m).textContent='h0 '+fmt(weights[0]);$('w1-'+m).textContent='h1 '+fmt(weights[1]);$('bar0-'+m).style.width=(weights[0]*100)+'%';$('bar1-'+m).style.width=(weights[1]*100)+'%';
  $('beliefnote-'+m).textContent=`语义 log-odds ${fmt(f.posterior.semantic_log_odds)} · 几何 log-odds ${fmt(f.posterior.geometry_log_odds)} · 本次几何增量 ${fmt(f.posterior.geometry_applied_log_odds)}`;
  const candidates=f.planning.candidates;
  $('candidates-'+m).innerHTML=candidates.length?candidates.map(c=>`<div class="candidate ${c.action===f.next_action?'chosen':''} ${!c.return_feasible?'invalid':''}"><span>${action(c.action)}${c.action===f.next_action?' ✓':''}</span><span class="bar"><i style="width:${c.expected_proxy===null?0:Math.max(0,Math.min(100,c.expected_proxy*100))}%"></i></span><span class="mono">${c.expected_proxy===null?'—':fmt(c.expected_proxy)}</span><span class="status">${!c.return_feasible?'不可返航':c.forecast_information?'预计新信息':'可返航'}</span></div>`).join(''):`<div class="placeholder">${step<18?'共同前缀：下一动作预先固定<br>此时没有基于信念的候选排序。':'预算内真实返航完成<br>没有后续规划动作。'}</div>`;
  if(ck){const checkpoint=run.checkpoints[ck];$('meshbox-'+m).innerHTML=`<img class="mesh" src="${imageAsset(checkpoint.views[camera])}" alt="${m} 第${checkpoint.step}步保存 TSDF 网格">`;$('meshstate-'+m).innerHTML=`<span class="checkpointbadge">保存快照 t = ${checkpoint.step}</span> · ${checkpoint.triangles.toLocaleString()} 个真实三角形<br>${step===42?'最终已观测表面；没有补洞或真值叠加。':step===18?'已到达首个保存快照。':'当前步 '+step+' 没有保存网格；仍显示 t=18，未作重积分。'}`;
  }else{$('meshbox-'+m).innerHTML='<div class="placeholder meshplaceholder">此时没有保存的 TSDF 快照<br>到第 18 步才揭示首个真实网格。</div>';$('meshstate-'+m).textContent='未提前展示前缀或最终网格。';}
  if($('evaluation').checked&&ck){const v=DATA.evaluation[caseId][ck];$('score-'+m).innerHTML=`<div class="scoregrid"><div class="score"><small>离线 C_map · t=${v.step}</small><strong>${fmt(v.C_map)}</strong></div><div class="score"><small>离线 F1@5cm</small><strong>${fmt(v.F1_5cm)}</strong></div><div class="score"><small>离线 J5 = C × F1</small><strong>${fmt(v.J5)}</strong></div></div><div class="offlinenote">仅抄录已保存评价，不参与规划；${v.eligible?'满足最终资格':'前缀尚未满足最终覆盖资格'}。${step!==v.step?'当前步不含新的评分。':''}</div>`;}else{$('score-'+m).innerHTML='';}
  sourceBoxes.push(`<div class="source mono"><b>${m} · paid step ${step}</b><br>${f.source.packet_file}<br>file SHA256: ${f.source.packet_file_sha256}<br><b>记录的逻辑包摘要（不等于 NPZ 文件摘要）</b><br>${f.source.recorded_packet_digest}<br>trace SHA256: ${run.trace_sha256}<br>controller SHA256: ${run.controller_sha256}${ck?'<br><b>当前显示的保存网格</b><br>'+run.checkpoints[ck].mesh_file+'<br>'+run.checkpoints[ck].mesh_file_sha256:''}</div>`);
 });
 $('sources').innerHTML=sourceBoxes.join('');
 $('assetInfo').textContent=`${Object.keys(DATA.assets).length} 个去重图像资产 · 单 HTML · 可断网运行`;
}
function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='▶ 播放';}
function play(){if(timer!==null){stop();return;}if(step===42)step=0;$('play').textContent='Ⅱ 暂停';timer=setInterval(()=>{if(step>=42){stop();return;}step++;render();if(step===42)stop();},1000/Number($('speed').value));render();}
function seek(value){stop();step=Number(value);render();}
$('condition').innerHTML=DATA.conditions.map(c=>`<option value="${c.id}">${c.offline_label}</option>`).join('');$('condition').value=current.id;
$('condition').onchange=()=>{stop();current=DATA.conditions.find(c=>c.id===$('condition').value);configure();};
$('play').onclick=play;$('start').onclick=()=>seek(0);$('divergence').onclick=()=>{if(current.first_divergence_step!==null)seek(current.first_divergence_step);};$('slider').oninput=event=>seek(event.target.value);$('event').onchange=event=>{if(event.target.value!=='')seek(event.target.value);};$('evaluation').onchange=render;$('speed').onchange=()=>{if(timer!==null){stop();play();}};
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName))return;if(event.key==='ArrowRight'){event.preventDefault();seek(Math.min(42,step+1));}if(event.key==='ArrowLeft'){event.preventDefault();seek(Math.max(0,step-1));}if(event.code==='Space'){event.preventDefault();play();}});
configure();window.NSO_REPLAY={checkpointAt,getState:()=>({condition:current.id,step,camera,playing:timer!==null}),seek};
</script></body></html>'''


def render_html(payload):
    encoded = json_bytes(payload).decode("utf-8").replace("<", "\\u003c")
    return HTML.replace("__PAYLOAD__", encoded).encode("utf-8")


def atomic_write(path: Path, data: bytes):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".v40-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("write output inside the project")
    if shutil.disk_usage(ROOT).free < RESERVE + MAX_BYTES:
        raise RuntimeError("64 MiB reserve plus 20 MB output budget required")
    if out.exists() and any(p.name not in ("index.html", "manifest.json") for p in out.iterdir()):
        raise RuntimeError("unexpected files in owned output folder; will not remove them")
    payload, sources = build_payload()
    html = render_html(payload)
    manifest = {"schema": "nso-v40-saved-replay-manifest-1", "date": "2026-09-20",
                "input_cohort": COHORT, "fixed_cases": [case for case, _, _ in CASES],
                "parent_count": 1, "conditions": ["h0", "h1"], "saved_runs": 4,
                "saved_sensor_frames": 172, "displayed_mesh_checkpoints": 8,
                "rendered_camera_views": 24, "image_assets_deduplicated": len(payload["assets"]),
                "selection": "Previously published P00/h1 positive example plus P00/h0 null control; not randomly sampled independent scenes",
                "scope": payload["scope"], "sources": sources.records,
                "outputs": {"index.html": {"sha256": digest(html), "bytes": len(html)}},
                "limits": {"maximum_total_output_bytes": MAX_BYTES, "minimum_free_disk_bytes": RESERVE},
                "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "Pillow": PIL_VERSION},
                "presentation": {"sensor_native_pixels": [96, 72], "depth_range_m": [0, 4],
                                  "invalid_depth_rgb": [215, 220, 226],
                                  "camera_projection": "shared orthographic",
                                  "camera_presets": payload["cameras"],
                                  "mesh_local_translation_subtracted": [3.5, .5, 0],
                                  "mesh_common_bounds": [[-3.1, .85, -.1], [3.1, 3.7, 1.95]],
                                  "all_saved_triangles_drawn": True, "mesh_decimation": False,
                                  "hole_filling": False, "alignment": False,
                                  "current_depth_projection_stride_pixels": 4,
                                  "mesh_visibility_rule": "none before 18; saved prefix at 18..41; saved final at 42",
                                  "metrics_visibility_rule": "opt-in offline layer, only the reached saved checkpoint",
                                  "frame_whitelist": list(FRAME_KEYS),
                                  "never_exported": ["packet.metadata", "frame__semantic", "actual hidden structure ID as online input", "evaluation_floor", "GT geometry", "controller state masks/potential tables", "future metric as candidate gain"]},
                "events": {c["id"]: c["events"] for c in payload["conditions"]},
                "input_files_verified_unchanged": True,
                "command": "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv-3d/bin/python scripts/build_replay_viewer_v40.py"}
    manifest_raw = json_bytes(manifest)
    total = len(html) + len(manifest_raw)
    if total > MAX_BYTES:
        raise RuntimeError(f"bounded artifact exceeds {MAX_BYTES} bytes: {total}")
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(out / "index.html", html)
    atomic_write(out / "manifest.json", manifest_raw)
    if shutil.disk_usage(ROOT).free < RESERVE:
        raise RuntimeError("disk reserve violated")
    print(json.dumps({"output": str(out / "index.html"), "total_bytes": total,
                      "sources_verified": len(sources.records), "html_sha256": digest(html),
                      "new_experiments": 0, "new_integrations": 0}, indent=2))


if __name__ == "__main__":
    main()
