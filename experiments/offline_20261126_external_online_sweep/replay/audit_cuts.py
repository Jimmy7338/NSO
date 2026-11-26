"""Audit all currently observed centre-to-centre cuts from the recovered component."""

from collections import Counter
import json
import sys

import numpy as np
from scipy.ndimage import label

from reconstruct_history import HERE, load_history, read


ROOT = HERE.parent
CROSS = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRECT_PAID_38 = ((45, 25), (46, 25))


def rejection_cuts(sweep, graph, belief, position, proposals):
    regions, _ = graph.regions()
    centre = regions[position]
    current = regions == centre
    uninflated, _ = label(belief == 0, structure=CROSS)
    potential_connected = uninflated == uninflated[position]
    target_regions = {regions[tuple(row["pose"][:2])] for row in proposals if row["status"] != "admitted"}
    cuts = []
    for a in np.argwhere(current):
        p = tuple(int(x) for x in a)
        for h, (dr, dc) in enumerate(DIRECTIONS):
            q = (p[0] + dr, p[1] + dc)
            valid = 0 <= q[0] < belief.shape[0] and 0 <= q[1] < belief.shape[1]
            if not valid or current[q] or not potential_connected[q]:
                continue
            assert belief[p] == belief[q] == 0 and not graph.edges[p + (h,)]
            proof = sweep.raw.certify(sweep.xy(p), sweep.xy(q), sweep.radius)
            cuts.append({"origin": p, "destination": q, "destination_region": int(regions[q]),
                         "destination_is_proposal_region": regions[q] in target_regions, "proof": proof})
    return cuts


def run():
    history = load_history(38)
    receipt = read(HERE / "FIXED_REPLAY_V1.json")
    safe_checks = {row["paid"]: row for row in receipt["checks"]}
    sys.path.insert(0, str(ROOT))
    from observed_sweep import SweepEvidence, inflated
    from nso.observed_gain_calibration_v10_1 import ObservedGainCalibration
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10

    original = history["mapper"]
    config = original.config
    model = ObservedRuntimeMapperV10(original.shape, config, config.truncation_m)
    gain = ObservedGainCalibration(original.shape, consume_predicted_attempts=False)
    sweep = SweepEvidence(original.shape, config.resolution_m, config.robot_radius_m)
    results = []
    for tick, packet in enumerate(history["packets"]):
        model.update(packet.frame, packet.scan)
        sweep.observe(packet.frame, packet.scan)
        if tick not in safe_checks:
            continue
        check = safe_checks[tick]["adapted_observed_sweep"]
        graph = sweep.build(model.belief)
        cuts = rejection_cuts(sweep, graph, model.belief, packet.position, check["fixed_pool_native_admission"])
        options = []
        legacy = ~inflated(model.belief != 0, config.robot_radius_m / config.resolution_m)
        legacy_labels, _ = label(legacy, structure=CROSS)
        legacy_component = legacy_labels == legacy_labels[packet.position]
        previous = {tuple(row["pose"]) for row in safe_checks[tick]["baseline_inflated_grid"]["native_routes"]}
        for row in check["native_routes"] + check["certified_regional_routes"]:
            if tuple(row["pose"]) in previous:
                continue
            path = [tuple(state) for state in row["outbound_states"]]
            radar, camera = gain.route_masks(model, path[1:], np.zeros(original.shape, dtype=bool))
            witnesses = [
                {"start": a[:2], "end": b[:2], "proof": graph.added[a]}
                for a, b in zip(path, path[1:]) if a[:2] != b[:2] and a in graph.added
            ]
            options.append({
                "pose": row["pose"], "outbound": row["outbound_cost"], "return": row["return_cost"],
                "centres_beyond_original_component": sorted({pose[:2] for pose in path if not legacy_component[pose[:2]]}),
                "forecast_radar_cells": int(radar.sum()), "forecast_camera_cells": int(camera.sum()),
                "outbound_exception_witnesses": witnesses,
            })
        case = {
            "paid": tick, "cut_reason_counts": dict(Counter(row["proof"]["reason"] for row in cuts)),
            "cut_edges": cuts, "new_admitted_options": options,
        }
        if tick == 38:
            p, q = DIRECT_PAID_38
            case["old_direct_bottleneck"] = sweep.raw.certify(sweep.xy(p), sweep.xy(q), sweep.radius)
        results.append(case)
    return {"scope": "Sensor-only diagnostic, finite registered prefix p0..38", "checks": results}


if __name__ == "__main__":
    data = run()
    (HERE / "CUT_EVIDENCE_V1.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    for case in data["checks"]:
        print(f"p{case['paid']} remaining cuts: {case['cut_reason_counts']}")
        for row in case["new_admitted_options"]:
            print(row["pose"], row["outbound"], row["return"], row["forecast_radar_cells"], row["forecast_camera_cells"])
