"""Check independent native evidence for every deliberate ground observation."""

from collections import Counter, defaultdict
import math

import numpy as np
from scipy.sparse.csgraph import dijkstra

from evaluate import DIRECTIONS, require


PREFIX = "ground_reacquisition_"
STAGES = ("dispatch", "view", "replan", "cross", "end")


def native_state(run, initial, paid):
    if paid == 0:
        return {"position": initial["anchor"][:2], "heading": initial["anchor"][2], **run["initial_raw"]}
    return run["trace"][paid-1]


def at_pose(row):
    return [*row["position"], row["heading"]]


def stage_at(attempt, name):
    return attempt["receipts"].get(name)


def inspect_run(run, initial, arm):
    total = run["actual_paid"]
    approvals = {}
    groups = defaultdict(dict)
    support_groups = defaultdict(list)
    rejections, reasons = [], Counter()
    diagnostics = Counter()
    for item in run["events"]:
        event = item["event"]
        if event == "paid_action_authorized":
            paid = item["next_action_id"]
            require(paid not in approvals, f"Duplicate native authorization at paid {paid}")
            approvals[paid] = item
        if not event.startswith(PREFIX):
            continue
        require(arm == "B", "Control arm executed the observation-only module")
        stage = event.removeprefix(PREFIX)
        require(0 <= item["paid"] <= total, "Ground receipt clock outside actual execution")
        if stage == "rejected":
            rejections.append(item)
            reasons[item["reason"]] += 1
        elif stage in STAGES:
            seq = item["inspection_id"]
            require(seq > 0 and stage not in groups[seq], f"Repeated {stage} receipt under inspection {seq}")
            groups[seq][stage] = item
        elif stage == "support":
            support_groups[item["inspection_id"]].append(item)
        else:
            diagnostics[stage] += 1

    attempts = []
    counts = Counter({name: 0 for name in (
        "dispatched", "reached_viewpoint", "fresh_viewpoint_packets", "positive_ground_additions",
        "full_view_certificates", "guarded_actual_crossings", "unfinished_at_native_stop", "positive_support_packets",
        "paid_to_inspection_view", "paid_to_cross_after_view", "collisions_on_inspection_actions",
    )})
    for seq, group in sorted(groups.items()):
        dispatch, view, replan, cross, end = (group.get(stage) for stage in STAGES)
        require(dispatch is not None, f"Ground action occurred without dispatch {seq}")
        ordered = [group[name]["paid"] for name in STAGES if name in group]
        require(ordered == sorted(ordered), f"Ground receipt clocks out of order {seq}")
        start = dispatch["paid"]
        actual_source = native_state(run, initial, start)
        edge, heading = dispatch["edge"], dispatch["heading"]
        continued_inspection = bool(attempts and attempts[-1]["edge"] == edge and attempts[-1]["last_paid"] == start
                                    and attempts[-1]["outcome"] in ("partial_evidence_retry", "no_new_evidence_retry"))
        require(at_pose(actual_source) == dispatch["pose"] and dispatch["packet_sha256"] == actual_source["packet_sha256"]
                and (edge[0] == dispatch["pose"][:2] or continued_inspection)
                and edge[1] == [a + b for a, b in zip(edge[0], DIRECTIONS[heading])]
                and all(run["beliefs"][start][tuple(cell)] == 0 for cell in edge),
                "Dispatched place, packet, or measured-free edge differs from native trace")
        blocked = dispatch["prior_certificate"]
        require(blocked["allowed"] is False and blocked["reason"] == "missing_positive_ground_support"
                and blocked["unsupported_xy_probes"] == len(dispatch["missing_xy_m"]) > 0,
                "Observer attempted an edge that was not missing positive ground evidence")
        offered = dispatch["candidates"]
        require(offered and offered[0]["pose"] == dispatch["view"] == dispatch["ordered_poses"][-1]
                and dispatch["ordered_poses"][0] == dispatch["pose"]
                and len(dispatch["ordered_poses"]) == len(dispatch["ordered_actions"])+1
                and offered[0]["outward_actions"] == len(dispatch["ordered_actions"])
                and offered[0]["return_actions"] == dispatch["return_actions"]
                and start + offered[0]["outward_actions"] + offered[0]["complete_reserve"] <= 192,
                "Observation did not select a recorded reachable viewpoint with budget reserve")
        visited = [native_state(run, initial, tick) for tick in range(start+1)]
        require(all(at_pose(sample) != dispatch["view"] for sample in visited),
                "Dispatched inspection view was already measured")
        counts["dispatched"] += 1
        end_paid = end["paid"] if end else total
        require(end_paid <= total and (end is None or end["action_cost"] == end_paid-start),
                "Inspection did not preserve charged action count")
        if not end:
            counts["unfinished_at_native_stop"] += 1
        actual_by_stage = {stage: [tick for tick in range(start+1, end_paid+1)
                           if approvals.get(tick, {}).get("ground_reacquisition_stage") == stage]
                           for stage in ("view", "cross")}
        for stage, paid_steps in actual_by_stage.items():
            plan = dispatch if stage == "view" else replan
            if not paid_steps:
                continue
            require(plan is not None, "Charged action lacks a corresponding ground route")
            first = plan["paid"]+1
            require(paid_steps == list(range(first, first+len(paid_steps))) and len(paid_steps) <= len(plan["ordered_actions"]),
                    "Native ground actions do not form a contiguous prefix of the chosen route")
            for index, tick in enumerate(paid_steps):
                action = native_state(run, initial, tick)
                executed = approvals[tick]["assessment"]
                require(action["action"] == plan["ordered_actions"][index]
                        and executed["next_pose"] == plan["ordered_poses"][index+1] and executed["allowed"]
                        and action["guard_allowed"] and (action["collision"] or at_pose(action) == executed["next_pose"]),
                        "Executed ground route differs from certified command receipt")
            count_field = "paid_to_inspection_view" if stage == "view" else "paid_to_cross_after_view"
            counts[count_field] += len(paid_steps)
            counts["collisions_on_inspection_actions"] += sum(run["trace"][tick-1]["collision"] for tick in paid_steps)
        novel_sensor_values = None
        if view:
            tick = view["paid"]
            native_view = native_state(run, initial, tick)
            earlier_views = [native_state(run, initial, p) for p in range(tick)]
            fresh = not any(at_pose(old) == at_pose(native_view)
                            and old["scan_ranges_sha256"] == native_view["scan_ranges_sha256"]
                            and old["depth_values_sha256"] == native_view["depth_values_sha256"] for old in earlier_views)
            novel_sensor_values = any(all(native_view[key] != old[key] for old in earlier_views)
                                      for key in ("depth_values_sha256", "scan_ranges_sha256"))
            measured_gain = max(0, view["unsupported_before"] - view["unsupported_after"])
            require(view["pose"] == dispatch["view"] == at_pose(native_view)
                    and not native_view["collision"] and tick == start + len(dispatch["ordered_actions"])
                    and tick in actual_by_stage["view"] and view["fresh_packet"] == fresh
                    and all(view[key] == native_view[key] for key in ("scan_ranges_sha256", "depth_values_sha256", "packet_sha256"))
                    and view["unsupported_before"] == len(dispatch["missing_xy_m"])
                    and view["unsupported_after"] == len(view["remaining_missing_xy_m"])
                    and view["newly_supported"] == measured_gain
                    and view["prior_certificate"] == blocked,
                    f"Actual view, packet provenance or measured support counts disagree {seq}")
            counts["reached_viewpoint"] += 1
            counts["fresh_viewpoint_packets"] += int(fresh)
            counts["positive_ground_additions"] += int(measured_gain > 0)
            counts["full_view_certificates"] += int(view["certificate"]["allowed"])
        witnesses = support_groups[seq]
        last_missing = len(dispatch["missing_xy_m"])
        for support in witnesses:
            tick = support["paid"]
            native = native_state(run, initial, tick)
            remaining = len(support["remaining_missing_xy_m"])
            require(tick in actual_by_stage["view"] and support["pose"] == at_pose(native)
                    and support["packet_sha256"] == native["packet_sha256"] and support["edge"] == edge
                    and 0 < support["newly_supported"] == last_missing - remaining,
                    "Individual support witness lacks its unique native sensor packet or probe update")
            last_missing = remaining
        if view:
            require(last_missing == view["unsupported_after"], "The support packet history does not sum to the final view")
        counts["positive_support_packets"] += len(witnesses)
        if replan:
            require(view and replan["paid"] == view["paid"] and replan["certificate"] == view["certificate"]
                    and view["certificate"]["allowed"] and view["fresh_packet"] and view["newly_supported"] > 0
                    and replan["pose"] == view["pose"] and replan["edge"] == edge
                    and replan["ordered_poses"][0] == view["pose"]
                    and replan["ordered_poses"][-1] == [*edge[1], heading]
                    and replan["paid"] + len(replan["ordered_actions"]) + replan["return_actions"] <= 192,
                    "Crossing route was not replanned with a novel fully certified observation and reserve")
        if cross:
            tick = cross["paid"]
            require(replan and view and cross["measured_view_paid"] == view["paid"]
                    and tick == replan["paid"] + len(replan["ordered_actions"]) and tick in actual_by_stage["cross"]
                    and at_pose(native_state(run, initial, tick-1)) == [*edge[0], heading]
                    and cross["pose"] == at_pose(native_state(run, initial, tick)) == [*edge[1], heading]
                    and cross["packet_sha256"] == native_state(run, initial, tick)["packet_sha256"]
                    and run["trace"][tick-1]["action"] == "forward" and not run["trace"][tick-1]["collision"]
                    and cross["native_assessment"] == approvals[tick]["assessment"]
                    and cross["certificate"]["allowed"] and cross["certificate"]["reason"] == "measured_full_capsule"
                    and end and end["outcome"] == "crossed" and end["paid"] == tick,
                    "Crossing receipt lacks an actual certified zero-collision forward primitive")
            counts["guarded_actual_crossings"] += 1
        attempts.append({
            "inspection_id": seq, "dispatch_paid": start, "arrival_paid": view["paid"] if view else None,
            "crossing_paid": cross["paid"] if cross else None, "last_paid": end_paid,
            "outcome": end["outcome"] if end else "incomplete_at_native_stop",
            "edge": edge, "view": dispatch["view"], "candidate_count": len(offered),
            "view_actions_executed": len(actual_by_stage["view"]), "cross_actions_executed": len(actual_by_stage["cross"]),
            "missing_ground_probes_at_dispatch": len(dispatch["missing_xy_m"]),
            "newly_supported_probes": view["newly_supported"] if view else None,
            "fresh_viewpoint_packet": bool(view and view["fresh_packet"]),
            "at_least_one_previously_unrecorded_sensor_array": novel_sensor_values,
            "full_view_certificate": bool(view and view["certificate"]["allowed"]),
            "positive_support_packets": witnesses,
            "endpoint_view_added_support": bool(view and any(event["paid"] == view["paid"] for event in witnesses)),
            "receipts": group,
        })
    return {"attempts": attempts, "event_counts": dict(counts), "rejection_counts": dict(reasons),
            "rejected_edges": rejections, "other_observer_receipt_counts": dict(diagnostics)}


def requested_event_clocks(audit):
    return {clock for attempt in audit["attempts"] for clock in (
        attempt["dispatch_paid"], attempt["arrival_paid"], attempt["crossing_paid"],
        attempt["crossing_paid"] - 1 if attempt["crossing_paid"] is not None else None,
        attempt["last_paid"],
    ) if clock is not None}


def attach_event_measures(audit, states, source, measure):
    for attempt in audit["attempts"]:
        before, arrival, crossing = (attempt[k] for k in ("dispatch_paid", "arrival_paid", "crossing_paid"))
        windows = {"inspection_interval": (before, attempt["last_paid"])}
        if arrival is not None:
            windows["route_to_new_view"] = (before, arrival)
        if crossing is not None:
            windows["crossing_primitive_only"] = (crossing-1, crossing)
        for name, (left, right) in windows.items():
            attempt[name] = {"from_paid": left, "to_paid": right,
                "from_C_free": states[left][0]["C_free"], "to_C_free": states[right][0]["C_free"],
                **measure(states[left], states[right], source)}


def matched_outcomes(runs, audits, last):
    a, b = runs.values()
    inspection = audits.get("B", {})
    attempts = inspection.get("attempts", [])
    crossed = any(attempt["crossing_paid"] is not None for attempt in attempts)
    category = ("technical_failure" if b["status"] != "completed_execution"
                else "certified_crossing" if crossed
                else "inspection_attempt_without_crossing" if attempts else "no_dispatched_inspection")
    different = None
    if last is not None:
        for earlier, later in zip(a["trace"], b["trace"]):
            if any(earlier[key] != later[key] for key in ("action", "position", "heading", "packet_sha256", "collision")):
                different = earlier["step"]
                break
    a95, b95 = a.get("A95_paid"), b.get("A95_paid")
    return {
        "B_inspection_state": category, "B_attempt_count": len(attempts),
        "B_charged_inspection_actions": sum(x["view_actions_executed"] + x["cross_actions_executed"] for x in attempts),
        "first_unequal_physical_action_or_packet": different,
        "identical_paid_prefix_through": (different - 1 if different is not None else last),
        "A_A95_paid": a95, "B_A95_paid": b95,
        "delta_A95_paid_B_minus_A": b95 - a95 if a95 is not None and b95 is not None else None,
        "A_end_at_anchor_pose": a.get("ended_at_anchor_pose"), "B_end_at_anchor_pose": b.get("ended_at_anchor_pose"),
        "terminal_observation_is_fixed_clock": a.get("actual_paid") == b.get("actual_paid") and last is not None,
    }


class RawObserverValidation:
    """Recompute exact receipts from raw measured sensor packets, with no policy calls."""

    def __init__(self, evidence, audit, initial):
        self.evidence = evidence
        self.anchor = initial["anchor"]
        self.requests = defaultdict(list)
        self.counts = Counter()
        for attempt in audit["attempts"]:
            dispatch, view, replan, cross = (stage_at(attempt, kind) for kind in ("dispatch", "view", "replan", "cross"))
            rows = [("dispatch", dispatch, dispatch["paid"], "prior_certificate")]
            if view:
                rows.append(("view", view, view["paid"], "certificate"))
            if replan:
                rows.append(("replan", replan, replan["paid"], "certificate"))
            if cross:
                rows.append(("cross", cross, cross["paid"]-1, "certificate"))
            rows.extend(("support", item, item["paid"], None) for item in attempt["positive_support_packets"])
            for kind, event, paid, field in rows:
                self.requests[paid].append((kind, event, field))

    def observe(self, packet, belief):
        self.evidence.observe(packet.frame, packet.scan)
        for kind, event, field in self.requests.get(packet.action_id, []):
            proof = self.evidence.raw.certify(*(self.evidence.xy(cell) for cell in event["edge"]), self.evidence.radius)
            if field:
                require(proof == event[field], f"Raw sensor prefix fails to reproduce {kind} certificate at p{packet.action_id}")
            if kind in ("view", "dispatch", "support"):
                geometry = self.evidence
                module = __import__("sensor_probe.evidence", fromlist=["_swept_xy", "MAX_FLOOR_SAMPLE_DISTANCE_M"])
                swept = module._swept_xy(*(geometry.xy(cell) for cell in event["edge"]), geometry.radius)
                tree = geometry.raw._trees[2]
                missing = swept if tree is None else swept[tree.query(swept)[0] > module.MAX_FLOOR_SAMPLE_DISTANCE_M]
                name = "missing_xy_m" if kind == "dispatch" else "remaining_missing_xy_m"
                require(missing.tolist() == event[name], f"Raw ground support coordinates disagree at p{packet.action_id}")
            if kind in ("dispatch", "replan"):
                graph = self.evidence.build(belief)
                matrix, _, ids = graph.orientation_graph()
                states = event["ordered_poses"]
                nodes = [int(ids[tuple(pose[:2])])*4 + pose[2] for pose in [*states, self.anchor]]
                require(min(nodes) >= 0 and all(matrix[a,b] == 1 for a,b in zip(nodes[:-2], nodes[1:-1])),
                        f"Planned {kind} path uses a missing measured edge at p{packet.action_id}")
                distance = dijkstra(matrix, directed=True, indices=nodes[-2])[nodes[-1]]
                require(math.isfinite(distance) and distance == event["return_actions"]
                        and packet.action_id+len(event["ordered_actions"])+distance <= 192,
                        f"Measured return reserve differs at p{packet.action_id}")
            self.counts[kind] += 1

    def finish(self):
        require(sum(self.counts.values()) == sum(map(len, self.requests.values())),
                "Missing one or more independently recomputed raw ground checks")
        return dict(self.counts)
