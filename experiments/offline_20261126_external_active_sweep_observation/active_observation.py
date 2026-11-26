"""Acquire a new measured viewpoint before crossing a recognized support gap."""

import hashlib

import numpy as np
from scipy.sparse.csgraph import dijkstra


LAYOUT = "b26_hm3d_00770_s1"
EDGE = ((52, 26), (53, 26))
HEADING = 2
VIEWS = ((44, 25, 2), (44, 26, 2), (44, 24, 2))
MISSING_GROUND = "missing_positive_ground_support"


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def graph_sha(graph):
    fields = (np.asarray(graph.safe.shape, dtype="<i8").tobytes(), np.packbits(graph.safe.ravel()).tobytes(),
              np.packbits(graph.edges.ravel()).tobytes())
    return hashlib.sha256(b"".join(fields)).hexdigest()


def directed_plan(graph, start, goal, anchor, budget, *, cross=False):
    """Reserve a return on the measured graph before any auxiliary action."""
    from nso.route_coverage_v2 import recover_actions

    sparse, cells, ids = graph.orientation_graph()
    node = lambda pose: 4 * int(ids[pose[:2]]) + pose[2]
    entry, destination, home = map(node, (start, goal, anchor))
    if min(entry, destination, home) < 0:
        return None
    target = (*EDGE[1], HEADING)
    reserve_from = node(target) if cross else destination
    if reserve_from < 0 or cross and not graph.edges[*EDGE[0], HEADING]:
        return None
    outward, predecessors = dijkstra(sparse, directed=True, indices=entry, return_predecessors=True)
    backwards = dijkstra(sparse.T.tocsr(), directed=True, indices=home)
    total = outward[destination] + backwards[reserve_from] + int(cross)
    if not np.isfinite(total) or total > budget or outward[destination] + int(cross) <= 0:
        return None
    actions = recover_actions(predecessors, entry, destination, cells)
    cursor = destination
    poses = []
    while cursor != entry:
        poses.append((*map(int, cells[cursor // 4]), int(cursor % 4)))
        cursor = int(predecessors[cursor])
    poses = [start, *reversed(poses)]
    if cross:
        poses.append(target)
        actions.append("forward")
    return {"outbound_states": [list(pose) for pose in poses], "outbound_actions": actions,
            "known_return_from_plan_goal": int(backwards[reserve_from]), "plan_total": int(total)}


class Inspector:
    def __init__(self, components, runtime):
        self.components = components
        self.runtime = runtime
        self.site = components._cpu_backend.scenes[0]
        self.evidence = runtime._observed_sweep
        self.visited = set()
        self.seen_values = set()
        self.tried = set()
        self.route = None
        self.stage = "waiting"
        self.received = None
        self.pending = None
        self.cursor = 0
        self.record_current()

    @property
    def state(self):
        return self.runtime.states[0]

    @property
    def packet(self):
        return self.state["packet"]

    @property
    def pose(self):
        return (*self.packet.position, self.packet.heading)

    @property
    def budget(self):
        return self.site["ledger"].remaining_budget

    def record_current(self):
        packet = self.packet
        values = array_sha(packet.scan.ranges_m), array_sha(packet.frame.depth_m)
        novel = values not in self.seen_values
        self.seen_values.add(values)
        self.visited.add(self.pose)
        return values, novel

    def report(self, event, **details):
        self.runtime.audit.append({"event": event, **details})

    def proof(self):
        p, q = (self.evidence.xy(cell) for cell in EDGE)
        return self.evidence.raw.certify(p, q, self.evidence.radius)

    def stop_inspection(self, reason):
        self.report("active_observation_stopped", paid=self.packet.action_id, reason=reason)
        self.route = None
        self.stage = "done"

    def begin(self, *, cross=False):
        belief = self.site["mapper"].belief
        proof = self.proof()
        if any(belief[cell] != 0 for cell in EDGE):
            return False
        if not cross and proof["reason"] != MISSING_GROUND:
            return False
        if cross and not proof["allowed"]:
            return False
        graph = self.evidence.build(belief)
        goals = ((*EDGE[0], HEADING),) if cross else tuple(v for v in VIEWS if v not in self.visited | self.tried)
        for goal in goals:
            route = directed_plan(graph, self.pose, goal, self.site["return_anchor"], self.budget, cross=cross)
            if route is None:
                continue
            self.stage = "cross" if cross else "observe"
            self.route, self.cursor, self.prior_sweep = route, 0, proof
            self.tried.add(goal)
            state = self.state
            state["active_actions"], state["option"], state["phase"] = [], None, "outbound"
            self.components.bind_execution_option(0, None)
            target_fields = ({"target_sweep_certificate": proof} if cross else {"prior_sweep": proof})
            self.report("active_observation_crossing_replanned" if cross else "active_observation_dispatched",
                        paid_before=self.packet.action_id, pose=list(self.pose), packet_sha256=self.packet.sha256(),
                        selected_view=None if cross else list(goal), target_edge=[list(cell) for cell in EDGE],
                        graph_sha256=graph_sha(graph), **target_fields, **route)
            print(f"active sweep {self.stage} at paid {self.packet.action_id}, goal {goal}", flush=True)
            return True
        return False

    def observe(self):
        (scan_hash, depth_hash), novel = self.record_current()
        if self.pending is None:
            return
        paid, expected = self.pending
        self.pending = None
        if self.packet.action_id != paid or self.packet.collision or self.pose != expected:
            self.stop_inspection("measured_pose_or_collision_prevented_inspection")
            return
        self.cursor += 1
        if self.cursor < len(self.route["outbound_actions"]):
            return
        proof = self.proof()
        if self.stage == "observe":
            self.report("active_observation_acquired", paid=paid, pose=list(self.pose),
                        packet_sha256=self.packet.sha256(), scan_ranges_sha256=scan_hash, depth_values_sha256=depth_hash,
                        novel_sensor_values=novel, prior_sweep=self.prior_sweep, target_sweep_certificate=proof)
            if proof["allowed"] and novel:
                self.received = paid, self.packet.sha256()
                self.stage = "replan"
            elif proof["reason"] == MISSING_GROUND:
                self.stage = "next_view"
            else:
                self.stop_inspection("new_observation_did_not_safely_certify_the_gap")
            print(f"active sweep acquired at paid {paid}, {self.pose}: {proof['reason']}", flush=True)
        else:
            scan_paid, scan_packet = self.received
            self.report("active_observed_edge_executed", scan_paid=scan_paid, edge_paid=paid,
                        from_cell=list(EDGE[0]), to_cell=list(EDGE[1]), complete_sweep=True,
                        scan_packet_sha256=scan_packet, edge_packet_sha256=self.packet.sha256(),
                        target_sweep_certificate=self.final_certificate)
            self.stage = "done"
            print(f"active sweep crossed at paid {paid}, {self.pose}", flush=True)
        self.route = None

    def next(self):
        if self.state["closed"] or self.stage == "done":
            return None
        if self.pending is not None or self.state["pending"] is not None:
            raise RuntimeError("Active inspection is awaiting the paid action's observation")
        if self.stage == "waiting":
            if self.pose[:2] != EDGE[0] or self.state["phase"] == "return":
                return None
            if not self.begin():
                return None
        elif self.stage in ("next_view", "replan"):
            if not self.begin(cross=self.stage == "replan"):
                self.stop_inspection("no_certified_view_or_crossing_with_reserved_return")
                return None

        action = self.route["outbound_actions"][self.cursor]
        expected = tuple(self.route["outbound_states"][self.cursor+1])
        if self.stage == "cross" and self.cursor+1 == len(self.route["outbound_actions"]):
            self.final_certificate = self.proof()
            if not self.final_certificate["allowed"]:
                self.stop_inspection("latest_observation_revoked_full_capsule")
                return None
        self.components.bind_execution_option(0, None)
        check = self.components.assess_local_action(0, action)
        if not check["allowed"] or tuple(check["next_pose"]) != expected:
            self.stop_inspection(check["reason"])
            return None
        state = self.state
        state["active_actions"], state["option"], state["phase"] = [action], None, "outbound"
        state["pending"] = check
        self.pending = self.packet.action_id + 1, expected
        self.report("paid_action_authorized", frame_id=self.packet.frame_id,
                    next_action_id=self.pending[0], phase="outbound", assessment=check,
                    option_id=None, selection_call_id=None, active_observation_stage=self.stage)
        return action


def attach(components, runtime):
    inspector = Inspector(components, runtime)
    before_next, before_observe = runtime.next_local_action, runtime.observe

    def next_action(scene):
        action = inspector.next()
        return before_next(scene) if action is None else action

    def observe(scene, *args, **kwargs):
        receipt = before_observe(scene, *args, **kwargs)
        if receipt["accepted"] and not receipt["duplicate"]:
            inspector.observe()
        return receipt

    runtime.next_local_action, runtime.observe = next_action, observe
    runtime._active_observation = inspector


def install_for_actor(*, study_root, frozen, layout, arm):
    """Expose the same targeted intervention to all registered HM3D s1 arms."""
    if layout != LAYOUT:
        return
    import formal_runner

    regional = formal_runner.runner
    before_import = regional.import_file

    def load_inherited(name, path):
        inherited = before_import(name, path)
        if name != "original_weighted_actor":
            return inherited
        native = inherited.native
        setup_original, frontier_original = native.setup_runtime, native.frontier_adapter

        def setup(*args, **kwargs):
            result = setup_original(*args, **kwargs)
            attach(*result[1][3:5])
            return result

        def frontier(components, runtime, world):
            driver = frontier_original(components, runtime, world)
            next_original = driver.next_action

            def next_action():
                action = runtime._active_observation.next()
                if action is None:
                    return next_original()
                driver.last_decision = None
                driver.policy.goal, driver.policy._path = None, []
                return action

            driver.next_action = next_action
            return driver

        native.setup_runtime, native.frontier_adapter = setup, frontier
        return inherited

    regional.import_file = load_inherited
