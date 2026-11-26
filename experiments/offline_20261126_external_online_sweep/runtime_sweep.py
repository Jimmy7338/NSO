"""Bind one observed swept-edge graph to all three frozen planning arms."""

from pathlib import Path
from types import FunctionType

import numpy as np

from observed_sweep import DIRECTIONS, SweepEvidence


def rebound(fn, **dependencies):
    """Use a frozen function with explicit geometry dependencies in this actor."""
    bound = FunctionType(fn.__code__, fn.__globals__ | dependencies, fn.__name__, fn.__defaults__, fn.__closure__)
    bound.__kwdefaults__ = fn.__kwdefaults__
    return bound


def attach(components, runtime):
    from nso.execution_guard_v8 import ActionAssessment, ObservedExecutionGuard
    from nso_bridge import exploration_joint_policy as explorer
    import regional_adapter

    backend = components._cpu_backend
    site = backend.scenes[0]
    mapper = site["mapper"]
    config = mapper.config
    evidence = SweepEvidence(mapper.shape, config.resolution_m, config.robot_radius_m)
    packet = site["packet"]
    evidence.observe(packet.frame, packet.scan)
    before_bind = backend.bind_packet
    before_topo = backend.update_topo.__func__
    before_generate = explorer.generate_options
    before_certify = regional_adapter.certify
    original_inflate = before_generate.__globals__["inflated_obstacles"]

    class Guard(ObservedExecutionGuard):
        def safe_grid(self, belief):
            return evidence.build(belief).safe

        def return_plan(self, belief, *args):
            geometry = evidence.build(belief)
            bound = rebound(ObservedExecutionGuard.return_plan, orientation_graph=lambda _: geometry.orientation_graph())
            return bound(self, belief, *args)

        def assess(self, belief, position, heading, action, remaining_actions):
            result = super().assess(belief, position, heading, action, remaining_actions)
            if result.allowed and action == "forward":
                geometry = evidence.build(belief)
                key = (*position, heading)
                if not geometry.edges[key]:
                    return ActionAssessment(False, "forward_capsule_no_observed_edge", result.target)
                if key in geometry.added:
                    runtime.audit.append(dict(event="observed_swept_edge_certified",
                        paid_before=site["packet"].action_id, source=list(position), heading=int(heading),
                        target=list(result.target), certificate=geometry.added[key]))
            return result

    def bind_packet(scene, received, *, arrived=False):
        out = before_bind(scene, received, arrived=arrived)
        evidence.observe(received.frame, received.scan)
        return out

    def update_topo(scene):
        geometry = evidence.build(backend.scenes[scene]["mapper"].belief)
        return rebound(before_topo, label=lambda *a, **k: geometry.regions())(backend, scene)

    def generate(mapper_arg, *args, **kwargs):
        geometry = evidence.build(mapper_arg.belief)
        first = True

        def inflate(mask, scale):
            nonlocal first
            if first:
                first = False
                if not np.array_equal(mask, mapper_arg.belief != 0):
                    raise ValueError("Native option source changed its observed navigability call")
                return ~geometry.safe
            return original_inflate(mask, scale)  # retain the native gain mask

        return rebound(before_generate, inflated_obstacles=inflate,
                       orientation_graph=lambda _: geometry.orientation_graph())(mapper_arg, *args, **kwargs)

    def certify(options, existing, state, source):
        geometry = evidence.build(state["mapper"].belief)
        return before_certify(options, existing, state, GeometrySource(source, geometry))

    site["guard"] = Guard(config.resolution_m, config.robot_radius_m)
    site["guard"]._sweep_evidence = evidence
    explorer.generate_options = generate
    regional_adapter.certify = certify
    backend.bind_packet = bind_packet
    backend.update_topo = update_topo
    # Initial observations have already been accepted by the native bootstrap.
    labels, count = evidence.build(mapper.belief).regions()
    site["graph"].update(regions=labels, nodes=list(range(1, count+1)),
                         current_region=int(labels[packet.position]),
                         anchor_region=int(labels[site["return_anchor"][:2]]))
    runtime._observed_sweep = evidence


class GeometrySource:
    """Pass only navigation geometry to the original regional materializer."""

    def __init__(self, native, geometry):
        self.native, self.geometry = native, geometry

    def inflated_obstacles(self, *_args):
        return ~self.geometry.safe

    def orientation_graph(self, _safe):
        return self.geometry.orientation_graph()

    def __getattr__(self, name):
        return getattr(self.native, name)


def bind_frontier(driver, runtime):
    policy = driver.policy
    original_act, original_select = policy.act.__func__, policy._select.__func__

    def geometry():
        return runtime._observed_sweep.frontier(runtime.states[0]["mapper"].belief)

    def choose(obs, safe):
        graph = geometry()
        if not graph.added:
            return original_select(policy, obs, safe)
        return rebound(original_select, shortest_paths=graph.shortest_paths)(policy, obs, safe)

    def act(obs):
        graph = geometry()
        if not graph.added:
            return original_act(policy, obs)
        path = [obs.position, *policy._path]
        for a, b in zip(path, path[1:]):
            delta = b[0]-a[0], b[1]-a[1]
            if delta == (0, 0):
                continue
            if delta not in DIRECTIONS or not graph.edges[*a, DIRECTIONS.index(delta)]:
                policy.goal, policy._path = None, []
                break
        return rebound(original_act, inflated_obstacles=lambda *_: ~graph.safe)(policy, obs)

    policy._select, policy.act = choose, act
    return driver


def bind_local(module):
    before = module._local_variants

    def variants(state, *args, **kwargs):
        guard = state["guard"]
        geometry = guard._sweep_evidence.build(state["mapper"].belief)
        return rebound(before, orientation_graph=lambda _: geometry.orientation_graph())(state, *args, **kwargs)

    module._local_variants = variants


def install_for_actor(*, study_root, frozen, layout, arm):
    """Install a process-local adapter before the archived regional runner starts."""
    import formal_runner

    if not isinstance(study_root, Path) or not any(
        row["layout"] == layout and row["arm"] == arm for row in frozen["jobs"]
    ):
        raise ValueError("The swept-edge adapter requires a registered actor")
    regional_runner = formal_runner.runner
    before_import = regional_runner.import_file

    def load_inherited(name, path):
        inherited = before_import(name, path)
        if name != "original_weighted_actor":
            return inherited
        native = inherited.native
        before_setup, before_frontier, before_load = native.setup_runtime, native.frontier_adapter, native.load_source

        def setup(*args, **kwargs):
            result = before_setup(*args, **kwargs)
            components, runtime = result[1][3:5]
            attach(components, runtime)
            return result

        def frontier(components, runtime, world):
            return bind_frontier(before_frontier(components, runtime, world), runtime)

        def load(name, path):
            module = before_load(name, path)
            if Path(path) == native.METHOD:
                bind_local(module)
            return module

        native.setup_runtime, native.frontier_adapter, native.load_source = setup, frontier, load
        return inherited

    regional_runner.import_file = load_inherited
