"""Bind the online policy to the matched simulation actor."""

from autonomous_observation import ActiveGroundObserver


class Binding:
    def __init__(self, components, runtime):
        self.components, self.runtime = components, runtime
        self._site = components._cpu_backend.scenes[0]
        self.evidence = runtime._observed_sweep
        self.audit = runtime.audit

    @property
    def state(self):
        return self.runtime.states[0]

    @property
    def belief(self):
        return self._site["mapper"].belief

    @property
    def anchor(self):
        return self._site["return_anchor"]

    @property
    def remaining(self):
        return self._site["ledger"].remaining_budget

    def prepare(self):
        state = self.state
        state["active_actions"], state["option"], state["phase"] = [], None, "outbound"
        self.components.bind_execution_option(0, None)

    def assess(self, action):
        self.components.bind_execution_option(0, None)
        return self.components.assess_local_action(0, action)

    def authorize(self, action, check, phase):
        state = self.state
        packet = state["packet"]
        state["active_actions"], state["pending"] = [action], check
        self.audit.append(dict(event="paid_action_authorized", frame_id=packet.frame_id,
                               next_action_id=packet.action_id + 1, phase=state["phase"],
                               assessment=check, option_id=None, selection_call_id=None,
                               ground_reacquisition_stage=phase))


def attach(components, runtime):
    observer = ActiveGroundObserver(Binding(components, runtime))
    original_next, original_observe = runtime.next_local_action, runtime.observe

    def next_action(scene):
        action = observer.next()
        return original_next(scene) if action is None else action

    def observe(scene, *args, **kwargs):
        receipt = original_observe(scene, *args, **kwargs)
        if receipt["accepted"] and not receipt["duplicate"]:
            observer.observe()
        return receipt

    runtime.next_local_action, runtime.observe = next_action, observe


def install_for_actor(*, study_root, frozen, layout, arm):
    """External experimental routing of arm assignment, never exposed to the policy."""
    import formal_runner

    if not any(row["layout"] == layout and row["arm"] == arm and row["active_ground_reacquisition"]
               for row in frozen["jobs"]):
        raise ValueError(f"Unregistered active-observation arm: {study_root}")
    regional = formal_runner.runner
    prior_import = regional.import_file

    def load_inherited(name, path):
        inherited = prior_import(name, path)
        if name != "original_weighted_actor":
            return inherited
        prior_setup = inherited.native.setup_runtime

        def setup(*args, **kwargs):
            result = prior_setup(*args, **kwargs)
            attach(*result[1][3:5])
            return result

        inherited.native.setup_runtime = setup
        return inherited

    regional.import_file = load_inherited
