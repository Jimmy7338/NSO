"""Run registered combined-S actors with the autonomous observer enabled only on B."""

import argparse
import faulthandler
import importlib.util
from pathlib import Path
import sys

from study import ARMS, DEV_FREEZE, FREEZE, PREVIOUS, ROOT, SCHEMA, SOURCE, supervisor


def main():
    faulthandler.enable()
    cli = argparse.ArgumentParser(description=__doc__, add_help=False)
    cli.add_argument("layout")
    cli.add_argument("arm", choices=ARMS)
    cli.add_argument("--freeze", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args, _ = cli.parse_known_args()
    freeze = args.freeze.resolve()
    phase = {DEV_FREEZE: "development", FREEZE: "formal"}.get(freeze)
    if phase is None:
        raise ValueError("Worker requires a registered development or formal freeze")
    helper = supervisor(phase)
    registered = helper.read(freeze)
    jobs = [row for row in registered["jobs"] if (row["layout"], row["arm"]) == (args.layout, args.arm)]
    location = args.out.resolve()
    if (registered["schema"] != SCHEMA or registered["phase"] != phase or location.name != "run"
            or not location.parent.name.startswith("attempt_") or location.parent.parent != helper.RUNS / args.layout / args.arm
            or len(jobs) != 1 or jobs[0]["native_policy"] != "combined_S"
            or jobs[0]["active_ground_reacquisition"] != (args.arm == "B")):
        raise ValueError("Unregistered paired job, output directory or treatment assignment")
    if helper.sha(ROOT / registered["protocol_relative"]) != registered["protocol_sha256"]:
        raise ValueError("Worker protocol differs from the registered freeze")
    for key in ("qualification_files", "simulator_only_files"):
        helper.verify(registered[key])

    sys.path.insert(1, str(PREVIOUS))
    from runtime_sweep import install_for_actor as install_sweep

    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("published_autonomous_reacquisition_actor", SOURCE / "runner.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.ROOT = ROOT
    context = {"study_root": ROOT, "frozen": registered, "layout": args.layout, "arm": args.arm}
    install_sweep(**context)
    if jobs[0]["active_ground_reacquisition"]:
        from reacquisition_adapter import install_for_actor as install_observer
        install_observer(**context)
    policy.main()


if __name__ == "__main__":
    main()
