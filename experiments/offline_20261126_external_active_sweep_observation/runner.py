"""Run an active-observation actor in its sealed gate or formal phase."""

import argparse
import faulthandler
import importlib.util
from pathlib import Path
import sys

from study import FREEZE, GATE_FREEZE, PREVIOUS, ROOT, SCHEMA, SOURCE, supervisor


def main():
    faulthandler.enable()
    cli = argparse.ArgumentParser(description=__doc__, add_help=False)
    cli.add_argument("layout")
    cli.add_argument("arm")
    cli.add_argument("--freeze", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args, _ = cli.parse_known_args()
    freeze = args.freeze.resolve()
    phase = {GATE_FREEZE: "gate", FREEZE: "formal"}.get(freeze)
    if phase is None:
        raise ValueError("Worker requires a registered active-observation gate or formal freeze")
    helper = supervisor(phase)
    frozen = helper.read(freeze)
    attempt = args.out.resolve()
    if (frozen["schema"] != SCHEMA or frozen["phase"] != phase
            or not any((row["layout"], row["arm"]) == (args.layout, args.arm) for row in frozen["jobs"])
            or attempt.parent.parent != helper.RUNS / args.layout / args.arm):
        raise ValueError("Unregistered active-observation worker or output directory")
    if helper.sha(ROOT / frozen["protocol_relative"]) != frozen["protocol_sha256"]:
        raise ValueError("Worker protocol differs from the phase freeze")
    for name in ("qualification_files", "simulator_only_files"):
        helper.verify(frozen[name])

    sys.path.insert(1, str(PREVIOUS))
    from runtime_sweep import install_for_actor as install_sweep
    from active_observation import install_for_actor as install_active_observation

    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("published_active_observation_actor", SOURCE / "runner.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.ROOT = ROOT
    context = {"study_root": ROOT, "frozen": frozen, "layout": args.layout, "arm": args.arm}
    install_sweep(**context)
    install_active_observation(**context)
    policy.main()


if __name__ == "__main__":
    main()
