"""Run a registered V2 actor with the qualified online swept-footprint adapter."""

import argparse
import faulthandler
import importlib.util
from pathlib import Path
import sys

from study import FREEZE, ROOT, SCHEMA, SOURCE, supervisor


def main():
    faulthandler.enable()
    cli = argparse.ArgumentParser(description=__doc__, add_help=False)
    cli.add_argument("layout")
    cli.add_argument("arm")
    cli.add_argument("--freeze", required=True, type=Path)
    args, _ = cli.parse_known_args()
    if args.freeze.resolve() != FREEZE:
        raise ValueError("Online-sweep workers require their own registered formal freeze")
    helper = supervisor()
    frozen = helper.read(FREEZE)
    if frozen["schema"] != SCHEMA or not any(
            row["layout"] == args.layout and row["arm"] == args.arm for row in frozen["jobs"]):
        raise ValueError("Unregistered online-sweep worker")
    if helper.sha(ROOT / frozen["protocol_relative"]) != frozen["protocol_sha256"]:
        raise ValueError("Worker protocol differs from the formal freeze")
    for name in ("qualification_files", "simulator_only_files"):
        helper.verify(frozen[name])

    from runtime_sweep import install_for_actor

    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("published_online_sweep_actor", SOURCE / "runner.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.ROOT = ROOT
    install_for_actor(study_root=ROOT, frozen=frozen, layout=args.layout, arm=args.arm)
    policy.main()


if __name__ == "__main__":
    main()
