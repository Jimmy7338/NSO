"""Start the unchanged regional or frontier policy on a sealed external floor."""

import argparse
import faulthandler
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "region_global_revision_20261122/next/global_attribution"


def main():
    faulthandler.enable()
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("layout")
    cli.add_argument("arm")
    cli.add_argument("--freeze", type=Path, required=True)
    args, _ = cli.parse_known_args()
    from study_external import supervisor
    control = supervisor()
    frozen = json.loads(args.freeze.read_text())
    if (args.freeze.resolve() != ROOT / "FORMAL_FREEZE.json"
            or control.sha(ROOT / frozen["protocol_relative"]) != frozen["protocol_sha256"]):
        raise ValueError("Unsealed heterogeneous actor request")
    control.verify(frozen["simulator_only_files"])
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("published_external_floor_actor", SOURCE / "runner.py")
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    actor.ROOT = ROOT
    actor.main()


if __name__ == "__main__":
    main()
