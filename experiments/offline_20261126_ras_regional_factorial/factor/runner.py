"""Start the published actor with the sealed regional factor flags."""

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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("layout")
    parser.add_argument("arm")
    parser.add_argument("--freeze", type=Path, required=True)
    args, _ = parser.parse_known_args()
    from factor_study import supervisor
    native = supervisor()
    frozen = json.loads(args.freeze.read_text())
    if (args.freeze.resolve() != ROOT / "FORMAL_FREEZE.json"
            or native.sha(ROOT / "PROTOCOL.json") != frozen["protocol_sha256"]):
        raise ValueError("Unsealed factorial actor request")
    native.verify(frozen["simulator_only_files"])
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("published_region_factor_actor", SOURCE / "runner.py")
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    actor.ROOT = ROOT
    actor.main()


if __name__ == "__main__":
    main()
