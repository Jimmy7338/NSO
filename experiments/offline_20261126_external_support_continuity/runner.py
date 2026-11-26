"""Run each published policy with the frozen support-cache repair."""

import argparse
import faulthandler
import importlib.util
from pathlib import Path
import sys

from study import ROOT, SOURCE, supervisor


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("layout")
    parser.add_argument("arm")
    parser.add_argument("--freeze", type=Path, required=True)
    args, _ = parser.parse_known_args()
    helper = supervisor()
    frozen = helper.read(args.freeze)
    if not args.freeze.resolve().is_relative_to(ROOT):
        raise ValueError("Runner freeze must belong to this repair study")
    if helper.sha(ROOT / frozen["protocol_relative"]) != frozen["protocol_sha256"]:
        raise ValueError("Protocol differs from sealed launch")
    helper.verify(frozen["simulator_only_files"])
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("published_support_repair_actor", SOURCE / "runner.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.ROOT = ROOT
    policy.main()


if __name__ == "__main__":
    main()
