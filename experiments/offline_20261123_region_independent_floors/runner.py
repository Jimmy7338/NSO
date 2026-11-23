#!/usr/bin/env python3
"""Launch the published region-attribution runner on the new frozen floors."""

import argparse
import faulthandler
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/"region_global_revision_20261122/next/global_attribution"


def main():
    from study import read, sha, verify

    faulthandler.enable()
    interface=argparse.ArgumentParser(add_help=False)
    interface.add_argument("layout")
    interface.add_argument("arm")
    interface.add_argument("--freeze",type=Path,required=True)
    args,_=interface.parse_known_args()
    frozen=read(args.freeze)
    if args.freeze.resolve()!=ROOT/"FORMAL_FREEZE.json":
        raise ValueError("Use this study's sealed actor configuration.")
    if frozen["protocol_sha256"]!=sha(ROOT/frozen["protocol_relative"]):
        raise ValueError("The formal protocol has changed.")
    verify(frozen["simulator_only_files"])
    sys.path.insert(0,str(SOURCE))
    definition=importlib.util.spec_from_file_location("published_region_attribution_actor",SOURCE/"runner.py")
    runner=importlib.util.module_from_spec(definition)
    definition.loader.exec_module(runner)
    runner.ROOT=ROOT
    runner.main()


if __name__=="__main__":
    main()
