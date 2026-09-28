#!/usr/bin/env python3
"""Build small new semantic assets and public navigation; never create a World."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nso.semantic_scene_assets import DEFAULT_DRAFT, build_assets
from nso.semantic_scene_navigation import build_public_navigation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, default=DEFAULT_DRAFT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--navigation-output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.navigation_output.resolve():
        raise ValueError('private asset root and public navigation root must be separate')
    if args.output.exists() or args.navigation_output.exists():
        raise FileExistsError('both output roots must be new')
    assets = build_assets(args.output, draft_path=args.draft)
    navigation = build_public_navigation(args.output, args.navigation_output,
        expected_asset_manifest_sha256=assets['manifest_sha256'])
    print(json.dumps(dict(assets=assets, navigation=navigation,
        worlds_created=0, sensor_frames_created=0, policy_trajectories_created=0,
        quality_evaluations=0, primary_matrix_started=False), sort_keys=True))


if __name__ == '__main__':
    main()
