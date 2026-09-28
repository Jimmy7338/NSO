#!/usr/bin/env python3
"""New-phase bounded CPU pilots and saved-observation review; no old ledger writes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.mechanism_development import DEFAULT_OUTPUT, run_pilot, review_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--run-id', required=True, choices=('pilot_A_G', 'pilot_A_S'))
    run.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    run.add_argument('--preflight-only', action='store_true')
    review = sub.add_parser('review')
    review.add_argument('--episode', type=Path, required=True)
    review.add_argument('--expected-manifest-sha256', required=True)
    review.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run_pilot(args.run_id, output_root=args.output_root, preflight_only=args.preflight_only)
    else:
        result = review_pilot(args.episode, expected_manifest_sha256=args.expected_manifest_sha256,
                              output=args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] in ('ready_without_world_creation', 'controller_stop', 'pilot_reviewed') else 2


if __name__ == '__main__':
    sys.exit(main())
