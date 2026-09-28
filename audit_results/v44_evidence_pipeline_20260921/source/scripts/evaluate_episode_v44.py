#!/usr/bin/env python3
"""Prepare a static DEV reference, or evaluate a complete saved episode read-only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.offline_evaluation_v44 import prepare_reference_v44, evaluate_saved_episode_v44


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-reference')
    prepare.add_argument('--asset-id', required=True)
    prepare.add_argument('--output', required=True, type=Path)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--episode', required=True, type=Path)
    evaluate.add_argument('--reference', required=True, type=Path)
    evaluate.add_argument('--reference-manifest-sha256', required=True)
    evaluate.add_argument('--episode-manifest-sha256', required=True)
    evaluate.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare-reference':
        result = prepare_reference_v44(args.asset_id, args.output)
    else:
        destination = args.output.resolve()
        for root in (args.episode.resolve(), args.reference.resolve()):
            if destination.is_relative_to(root):
                parser.error('evaluation output must be outside immutable episode and reference directories')
        if destination.exists():
            raise FileExistsError('new evaluation output required; no overwriting retained results')
        result = evaluate_saved_episode_v44(args.episode, args.reference,
            reference_manifest_sha256=args.reference_manifest_sha256,
            expected_episode_manifest_sha256=args.episode_manifest_sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
