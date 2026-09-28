#!/usr/bin/env python3
"""Preflight/run one frozen SEM slot, inspect it, or replay and score it."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.semantic_scene_experiment import inspect_experiment,review_experiment,run_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('run','preflight'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--protocol',required=True,type=Path)
        cmd.add_argument('--run-id',required=True)
    for name in ('inspect','review'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--episode',required=True,type=Path)
        cmd.add_argument('--expected-manifest-sha256',required=True)
        if name=='review':
            cmd.add_argument('--output',required=True,type=Path)
            cmd.add_argument('--replay-only',action='store_true')
    args=parser.parse_args()
    if args.command in ('run','preflight'):
        result=run_experiment(args.protocol,args.run_id,preflight_only=args.command=='preflight')
    elif args.command=='inspect':
        episode=inspect_experiment(args.episode,args.expected_manifest_sha256)
        result=dict(status='inspected',run_id=episode['started']['run_id'],episode_status=episode['result']['status'],
            frames=episode['result']['acquired_and_saved_packets'],episode_manifest_sha256=episode['manifest_sha256'],
            current_sources_match=episode['current_sources_match'],current_source_differences=episode['current_source_differences'])
    else:
        result=review_experiment(args.episode,expected_manifest_sha256=args.expected_manifest_sha256,
            output=args.output,replay_only=args.replay_only)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status'] in ('ready_without_world_creation','controller_stop','inspected',
        'experiment_reviewed','experiment_replay_verified') else 2


if __name__=='__main__':sys.exit(main())
