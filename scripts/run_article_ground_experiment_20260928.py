#!/usr/bin/env python3
"""Run one frozen Ground V2 slot within the shared cumulative study ledger."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.article_experiment_ground_v2 import run_episode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    result = run_episode(args.protocol, args.run_id)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result['status'] not in ('controller_stop', 'budget_exhausted'):
        raise SystemExit(1)
