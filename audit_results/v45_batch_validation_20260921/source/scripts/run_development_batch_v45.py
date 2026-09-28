#!/usr/bin/env python3
"""Run/resume the same five development slots with verified saved evidence."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.development_batch_v45 import run_batch_v45

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
        default=ROOT/'audit_results/v45_development_evidence_20260921')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    result = run_batch_v45(args.output_root, preflight_only=args.preflight_only)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] in ('batch_completed', 'ready_without_world_creation') else 2

if __name__ == '__main__':
    sys.exit(main())
