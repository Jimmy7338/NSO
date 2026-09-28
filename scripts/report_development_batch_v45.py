#!/usr/bin/env python3
"""Report all declared development slots from saved records; no experiments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.batch_report_v45 import report_batch_v45


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-summary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = report_batch_v45(args.batch_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], reserved_slots=result['reserved_slots'],
                         verified_autonomous_endpoints=result['verified_autonomous_endpoints'])))
    return int(result['status'] == 'report_with_integrity_errors')


if __name__ == '__main__':
    sys.exit(main())
