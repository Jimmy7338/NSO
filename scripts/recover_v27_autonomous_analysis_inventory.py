#!/usr/bin/env python3
"""Output-only recovery for the frozen V27 analyzer's nested inventory filter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_observed_autonomous_v27 as analyzer

BATCH = ROOT/'audit_results/observed_autonomous_v27_20260916'
PROTOCOL = ROOT/'docs/research/V27_AUTONOMOUS_ANALYSIS_INVENTORY_RECOVERY_20260916.md'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def corrected_verify_inventory(folder):
    folder = Path(folder)
    inventory = analyzer.read(folder/'artifact_hashes.json')
    own = (folder/'artifact_hashes.json').resolve()
    actual = {str(path.relative_to(folder)) for path in folder.rglob('*')
              if path.is_file() and path.resolve() != own}
    analyzer.require(actual == set(inventory), 'artifact set differs: '+str(folder))
    for name, expected in inventory.items():
        analyzer.require(sha(folder/name) == expected, 'artifact hash differs: '+str(folder/name))
    return inventory


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def preflight():
    with tempfile.TemporaryDirectory(prefix='v27-analysis-inventory-') as temporary:
        root = Path(temporary)/'root'; child = root/'case_00'
        child.mkdir(parents=True)
        (child/'result.json').write_text('{}\n')
        write(child/'artifact_hashes.json', {'result.json': sha(child/'result.json')})
        root_inventory = {'case_00/result.json': sha(child/'result.json'),
                          'case_00/artifact_hashes.json': sha(child/'artifact_hashes.json')}
        write(root/'artifact_hashes.json', root_inventory)
        old_failed = False
        try:
            analyzer.verify_inventory(root)
        except ValueError:
            old_failed = True
        if not old_failed:
            raise ValueError('fixture did not reproduce frozen analyzer bug')
        corrected_verify_inventory(child)
        corrected_verify_inventory(root)
    return dict(status='passed', old_filter_rejected_nested_inventory=True,
        corrected_root_and_child_inventory_passed=True, metric_or_pairing_code_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source', type=Path, default=analyzer.SOURCE)
    parser.add_argument('--output', type=Path, default=analyzer.OUTPUT)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight()), flush=True)
        return
    if not args.run:
        parser.error('--preflight or --run required')
    manifest = analyzer.read(BATCH/'manifest.json')
    expected = manifest['source_sha256']['scripts/analyze_observed_autonomous_v27.py']
    if sha(Path(analyzer.__file__)) != expected:
        raise ValueError('frozen analyzer changed')
    if args.output.exists():
        raise ValueError('fresh analysis output required')
    preflight_result = preflight()
    analyzer.verify_inventory = corrected_verify_inventory
    analyzer.execute(args.source.resolve(), args.output.resolve())
    receipt = dict(status='complete_output_only_nested_inventory_recovery',
        frozen_analyzer_sha256=expected, wrapper_sha256=sha(Path(__file__)),
        recovery_protocol_sha256=sha(PROTOCOL), preflight=preflight_result,
        acquisition_modified=False, metric_or_pairing_code_changed=False,
        output=str(args.output.resolve()))
    recovery = ROOT/'audit_results/observed_autonomous_v27_analysis_inventory_recovery_20260916'
    recovery.mkdir()
    write(recovery/'result.json', receipt)
    files = {'result.json': sha(recovery/'result.json')}
    write(recovery/'artifact_hashes.json', files)


if __name__ == '__main__':
    main()
