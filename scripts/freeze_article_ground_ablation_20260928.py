#!/usr/bin/env python3
"""Freeze the declared twelve paired frontend ablations; creates no World."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.article_experiment_ground_v2 import (
    SCHEMA, METRIC_PREPROCESSING, runtime_metadata, source_hashes,
    validate_protocol, validate_pairing,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = runtime_counts_v41()
    old_path = ROOT/'configs/virtual3d/article_development_v1_20260928.json'
    assert sha(old_path) == '2bfac2bb22995b9b90cdb870f214f6d584439db5d126c02480d806e82cebb4a9'
    path = ROOT/'configs/virtual3d/article_ground_ablation_v2_20260928.json'
    archive = ROOT/'audit_results/article_stage_20260928/execution_sources_ground_v2.zip'
    if path.exists() or archive.exists():
        raise FileExistsError('Frozen configuration/source archive cannot be replaced')
    old = json.loads(old_path.read_text())
    pins = source_hashes()
    assert len(pins) == 62
    for name, pin in old['source_sha256'].items():
        assert sha(ROOT/name) == pin, name
    protocol = deepcopy(old)
    protocol.update(schema=SCHEMA, phase='ablation',
        output_relative_path='audit_results/article_stage_20260928/ground_ablation_v2',
        metric_preprocessing=deepcopy(METRIC_PREPROCESSING),
        numerical_runtime=runtime_metadata(), source_sha256=pins,
        source_archive=str(archive.relative_to(ROOT)),
        paired_baseline_protocol=dict(path=str(old_path.relative_to(ROOT)), sha256=sha(old_path)),
        scope='Post-diagnosis common association-frontend ablation on all three development layouts; not held-out confirmation.',
        scientific_predeclaration=dict(
            path='docs/research/ARTICLE_GROUND_ABLATION_PREDECLARATION_20260928.md',
            sha256=sha(ROOT/'docs/research/ARTICLE_GROUND_ABLATION_PREDECLARATION_20260928.md')),
        common_baseline_evaluation_plan=dict(
            path='audit_results/article_stage_20260928/common_evaluation_plan_v1.json',
            sha256=sha(ROOT/'audit_results/article_stage_20260928/common_evaluation_plan_v1.json')),
        slots={'ground_'+name.removeprefix('dev_'):dict(slot, paired_baseline_run_id=name)
            for name, slot in old['slots'].items()})
    protocol['controller']['ground_association'] = True
    protocol['formal_followup']['requires_actionable_shared_information_mechanism'] = True
    protocol['formal_followup']['positive_endpoint_not_an_activation_gate'] = True
    validate_protocol(protocol)
    validate_pairing(protocol)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for name, pin in sorted(pins.items()):
            content = (ROOT/name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == pin
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            stream.writestr(info, content)
    protocol['source_archive_sha256'] = sha(archive)
    with path.open('x') as output:
        json.dump(protocol, output, sort_keys=True, separators=(',', ':'), allow_nan=False)
        output.write('\n')
    assert runtime_counts_v41() == before
    assert source_hashes() == pins
    print(json.dumps(dict(protocol=str(path.relative_to(ROOT)), protocol_sha256=sha(path),
        source_files=len(pins), source_archive_sha256=sha(archive),
        slots=len(protocol['slots']), new_worlds=0, new_surface_evaluations=0)))


if __name__ == '__main__':
    main()
