#!/usr/bin/env python3
"""Execute the frozen CPU detector smoke, never a device-efficacy experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from nso.research_evidence_v31 import (
    freeze, read, seal, sha, verify_inventory, verify_sources, write, write_bytes)
from nso.cpu_semantic_detector_v37 import CPUSemanticDetectorV37, PREDICTION_SETTINGS

OUTPUT = ROOT / 'audit_results/v37_cpu_detector_smoke_20260920'
PROTOCOL = ROOT / 'configs/virtual3d/cpu_frontend_smoke_v37_20260920.json'
PACKAGE = ROOT / '.venv/lib/python3.12/site-packages/ultralytics'


def rows(record):
    return sorted(record['detections'] + record['rejected_detections'], key=lambda r: r['source_index'])


def assert_match(record, reference):
    actual = rows(record)
    np.testing.assert_array_equal([r['source_class_id'] for r in actual], reference.boxes.cls.cpu().numpy())
    np.testing.assert_allclose(np.array([r['box_xyxy'] for r in actual]).reshape(-1, 4),
                               reference.boxes.xyxy.cpu().numpy(), rtol=0, atol=1e-5)
    np.testing.assert_allclose([r['detector_score'] for r in actual],
                               reference.boxes.conf.cpu().numpy(), rtol=0, atol=1e-5)


def run():
    if OUTPUT.exists():
        raise FileExistsError('exclusive-create smoke; no overwrite')
    protocol = read(PROTOCOL)
    assert importlib.metadata.version('ultralytics') == protocol['ultralytics_version']
    for key in ('conf', 'iou', 'imgsz', 'device', 'half', 'rect', 'augment', 'max_det', 'agnostic_nms'):
        assert PREDICTION_SETTINGS[key] == protocol[key], key
    oid = protocol['checkpoint_sha256']
    cache = ROOT / '.git/lfs/objects' / oid[:2] / oid[2:4] / oid
    assert cache.stat().st_size == protocol['checkpoint_bytes'] and sha(cache) == oid
    runtime_dir = ROOT / 'tmp/v37_cpu_detector_runtime'
    runtime_dir.mkdir(exist_ok=True)
    config_dir = runtime_dir / 'settings'
    config_dir.mkdir(exist_ok=True)
    alias = runtime_dir / 'verified_yolov8n.pt'
    if alias.is_symlink():
        assert alias.resolve() == cache.resolve()
    else:
        assert not alias.exists()
        alias.symlink_to(cache)
    os.environ.update(YOLO_CONFIG_DIR=str(config_dir), YOLO_OFFLINE='true',
                      YOLO_AUTOINSTALL='false', YOLO_VERBOSE='false')
    OUTPUT.mkdir()
    (OUTPUT / 'inputs').mkdir()
    for name in protocol['public_package_examples']:
        write_bytes(OUTPUT, OUTPUT / 'inputs' / name, (PACKAGE / 'assets' / name).read_bytes())
    installed_sources = [PACKAGE / name for name in (
        '__init__.py', 'data/loaders.py', 'engine/predictor.py', 'engine/model.py',
        'nn/tasks.py', 'nn/autobackend.py', 'utils/__init__.py', 'utils/torch_utils.py', 'utils/patches.py')]
    inputs = {str((OUTPUT / 'inputs' / name).relative_to(ROOT)): sha(OUTPUT / 'inputs' / name)
              for name in protocol['public_package_examples']}
    source_paths = [Path(__file__), PROTOCOL, ROOT / 'nso/cpu_semantic_detector_v37.py',
                    ROOT / 'tests/virtual3d/test_cpu_semantic_detector_v37.py',
                    ROOT / 'yolov8n.pt'] + installed_sources
    freeze(OUTPUT, source_paths, input_sha256=inputs,
           fixed_model_sha256=oid, checkpoint_bytes=protocol['checkpoint_bytes'],
           checkpoint_downloaded_this_round=False,
           scope=protocol['scope'], package_example_images_are_not_equipment_test_instances=True,
           model_bytes_not_copied_into_small_source_archive=True)
    wheel_receipt = read('/tmp/nso_v37_cpu_wheels_receipt.json')
    write(OUTPUT, OUTPUT / 'environment_installation.json', wheel_receipt)
    write(OUTPUT, OUTPUT / 'prior_import_probe.json', read('/tmp/nso_v37_import_probe.json'))
    counts = dict(explicit_predict_calls=0, new_world_calls=0, new_TSDF_calls=0,
                  new_quality_calls=0, new_main_tasks=0)
    network_attempts = []

    def deny_network(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'socket.sendmsg'):
            network_attempts.append(event)
            raise OSError('V37 inference has network disabled')

    sys.addaudithook(deny_network)
    started = time.perf_counter()
    status = 'failed'
    try:
        transcript = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests/virtual3d'),
                                                    pattern='test_cpu_semantic_detector_v37.py')
        tests = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
        write_bytes(OUTPUT, OUTPUT / 'tests.txt', transcript.getvalue().encode())
        assert tests.wasSuccessful() and tests.testsRun == 5
        detector = CPUSemanticDetectorV37(alias, oid, target_taxonomy=protocol['target_taxonomy'])

        def detect(rgb):
            counts['explicit_predict_calls'] += 1
            result = detector.detect(rgb)
            assert result['model_execution_performed']
            assert result['runtime']['execution_backend'] == 'ultralytics_pinned_cpu'
            assert all(p['geometry_fallback'] and p['probabilities'] == [.5, .5]
                       for p in result['configuration_priors'])
            return result

        blank = np.zeros((640, 640, 3), np.uint8)
        warmup = detect(blank)
        write(OUTPUT, OUTPUT / 'warmup.json', warmup)
        summaries = []
        for name in protocol['public_package_examples']:
            rgb = np.array(Image.open(OUTPUT / 'inputs' / name).convert('RGB'))
            records = [detect(rgb) for _ in range(protocol['timed_repeats_per_example'])]
            for record in records[1:]:
                assert rows(record) == rows(records[0])
                assert record['configuration_priors'] == records[0]['configuration_priors']
            counts['explicit_predict_calls'] += 1
            reference = detector._model.predict(source=Image.fromarray(rgb), **PREDICTION_SETTINGS)[0]
            assert_match(records[0], reference)
            milliseconds = [r['runtime']['prediction_wall_seconds'] * 1000 for r in records]
            summary = dict(input_file=name, input_sha256=inputs[str((OUTPUT / 'inputs' / name).relative_to(ROOT))],
                           detections=len(rows(records[0])),
                           class_counts=dict(Counter(r['source_class_name'] for r in rows(records[0]))),
                           prediction_wall_ms=milliseconds, median_prediction_wall_ms=statistics.median(milliseconds),
                           PIL_reference_matches=True, repeats_identical=True,
                           independent_samples_added_by_repeats=0)
            write(OUTPUT, OUTPUT / (Path(name).stem + '_predictions.json'),
                  dict(summary=summary, repetitions=records,
                       PIL_reference=dict(boxes=reference.boxes.xyxy.cpu().numpy().tolist(),
                                          scores=reference.boxes.conf.cpu().numpy().tolist(),
                                          classes=reference.boxes.cls.cpu().numpy().tolist())))
            summaries.append(summary)
        negative = detect(blank)
        write(OUTPUT, OUTPUT / 'blank_control.json', negative)
        assert counts['explicit_predict_calls'] == protocol['expected_explicit_predict_calls']
        assert not network_attempts
        status = 'passed_software_smoke'
        result = dict(status=status, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      boundary_tests_passed=5, public_example_images=2, qualified_equipment_instances=0,
                      image_summaries=summaries, blank_detections=len(rows(negative)),
                      counts=counts, backend_internal_warmups_not_counted_as_explicit_predict_calls=True,
                      runtime=warmup['runtime'], source_format_contract_passed=True,
                      actual_cpu_model_inference_executed=True, detector_accuracy=None,
                      natural_equipment_accuracy=None, natural_configuration_prior_accuracy=None,
                      natural_semantic_planning_gain=None, real_ROS_communication_executed=False,
                      main_tasks_used=35, main_tasks_cap=36, network_attempts=network_attempts,
                      full_V37_natural_efficacy_gate_passed=False,
                      optional_polars_table_export_not_installed=True,
                      opencv_headless_substitutes_for_GUI_package_for_this_verified_inference_path=True,
                      elapsed_seconds=time.perf_counter()-started)
        write(OUTPUT, OUTPUT / 'result.json', result)
    except Exception as error:
        write(OUTPUT, OUTPUT / 'failure.json', dict(status='failed', error_type=type(error).__name__,
            error=str(error), counts=counts, network_attempts=network_attempts))
        raise
    finally:
        verify_sources(OUTPUT)
        seal(OUTPUT)
        verify_inventory(OUTPUT)
    print(json.dumps(dict(status=status, counts=counts, result_sha256=sha(OUTPUT / 'result.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run:
        run()
    else:
        parser.print_help()
