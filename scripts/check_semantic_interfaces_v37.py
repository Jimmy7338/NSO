#!/usr/bin/env python3
"""Seal V37 offline interface regression checks; no detector or robot execution."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.research_evidence_v31 import freeze, read, seal, sha, verify_inventory, verify_sources, write, write_bytes
from semantic.class_mapping import map_yolo_to_custom
from scripts.preflight_ros1_field_capture_v37 import validate_configuration

OUTPUT = ROOT / 'audit_results/v37_interface_contract_tests_20260920'
TESTS = ('test_semantic_frontend_v37.py', 'test_ros1_field_capture_v37.py')


def legacy_reproductions():
    mapped = map_yolo_to_custom([56, 2])
    try:
        np.zeros((2, 4))[np.array([v >= 0 for v in mapped])]
    except IndexError as error:
        mapping = dict(reproduced=True, source_classes=[56, 2], mapped_classes=mapped,
                       error_type=type(error).__name__, error=str(error))
    else:
        raise AssertionError('expected frozen legacy mixed-category defect was not reproduced')
    tree = ast.parse((ROOT / 'semantic/semantic_map.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SemanticMap2D')
    function = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                 and n.name == '_draw_rect_in_map'))
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, 'semantic/semantic_map.py:extracted_method', 'exec'), namespace)
    counts = np.zeros((3, 8, 8))
    namespace['_draw_rect_in_map'](counts, (4, 4), 0, (2, 2), 1)
    sums = counts.sum(axis=(1, 2)).tolist()
    assert sums == [4, 4, 4], sums
    return dict(mixed_category_index_error=mapping,
                all_category_channel_update=dict(reproduced=True, per_channel_sums=sums,
                    execution='original method AST, annotations/decorator removed, NumPy array'),
                old_sources_modified=False, V35_V36_used_this_legacy_chain=False)


def run():
    if OUTPUT.exists():
        raise FileExistsError('exclusive-create tests: ' + str(OUTPUT))
    assets = ROOT / 'audit_results/v37_semantic_assets_20260918'
    verify_sources(assets)
    verify_inventory(assets)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in assets.iterdir() if p.is_file()}
    sources = [Path(__file__), ROOT / 'nso/semantic_frontend_v37.py',
               ROOT / 'semantic_detector.py', ROOT / 'semantic/semantic_map.py',
               ROOT / 'scripts/record_ros1_rgbd_v3.py',
               ROOT / 'configs/virtual3d/ros1_field_capture_v37.example.json']
    sources += [ROOT / 'tests/virtual3d' / name for name in TESTS]
    OUTPUT.mkdir()
    freeze(OUTPUT, sources, input_sha256=inputs,
           scope='Offline array/configuration regressions only; no physical or model inference')
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in TESTS:
        suite.addTests(loader.discover(str(ROOT / 'tests/virtual3d'), pattern=name))
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    write_bytes(OUTPUT, OUTPUT / 'tests.txt', transcript.getvalue().encode())
    diagnostics = legacy_reproductions()
    preflight = validate_configuration(read(ROOT / 'configs/virtual3d/ros1_field_capture_v37.example.json'))
    assert preflight['ready_to_print_capture_commands'] is False
    assert 'commands' not in preflight
    write(OUTPUT, OUTPUT / 'legacy_diagnostics.json', diagnostics)
    write(OUTPUT, OUTPUT / 'unverified_example_preflight.json', preflight)
    summary = dict(status='passed' if result.wasSuccessful() else 'failed',
                   completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   skipped=len(result.skipped), natural_frontend_inference_executed=False,
                   natural_semantic_accuracy=None, ros_live_communication_executed=False,
                   old_legacy_map_repaired_or_replaced_in_runtime=False,
                   V35_V36_controller_modified=False, new_main_tasks=0,
                   new_world_TSDF_quality_calls=0, main_tasks_used=35, main_tasks_limit=36)
    write(OUTPUT, OUTPUT / 'result.json', summary)
    verify_sources(OUTPUT)
    seal(OUTPUT)
    verify_inventory(OUTPUT)
    print(json.dumps(summary))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run:
        run()
    else:
        parser.print_help()
