#!/usr/bin/env python3
"""Conditional once-only support diagnostic of immutable V30 saved instances."""
import argparse
import hashlib
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import ROOT, read, write, freeze, verify_sources, verify_inventory, seal, sha
from nso.support_band_v32 import FullSupportBandEvaluatorV32
from scripts.diagnose_saved_outlines_v31 import legacy_sources, reference_assets, SOURCE
from scripts.audit_supported_outline_v31 import no_acquisition_guards

OUT = ROOT/'audit_results/v32_saved_support_band_20260917'
GATE = ROOT/'audit_results/v32_support_band_20260917'
V31 = ROOT/'audit_results/v31_saved_outline_diagnostic_20260917'
PROTOCOL = ROOT/'docs/research/V32_SUPPORT_BAND_PROTOCOL_20260917.md'


def run():
    if OUT.exists(): raise FileExistsError('diagnostic already started; no implicit retry')
    verify_inventory(GATE); verify_sources(GATE)
    if not read(GATE/'result.json')['all_gates_passed']:
        raise ValueError('analytic gates failed; no saved scoring permitted')
    verify_inventory(V31); verify_sources(V31)
    verify_inventory(SOURCE); old_manifest = legacy_sources()
    for i in range(4): verify_inventory(SOURCE/f'case{i:02d}')
    OUT.mkdir(); started = perf_counter(); rows = []
    counts = dict(candidate_evaluations=0, standalone_legacy_evaluations=0, mesh_inputs_read=0,
        worlds=0, sensor_packets=0, mapper_updates=0, TSDF_integrations=0,
        mapper_mesh_extractions=0, new_main_tasks=0)
    try:
        no_acquisition_guards()
        inputs = [SOURCE/'manifest.json', SOURCE/'artifact_hashes.json', SOURCE/'result.json',
            GATE/'result.json', GATE/'artifact_hashes.json', GATE/'manifest.json',
            V31/'scores.json', V31/'result.json', V31/'artifact_hashes.json']
        for i in range(4):
            folder = SOURCE/f'case{i:02d}'
            inputs.extend([folder/'result.json', folder/'artifact_hashes.json', folder/'replay.json'])
            for stage in ('prefix', 'final'):
                inputs.append(folder/f'{stage}_metadata.json')
                inputs.extend(folder/f'{stage}_slot{j}.npz' for j in range(2))
        manifest = freeze(OUT, [Path(__file__), PROTOCOL],
            status='frozen_before_saved_support_scoring', main_tasks_used=16,
            input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
            old_source_count=len(old_manifest['source_sha256']),
            posthoc_diagnostic_only=True, prediction_geometry_must_not_change=True)
        previous = read(V31/'scores.json')
        for i in range(4):
            folder = SOURCE/f'case{i:02d}'; old = read(folder/'result.json')
            evaluator = FullSupportBandEvaluatorV32(reference_assets(old['case']['hypothesis']))
            stages = {}
            for stage in ('prefix', 'final'):
                metadata = read(folder/f'{stage}_metadata.json'); saved = old['stages'][stage]
                if evaluator.reference.reference_signature != saved['window']['reference_signature']:
                    raise ValueError('underlying reference differs from V30')
                meshes = []; geometry_hashes = []
                for slot in range(2):
                    with np.load(folder/f'{stage}_slot{slot}.npz', allow_pickle=False) as npz:
                        vertices = npz['vertices'].copy(); triangles = npz['triangles'].copy()
                    vertices.setflags(write=False); triangles.setflags(write=False)
                    h = hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()
                    if h != saved['full_instance']['submitted_instance_support'][slot]['submitted_mesh']['sha256']:
                        raise ValueError('saved geometry differs from frozen V30')
                    meshes.append(SimpleNamespace(vertices=vertices, triangles=triangles))
                    geometry_hashes.append(h); counts['mesh_inputs_read'] += 1
                physical = saved['window']['main_observed']
                counts['candidate_evaluations'] += 1
                new = evaluator.evaluate(meshes, [x['seed'] for x in metadata['instances']],
                    len(metadata['observed_track_summary']), physical['coverage_2d'],
                    returned=physical['returned'], collisions=physical['collisions'], failed=physical['failed'],
                    paid_actions=18 if stage == 'prefix' else 48, budget=48)
                old31 = previous[i]['stages'][stage]['candidate']
                for tag in ('02cm', '05cm', '10cm'):
                    if abs(new[tag]['v31_region_outline_quality']-old31[tag]['outline_macro_quality']) > 1e-10:
                        raise ValueError('retained V31 region diagnostic differs from old result')
                if new['historical_task_eligible'] != physical['eligible']:
                    raise ValueError('historical physical qualification changed')
                stages[stage] = dict(candidate=new, original_window=saved['window']['main_observed'],
                    original_full_instance=saved['full_instance'],
                    original_v31={tag:old31[tag] for tag in ('02cm', '05cm', '10cm')},
                    prediction_geometry_sha256=geometry_hashes, physical_qualification_inherited=True)
                print(f'case {i} {stage}: support diagnostic complete, unchanged saved geometry', flush=True)
            rows.append(dict(case=old['case'], stages=stages))
        indices = read(SOURCE/'result.json')['observed_rule_case_indices']
        selection = {}
        for tag in ('02cm', '05cm', '10cm'):
            selection[tag] = {}
            for key in ('outline_macro_quality', 'joint_outline'):
                values = [r['stages']['final']['candidate'][tag][key] for r in rows]
                mean = lambda ids: float(np.mean([values[i] for i in ids]))
                fixed_a, fixed_b = mean([0, 2]), mean([1, 3]); best = max(fixed_a, fixed_b)
                selection[tag][key] = dict(always_A=fixed_a, always_B=fixed_b, best_fixed=best,
                    complex_rule=mean(indices['complex']), swapped_rule=mean(indices['swapped']),
                    oracle=float((max(values[:2])+max(values[2:]))/2),
                    complex_relative=mean(indices['complex'])/best-1 if best else None,
                    scope='new support function; ineligible previously seen fixed-route data, not adaptive G/S')
        verify_sources(OUT); verify_sources(GATE); verify_sources(V31); verify_inventory(SOURCE); legacy_sources()
        for rel, digest in manifest['input_sha256'].items():
            if sha(ROOT/rel) != digest: raise ValueError('frozen input changed: '+rel)
        write(OUT, OUT/'scores.json', rows)
        result = dict(status='complete', counts=counts, main_tasks_used=16,
            source_count=len(manifest['source_sha256']), input_count=len(manifest['input_sha256']),
            final_support_Q05=[r['stages']['final']['candidate']['05cm']['outline_macro_quality'] for r in rows],
            final_v31_region_Q05=[r['stages']['final']['original_v31']['05cm']['outline_macro_quality'] for r in rows],
            final_candidate_eligible=[r['stages']['final']['candidate']['eligible'] for r in rows],
            selection=selection, physical_qualification_unchanged=True, observed_geometry_changed=False,
            legacy_region_diagnostic_matches_v31=True, semantic_efficacy_proven=False,
            candidate_ready_as_sole_training_target=False, full_3d_accuracy_certified=False,
            elapsed_seconds=perf_counter()-started)
        write(OUT, OUT/'result.json', result); seal(OUT); print(result, flush=True)
    except BaseException:
        write(OUT, OUT/'failure.json', dict(error=traceback.format_exc(), counts=counts, completed_cases=len(rows)))
        if not (OUT/'scores.json').exists(): write(OUT, OUT/'partial_scores.json', rows)
        if not (OUT/'artifact_hashes.json').exists(): seal(OUT)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run: run()
    else: parser.print_help()
