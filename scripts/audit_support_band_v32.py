#!/usr/bin/env python3
"""Once-only frozen analytic support-band audit; no physical acquisition."""
import argparse
from pathlib import Path
import sys
from time import perf_counter
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import shapely
from nso.research_evidence_v31 import ROOT, write, freeze, verify_sources, seal
from nso.support_band_v32 import FullSupportBandEvaluatorV32, support_band_v32
from nso.supported_outline_v31 import supported_projection_v31
from nso.outline_fixtures_v32 import noise_fixtures_v32
from nso.outline_fixtures_v31 import (MeshFixtureV31, geometry_fixtures_v31,
    reference_meshes_v31, association_fixtures_v31, translated_mesh_v31, empty_mesh_v31)
from scripts.audit_supported_outline_v31 import no_acquisition_guards, as_o3d, sha_arrays
from utils.facility_outline_v23 import PROJECTIONS

OUT = ROOT/'audit_results/v32_support_band_20260917'
PROTOCOL = ROOT/'docs/research/V32_SUPPORT_BAND_PROTOCOL_20260917.md'
ARGS = dict(returned=True, collisions=0, failed=False, paid_actions=0, budget=0)


def asset(mesh, identifier=0):
    v = np.asarray(mesh.vertices)
    return dict(id=identifier, mesh=as_o3d(mesh),
                bounds=[v.min(axis=0)-.00001, v.max(axis=0)+.00001])


def gates_for(rows, counts, ambiguity):
    q = lambda name: rows[name]['candidate']['05cm']['outline_macro_quality']
    gates = {}
    correct = ['box_closed', 'box_vertical', 'box_retriangulated', 'box_vertical_retriangulated',
               'box_duplicate', 'box_vertical_duplicate', 'l_closed', 'l_vertical']
    for name in correct:
        gates['anchor_'+name+'_near_one'] = q('anchor_'+name) >= .99
    gates['equivalent_box_representations'] = max(q('anchor_'+n) for n in correct[:6])-min(q('anchor_'+n) for n in correct[:6]) <= 1e-8
    gates['thin_wall_anchor'] = q('anchor_box_four_thin_walls') >= .95
    gates['one_mm_line_gap_stable'] = abs(q('corner_gap_001mm')-q('corner_gap_000mm')) <= .01
    gates['one_mm_thin_wall_gap_stable'] = abs(q('thin_walls_001mm_opening')-q('thin_walls_continuous_002mm')) <= .01
    gaps = [0, 1, 5, 10, 20, 50, 100, 300]
    for gap in gaps[:5]:
        gates[f'gap_{gap:03d}mm_high_support'] = q(f'corner_gap_{gap:03d}mm') >= .98
    gates['gap_ladder_nonincreasing'] = all(q(f'corner_gap_{b:03d}mm') <= q(f'corner_gap_{a:03d}mm')+1e-8 for a, b in zip(gaps[:-1], gaps[1:]))
    for mm, minimum in [(1, .95), (5, .85), (10, .75), (20, .55)]:
        gates[f'translation_{mm:03d}mm_stable'] = q(f'translate_x_{mm:03d}mm') >= minimum
    for mm, minimum in [(1, .90), (5, .75), (10, .55)]:
        gates[f'face_jitter_{mm:03d}mm_stable'] = q(f'independent_face_jitter_{mm:03d}mm') >= minimum
    gates['whole_side_missing_penalized'] = q('corner_gap_000mm')-q('one_vertical_side_missing') >= .05
    for name in ['box_expanded_030', 'box_shifted_030', 'l_notch_filled', 'l_convex_shortcut',
                 'box_extra_detached_box', 'box_extra_open_strip']:
        baseline = 'l_closed' if name.startswith('l_') else 'box_closed'
        gates['anchor_'+name+'_penalized'] = q('anchor_'+baseline)-q('anchor_'+name) >= .01
    for gap in [10, 50, 200]:
        name = f'neighbors_gap_{gap:03d}mm'
        row = rows[name]['candidate']
        gates[name+'_correct'] = q(name) >= .99 and row['mission_asset_count'] == 2 and row['unique_complete_seed_association']
    gates['wide_wrong_bridge_penalized'] = q('neighbors_gap_200mm')-q('neighbors_gap_200mm_wrong_bridge') >= .005
    gates['empty_zero'] = q('anchor_empty') == q('anchor_all_instances_missing') == 0.
    gates['missing_instance_fixed_denominator'] = q('anchor_one_instance_missing') <= .5
    gates['duplicate_seed_zero'] = q('anchor_duplicate_seed') == 0.
    gates['extra_output_penalty'] = q('anchor_extra_unassigned_instance') <= 2/3+1e-12 and not rows['anchor_extra_unassigned_instance']['candidate']['candidate_output_gate_passed']
    gates['empty_extra_no_penalty'] = abs(q('anchor_empty_extra_duplicate_seed')-q('anchor_both_present')) <= 1e-12
    gates['same_observation_same_prediction_representation'] = all(ambiguity.values())
    gates['no_topology_surface_or_training_certification'] = all(
        not r['candidate']['connectivity_certified'] and not r['candidate']['observed_surface_completion_certified']
        and not r['candidate']['candidate_ready_as_sole_training_target']
        and r['candidate']['completion_fraction'] is None for r in rows.values())
    gates['inputs_unchanged'] = all(r['inputs_unchanged'] for r in rows.values())
    gates['counts_and_two_invalid_rejections'] = (len(rows) == 56 and counts['candidate_evaluations'] == 58
        and counts['expected_invalid_rejections'] == 2 and counts['worlds'] == counts['sensor_packets']
        == counts['mapper_updates'] == counts['TSDF_integrations'] == counts['new_main_tasks'] == 0)
    return gates


def run():
    if OUT.exists():
        raise FileExistsError('audit directory exists; no overwrite or implicit retry')
    OUT.mkdir(); started = perf_counter(); rows = {}
    counts = dict(candidate_evaluations=0, expected_invalid_rejections=0,
        standalone_legacy_evaluations=0, worlds=0, sensor_packets=0, mapper_updates=0,
        TSDF_integrations=0, new_main_tasks=0)
    try:
        no_acquisition_guards()
        manifest = freeze(OUT, [Path(__file__), PROTOCOL,
            ROOT/'docs/research/V32_NOISE_FIXTURES_PROTOCOL_20260917.md',
            ROOT/'docs/research/V32_SUPPORT_BAND_CONTRACT_REVIEW_20260917.md',
            ROOT/'tests/virtual3d/test_support_band_v32.py',
            ROOT/'tests/virtual3d/test_outline_fixtures_v32.py'],
            status='frozen_before_formal_scoring', main_tasks_used=16,
            physical_execution_forbidden=True, versions=dict(numpy=np.__version__, shapely=shapely.__version__),
            preliminary_kernel_tests=5, preliminary_empty_projection_comparisons=1)
        def evaluate(name, assets, meshes, seeds, tracks, facts=None):
            before = [sha_arrays(m) for m in meshes]
            evaluator = FullSupportBandEvaluatorV32(assets)
            counts['candidate_evaluations'] += 1
            result = evaluator.evaluate(meshes, seeds, tracks, 1., **ARGS)
            if before != [sha_arrays(m) for m in meshes]:
                raise AssertionError('submitted arrays changed')
            rows[name] = dict(candidate=result, input_mesh_hashes=before, inputs_unchanged=True,
                facts=facts, qualification_fields_are_analytic_only=True)
        fixtures = noise_fixtures_v32()
        for name, case in fixtures.items():
            evaluate(name, [asset(m, i) for i, m in enumerate(case.references)], case.predictions,
                [dict(observed_seed_xyz=p, source='analytic_fixture_not_observation')
                 for p in case.facts['reference_seed_xyz']], len(case.predictions), case.facts)
        refs = reference_meshes_v31()
        for name, case in geometry_fixtures_v31().items():
            ref = refs[case.reference_name]
            evaluate('anchor_'+name, [asset(ref)], [case.prediction],
                [dict(observed_seed_xyz=np.asarray(ref.vertices).mean(axis=0))], 1, case.expected)
        association = association_fixtures_v31()
        for name, case in association.items():
            assets = [dict(a, mesh=as_o3d(a['mesh'])) for a in case['assets']]
            evaluate('anchor_'+name, assets, case['observed_meshes'], case['seeds'], case['observed_track_count'])
        present = association['both_present']
        assets = [dict(a, mesh=as_o3d(a['mesh'])) for a in present['assets']]
        evaluate('anchor_extra_unassigned_instance', assets,
            (*present['observed_meshes'], translated_mesh_v31(refs['box'], (10., 0., 0.))),
            (*present['seeds'], dict(observed_seed_xyz=[11., 0., .8])), 3)
        evaluate('anchor_empty_extra_duplicate_seed', assets,
            (*present['observed_meshes'], empty_mesh_v31()), (*present['seeds'], present['seeds'][0]), 3)
        invalid = MeshFixtureV31(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]), np.array([[0,1,2]]))
        for meshes, seeds in [([invalid], [dict(observed_seed_xyz=[1.,0.,.8])]),
                              ([refs['box'], invalid], [dict(observed_seed_xyz=[1.,0.,.8]), None])]:
            counts['candidate_evaluations'] += 1
            try:
                FullSupportBandEvaluatorV32([asset(refs['box'])]).evaluate(meshes, seeds, len(meshes), 1., **ARGS)
            except ValueError as error:
                if 'zero-area 3D triangle' not in str(error): raise
                counts['expected_invalid_rejections'] += 1
        a = fixtures['ambiguity_closed_solid_missing_corner'].predictions[0]
        b = fixtures['ambiguity_true_open_shell_same_observation'].predictions[0]
        ambiguity = dict(prediction_arrays_equal=sha_arrays(a) == sha_arrays(b))
        for name, axes in PROJECTIONS.items():
            pa, pb = supported_projection_v31(a, axes), supported_projection_v31(b, axes)
            ambiguity[name+'_area_equal'] = pa['area'].wkb == pb['area'].wkb
            ambiguity[name+'_support_equal'] = pa['support'].wkb == pb['support'].wkb
            for tolerance in (.02, .05, .10):
                ambiguity[f'{name}_band_{tolerance}_equal'] = support_band_v32(pa['support'], tolerance).wkb == support_band_v32(pb['support'], tolerance).wkb
        gates = gates_for(rows, counts, ambiguity)
        verify_sources(OUT)
        write(OUT, OUT/'scores.json', rows)
        result = dict(status='passed' if all(gates.values()) else 'failed_gates',
            gates=gates, all_gates_passed=all(gates.values()), failed_gates=[k for k,v in gates.items() if not v],
            support_quality05={k:r['candidate']['05cm']['outline_macro_quality'] for k,r in rows.items()},
            v31_region_quality05={k:r['candidate']['05cm']['v31_region_outline_quality'] for k,r in rows.items()},
            ambiguity=ambiguity, counts=counts, main_tasks_used=16, source_count=len(manifest['source_sha256']),
            elapsed_seconds=perf_counter()-started, semantic_efficacy_proven=False,
            candidate_ready_as_sole_training_target=False, full_3d_accuracy_certified=False,
            scope='fixed analytic exterior-support applicability; new task diagnostic, not original region Q')
        write(OUT, OUT/'result.json', result); seal(OUT)
        print(dict(status=result['status'], gates=len(gates), failed_gates=result['failed_gates'],
                   counts=counts, elapsed_seconds=result['elapsed_seconds']), flush=True)
    except BaseException:
        write(OUT, OUT/'failure.json', dict(error=traceback.format_exc(), counts=counts, completed=list(rows)))
        if not (OUT/'scores.json').exists(): write(OUT, OUT/'partial_scores.json', rows)
        if not (OUT/'artifact_hashes.json').exists(): seal(OUT)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run: run()
    else: parser.print_help()
