#!/usr/bin/env python3
"""Extract sealed AISLE B/G macro receipts; never run sensing or evaluation."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_article_ground_comparisons_20260928 as analysis

STEPS = (27, 60, 100, 103, 105, 107, 113, 128, 130)


def explain(output):
    inputs = analysis.old.Inputs()
    protocol_path = analysis.ROOT / 'configs/virtual3d/article_ground_ablation_v2_20260928.json'
    protocol = inputs.json(protocol_path)
    protocol_sha = analysis.old.digest(inputs.read(protocol_path))
    entries, phase = analysis.ledger_entries(inputs, protocol, protocol_sha)
    rows, frames, components = [], [], []
    for method in ('B', 'G'):
        run = f'ground_AISLE_{method}_b160_n92801'
        entry = entries[run]
        row = analysis.blank_row('GroundOn', run, protocol['slots'][run], entry)
        ready = analysis.load_on(inputs, phase, analysis.STAGE / 'episode_reviews_ground_v2', row, entry, protocol_sha)
        analysis.require(ready is not None and row['review_passed'], 'terminal independently reviewed episode')
        episode, manifest = ready
        def read(name):
            return analysis.old.sealed_read(inputs, episode, manifest, name)
        result = json.loads(read('result.json'))
        logs = [json.loads(gzip.decompress(read(f'steps/{s:03d}.json.gz')))
                for s in range(result['acquired_and_saved_packets'])]
        packets = [json.loads(read(f'packets/{s:03d}_receipt.json'))['execution'] for s in range(len(logs))]
        def instance(step):
            return next(x for x in logs[step]['controller_evidence']['association']['instances'] if x['instance_id'] == 'instance_0001')
        for step in STEPS:
            log = logs[step]
            decision = log['decision']
            selected = decision['global_selection']['selected']
            analysis.require(selected['kind'] == 'direct', 'declared selected direct goal')
            view_id = selected['target']['node'] + ':' + str(selected['target']['heading'])
            end = next(s for s in range(step + 1, len(logs))
                       if logs[s]['controller_evidence']['completed_macro_id'] == decision['macro_id'])
            analysis.require(packets[end]['action'] == 'observe', 'macro ends in actual paid observe receipt')
            candidates = next(f['candidates'] for f in decision['global_selection']['forecasts'] if f['instance_id'] == 'instance_0001')
            forecast = next(c for c in candidates if c['view_id'] == view_id)
            start_instance, end_instance = instance(step), instance(end)
            valid = [logs[s] for s in range(step + 1, end + 1)]
            actions = Counter(packets[s]['action'] for s in range(step + 1, end + 1))
            rows.append(dict(run_id=run, method=method, selection_step=step, completed_paid_step=end,
                macro_id=decision['macro_id'], view_id=view_id, actual_completed_xyyaw=packets[end]['pose_xyyaw_rad'],
                expected_gain=selected['expected_gain'], cabinet_expected_new_surface_area_m2=forecast['expected_new_surface_area_m2'],
                repeated_view_excluded=forecast['repeated_view_excluded'], matching_paid_views=forecast['matching_paid_views'],
                support_points_before=len(start_instance['support_points_world_m']), support_points_after=len(end_instance['support_points_world_m']),
                support_growth=len(end_instance['support_points_world_m'])-len(start_instance['support_points_world_m']), support_capacity=4096,
                support_sha256_before=start_instance['support_sha256'], support_sha256_after=end_instance['support_sha256'],
                feedback_count_before=start_instance['geometric_feedback_frames'], feedback_count_after=end_instance['geometric_feedback_frames'],
                novel_support_qualified_frames_before=start_instance['novel_support_frames'], novel_support_qualified_frames_after=end_instance['novel_support_frames'],
                geometry_log_scores_before=start_instance['geometry_log_scores'], geometry_log_scores_after=end_instance['geometry_log_scores'],
                macro_paid_actions=end-step, macro_actions=dict(actions),
                mapper_newly_known_cells_during_macro=sum(x['mapper']['newly_known_cells'] for x in valid),
                mapper_newly_known_cells_on_final_observe=logs[end]['mapper']['newly_known_cells'],
                tsdf_integrations_during_macro=sum(x['mapper']['tsdf_integrated'] for x in valid),
                valid_depth_pixels_on_final_observe=logs[end]['mapper']['valid_depth_pixels'],
                measured_new_3d_surface_area=None, measured_quality_change=None))
            for component in forecast['components']:
                components.append(dict(run_id=run, method=method, selection_step=step, view_id=view_id, **component))
        for step in sorted(set(STEPS) | {r['completed_paid_step'] for r in rows if r['run_id'] == run}):
            i = instance(step);log = logs[step];e = log['controller_evidence']
            accepted = [x for x in e['association']['accepted'] if x['instance_id'] == 'instance_0001']
            frames.append(dict(run_id=run, paid_step=step, actual_action=packets[step]['action'],
                actual_xyyaw=packets[step]['pose_xyyaw_rad'], support_count=len(i['support_points_world_m']),
                support_sha256=i['support_sha256'], association_uncertain=i['association_uncertain'],
                feedback_count=i['geometric_feedback_frames'], novel_support_frames=i['novel_support_frames'],
                accepted=[{k:x.get(k) for k in ('component_pixels','no_new_support','novel_support_voxels',
                    'geometry_feedback_eligible','article_feedback_eligibility')} for x in accepted],
                newly_known_cells=log['mapper']['newly_known_cells'], tsdf_integrated=log['mapper']['tsdf_integrated'],
                tsdf_integration_count=log['mapper']['tsdf_integration_count'], valid_depth_pixels=log['mapper']['valid_depth_pixels']))
    inputs.unchanged()
    analysis.require(not output.exists(), 'exclusive diagnosis output')
    output.mkdir(parents=True)
    def save(name, value):
        with (output / name).open('xb') as stream:
            stream.write(analysis.old.canonical(value))
    analysis.old.write_csv(output / 'macro_receipts.csv', rows, ['run_id','selection_step'])
    analysis.old.write_csv(output / 'nominal_structure_scale_components.csv', components, ['run_id','selection_step','structure','scale'])
    save('frame_receipts.json', frames)
    save('summary.json', dict(schema='article.paid_repeat_observation_diagnosis.v1', rows=rows,
        new_worlds=0, new_tsdf_integrations=0, new_quality_evaluations=0,
        field_limits='newly_known_cells is the stored mapper count, not newly free cells or measured 3D surface. Support growth is associated voxel-cache growth, not exact physical area. Paid TSDF integration does not by itself prove quality gain. Forecasts are saved nominal model predictions, not GT. Completion uses actual completed_macro_id and paid observe receipt. B/G are paired descriptive results; S is not inspected.'))
    files={p.name:dict(bytes=p.stat().st_size,sha256=analysis.old.digest(p.read_bytes())) for p in output.iterdir()}
    save('manifest.json', dict(files=files, input_sha256=inputs.pins,
        source_sha256={str(Path(p).relative_to(analysis.ROOT)):analysis.old.digest(Path(p).read_bytes())
                       for p in (__file__, analysis.__file__, analysis.old.__file__)},
        new_worlds=0, new_quality_evaluations=0))
    print(json.dumps(dict(output=str(output), macros=len(rows), frames=len(frames))))


if __name__ == '__main__':
    explain(analysis.STAGE / 'analysis_v1/ground_repeat_observation_diagnosis')
