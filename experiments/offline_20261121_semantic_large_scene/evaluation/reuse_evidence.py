"""Map compatible prior evidence to the current metric without reclassifying it as new."""

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evaluation = ARCHIVE / 'paired_unseen_followup_20261120/evaluation/results/summary.json'
    published = ARCHIVE / 'paired_unseen_followup_20261120/publication/receipt.json'
    data = read(evaluation)
    assert not data['integrity_errors'] and not data['missing_or_partial']
    assert read(published)['original_B11_tree_unchanged']
    source_files = [evaluation, published]
    rows = []
    for layout, runs in data['rows'].items():
        pair = [runs[arm][0] for arm in ('S_visibility', 'G_visibility', 'F_strict')]
        assert len({row['initial_packet_sha256'] for row in pair}) == 1
        assert {row['budget'] for row in pair} == {256}
        for row in pair:
            manifest = Path(row['attempt']) / 'artifact_manifest.json'
            assert sha(manifest) == row['manifest_sha256']
            result_file = manifest.parent / 'result.json'
            assert sha(result_file) == read(manifest)['files_sha256']['result.json']
            source_files.extend((manifest, result_file))
            c = row['coverage']['free_correct_fraction']
            f1 = row['surface_metrics']['f1_05cm']
            rows.append(dict(layout=layout, arm=row['arm'], C_free=c, F1_scene_05cm=f1,
                             J=c*f1, budget=row['budget'], actions=row['paid_actions'],
                             returned=row['returned'], collisions=row['collisions'],
                             stop_reason=row['termination']['reason'],
                             reachable_area_m2=row['coverage']['reference_free_cells']*.2**2,
                             evidence_role='previously_examined_small_layout', source=str(result_file)))
    assert len(rows) == 12
    with (HERE / 'historical_layouts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    q1 = ARCHIVE / 'rapid_evidence_20261108/semantic/execution/protocol.json'
    e1 = ARCHIVE / 'four_room_completion_20261119/matrix_study/scenes/20261120/runs/S/attempt_001'
    source_files.extend((q1, e1 / 'choices.json', e1 / 'steps.json', e1 / 'artifact_manifest.json'))
    decision = read(e1 / 'choices.json')[5]
    assert decision['inputs']['remaining_budget'] == 314
    assert len(decision['outputs']['candidates']) == 12
    receipt = dict(status='verified_historical_scope', metric_rows=12, original_layouts=4,
                   area_range_m2=[min(row['reachable_area_m2'] for row in rows), max(row['reachable_area_m2'] for row in rows)],
                   Q1='Historical fixed-cue positive control with one placement and an earlier coverage convention. Use as a regression or hook example.',
                   E1='Historical parent with the matching 198-action checkpoint. Use original sensor history, selected routes and reference for the declared branches.',
                   B11='Shared J metric and effective F_strict baseline. Four previously examined smaller layouts within 54.88–68.84 square metres of reachable free space.',
                   F_nearest='Excluded from effective comparison after original forced-current-pose candidate prevented translation.',
                   confirmatory_layouts_reused=0,
                   receipts_sha256={str(f): sha(f) for f in source_files})
    (HERE / 'historical_scope.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'new_confirmatory_layouts': 0, 'historical_e1_candidates': 12}))


if __name__ == '__main__':
    main()
