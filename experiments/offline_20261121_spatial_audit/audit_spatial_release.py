#!/usr/bin/env python3
"""Check the evaluated point masks, preregistered conclusion, and rendered formats."""

import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from verify_sources import COLD, ROOT, check_sha, read, sha256


EXPECTED_PANELS = {
    'chain13_terminal': 512, 'cycle14_terminal': 512, 'chain15_terminal': 512,
    'cycle16_terminal': 512, 'cycle16_paid256': 256, 'cycle16_paid320': 320,
}


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(measured, recorded):
    if not np.isclose(measured, float(recorded), atol=1e-9, rtol=0):
        raise ValueError(f'Release result {measured} differs from source {recorded}')


def verify_points():
    rows = {r['panel']: r for r in csv_rows(ROOT / 'spatial_pairs.csv')}
    frozen = {(r['layout'], r['comparison']): r for r in csv_rows(COLD / 'measurement/results/contrasts.csv')}
    with np.load(ROOT / 'spatial_pairs.npz', allow_pickle=False) as archive:
        assert set(archive['panels']) == set(rows) == set(EXPECTED_PANELS)
        assert float(archive['threshold_m']) == 0.05
        for panel in archive['panels']:
            row = rows[panel]
            assert int(row['actual_paid']) == EXPECTED_PANELS[panel]
            states = archive[f'{panel}__reference_status']
            points = archive[f'{panel}__reference_xyz']
            weights = archive[f'{panel}__reference_weight_m2']
            labels = archive[f'{panel}__reference_source_id']
            assert len(points) == len(weights) == len(states) == len(labels)
            for group, mask in [('scene', np.ones(len(states), bool)), ('other_aligned', labels == 0),
                                ('fixture_aligned', labels != 0)]:
                for bit, name in enumerate(('neither', 'lost_G_only', 'new_S_only', 'both')):
                    selected = mask & (states == bit)
                    assert selected.sum() == int(row[f'{group}_{name}_count'])
                    close(weights[selected].sum(), row[f'{group}_{name}_m2'])
            if panel == 'chain15_terminal':
                assert not np.isin(states, [1, 2]).any()
                assert row['native_mesh_sha256_G'] == row['native_mesh_sha256_S']
            for bit, arm in [(1, 'G'), (2, 'S')]:
                recall = float((states & bit != 0).mean())
                predicted = archive[f'{panel}__pred_{arm}_matched']
                assert len(predicted) == 12000
                precision = float(predicted.mean())
                f1 = 2 * precision * recall / (precision + recall)
                close(recall, row[f'r_scene_05cm_{arm}'])
                close(precision, row[f'p_scene_05cm_{arm}'])
                close(f1, row[f'F1_scene_05cm_{arm}'])
                close(f1 * float(row[f'C_free_{arm}']), row[f'J_{arm}'])
            if panel.endswith('terminal'):
                original = frozen[row['layout'], 'fullS - fullG']
                for name in ('delta_J_pp', 'delta_F1_scene_05cm_pp', 'delta_C_free_pp'):
                    close(float(row[name]), original[name])
    terminal_deltas = {r['layout']: float(r['delta_J_pp']) for key, r in rows.items() if key.endswith('terminal')}
    gate = read(COLD / 'measurement/results/engineering_gate.json')['engineering_gate']
    qualified = sorted(layout for layout, value in terminal_deltas.items() if value > 0.10)
    assert qualified == gate['qualifying_buildings'] and gate['pass'] is False
    return {'checked_panels': list(rows), 'terminal_deltas_J_pp': terminal_deltas,
            'qualified_buildings': qualified, 'original_gate_preserved': True}


def verify_figures():
    verified = {}
    for name in ('terminal_spatial_matching', 'cycle16_spatial_matching_posthoc'):
        for language in ('zh', 'en'):
            base = ROOT / 'figures' / f'{name}_{language}'
            pdf, png = base.with_suffix('.pdf'), base.with_suffix('.png')
            images = subprocess.check_output(['pdfimages', '-list', str(pdf)], text=True).splitlines()
            assert len(images) == 2, f'Raster image in vector source: {pdf}'
            page = subprocess.check_output(['pdftotext', str(pdf), '-'], text=True)
            tokens = ('256', '320', '512') if 'posthoc' in name else ('13', '14', '15', '16')
            assert all(token in page for token in tokens)
            with Image.open(png) as bitmap:
                assert bitmap.width >= 3000 and bitmap.height >= 1500
                dpi = list(bitmap.info['dpi'])
                assert all(299 <= value <= 301 for value in dpi)
                verified[base.name] = {'pdf_vector_only': True, 'pdf_sha256': sha256(pdf),
                                       'png_sha256': sha256(png), 'png_px': list(bitmap.size), 'png_dpi': dpi}
    return verified


def main():
    sources = read(ROOT / 'source_hashes.json')
    for item in ('frozen_evaluator_and_reference_inputs_sha256', 'audited_primary_run_assets_sha256'):
        for filename, digest in sources[item].items():
            check_sha(Path(filename), digest)
    audit = {'status': 'verified_read_only_spatial_audit', 'source_commit': sources['commit'],
             'measurement': verify_points(), 'figures': verify_figures(),
             'source_hash_ledger_sha256': sha256(ROOT / 'source_hashes.json'),
             'measured_spatial_pairs_sha256': sha256(ROOT / 'spatial_pairs.csv'),
             'manuscript_sha256': sha256(ROOT / 'manuscript_zh_en.md')}
    (ROOT / 'completion_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print('Source inputs, six paired evaluations, original qualification rule, and eight figures verified.')


if __name__ == '__main__':
    main()
