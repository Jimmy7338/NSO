#!/usr/bin/env python3
"""Recompute declared saved nominal forecasts; never run policy or mapping.

The nine diagnostic times are fixed before this script's execution, selected
from the completed Ground AISLE G/B observations to investigate repeated
nominal gain. They are not an additional performance experiment or an unbiased
sample of all replans. Outputs preserve this conditional interpretation.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from nso.controller_article_v1 import ArticleObservedResidualV1, ArticleViewPredictorV1
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_article_exposure_v3 import ArticleExposureViewPredictorV3
from nso.view_quality_v42 import _digest


STEPS = (27, 60, 100, 103, 105, 107, 113, 128, 130)
RUNS = ('ground_AISLE_G_b160_n92801', 'ground_AISLE_B_b160_n92801')
STAGE = ROOT / 'audit_results/article_stage_20260928'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path, maximum=16*1024*1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('bounded regular JSON input required: ' + str(path))
    return json.loads(path.read_text())


def bounded_member(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('manifest path escape')
    resolved = root / path
    if resolved.is_symlink() or not resolved.resolve().is_relative_to(root.resolve()):
        raise ValueError('manifest symlink/path escape')
    return resolved


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def verified_episode(run):
    episode = STAGE / 'ground_ablation_v2/episodes' / run
    manifest_path = episode / 'artifact_manifest.json'
    manifest = read(manifest_path)
    review_path = STAGE / 'episode_reviews_ground_v2' / run / 'review.json'
    review = read(review_path)
    if not review.get('all_checks_passed') or not review.get('qualified'):
        raise ValueError('completed reviewed qualified source required')
    if review['input_manifest_sha256'] != sha(manifest_path):
        raise ValueError('review does not bind source manifest')
    count = 0
    for relative, record in manifest['files'].items():
        path = bounded_member(episode, relative)
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('source artifact changed: ' + str(path))
        count += 1
    protocol = read(episode/'protocol.json')
    if protocol['source_sha256'] != manifest['source_sha256']:
        raise ValueError('episode source closure mismatch')
    for relative, expected in protocol['source_sha256'].items():
        if sha(bounded_member(ROOT, relative)) != expected:
            raise ValueError('frozen historical source changed: ' + relative)
    return episode, dict(run_id=run, artifact_manifest_sha256=sha(manifest_path),
        review_path=str(review_path.relative_to(ROOT)), review_sha256=sha(review_path),
        verified_artifact_files=count, frozen_source_sha256=protocol['source_sha256'],
        frozen_source_count=len(protocol['source_sha256']))


def diagnose(episode, run):
    old = ArticleViewPredictorV1(residual=ArticleObservedResidualV1())
    new = ArticleExposureViewPredictorV3(residual=ArticleObservedResidualV1())
    cameras = []
    for path in sorted((episode/'packets').glob('*_rgbd.npz')):
        with np.load(path, allow_pickle=False) as data:
            camera = dict(intrinsic=data['intrinsic'].tolist(),
                world_from_camera=data['world_from_camera'].tolist(),
                width=int(data['rgb'].shape[1]), height=int(data['rgb'].shape[0]))
            cameras.append(dict(**camera, camera_geometry_sha256=_digest(camera),
                frame_id=str(data['frame_id']), paid_step=int(data['paid_step']),
                observation_sha256=read(path.with_name(path.name.replace('_rgbd.npz', '_receipt.json')))['observation_sha256']))
    if [x['paid_step'] for x in cameras] != list(range(len(cameras))):
        raise ValueError('source camera history not consecutive')
    rows = []
    for step in STEPS:
        with gzip.open(episode/'steps'/f'{step:03d}.json.gz', 'rb') as stream:
            raw = stream.read(32*1024*1024+1)
        if len(raw) > 32*1024*1024:
            raise ValueError('bounded saved step exceeded')
        record = json.loads(raw)
        selection = record['decision']['global_selection']
        selected = selection['selected']
        if selected['kind'] != 'direct':
            raise ValueError('declared diagnostic point is not a direct choice')
        view_id = f"{selected['target']['node']}:{selected['target']['heading']}"
        forecast = next(f for f in selection['forecasts'] if f['instance_id'] == 'instance_0001')
        candidate = next(r for r in forecast['candidates'] if r['view_id'] == view_id)
        instance = next(r for r in record['controller_evidence']['association']['instances']
                        if r['instance_id'] == 'instance_0001')
        if candidate['fallback'] or candidate['repeated_view_excluded']:
            raise ValueError('declared candidate eligibility changed')
        data = candidate['candidate']
        view = CandidateViewV40(np.asarray(data['intrinsic']),
            np.asarray(data['world_from_camera']), data['width'], data['height'],
            near_m=data['near_m'], far_m=data['far_m'], view_id=data['view_id'])
        state = dict(plane=forecast['observed_geometry']['plane_fit'])
        support = np.asarray(instance['support_points_world_m'])
        old._paid_cameras = [x for x in cameras if x['paid_step'] <= step]
        new._paid_cameras = list(old._paid_cameras)
        new._exposure_mask_cache.clear()
        old_components, components = old._areas(state, support, view), new._areas(state, support, view)
        names = forecast['structure_names']
        before = np.asarray([np.mean([c['new_surface_area_m2'] for c in old_components
                                      if c['structure'] == name]) for name in names])
        after = np.asarray([np.mean([c['new_surface_area_m2'] for c in components
                                     if c['structure'] == name]) for name in names])
        if not np.allclose(before, candidate['structure_new_surface_area_m2'], atol=1e-12, rtol=0):
            raise ValueError('frozen forecast reproduction failed')
        posterior = np.asarray(forecast['structure_probabilities'])
        old_area, new_area = float(posterior@before), float(posterior@after)
        if not np.isclose(old_area, candidate['expected_new_surface_area_m2'], atol=1e-12, rtol=0):
            raise ValueError('saved expected forecast reproduction failed')
        rows.append(dict(run_id=run, paid_step=step, instance_id='instance_0001',
            target=selected['target'], saved_selected_kind=selected['kind'],
            saved_selected_total_cost=selected['total_cost'], support_points=len(support),
            old_expected_area_m2=old_area, deduplicated_expected_area_m2=new_area,
            overlap_fraction=1-new_area/old_area if old_area else 0.,
            same_center_distinct_paid_views=components[0]['same_center_distinct_paid_views'],
            structure_probabilities=posterior.tolist(), old_structure_areas_m2=before.tolist(),
            deduplicated_structure_areas_m2=after.tolist(), components=components,
            saved_planar_scale_components=[c for c in old_components if c['structure']=='planar'],
            saved_forecast_reproduced_atol_m2=1e-12))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=STAGE/'analysis_v1/same_center_exposure_v3_diagnosis')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(STAGE.resolve()) or output.exists():
        raise ValueError('fresh output under article stage required; no overwrite')
    start = time.monotonic()
    verified = [verified_episode(run) for run in RUNS]
    all_rows = []
    for (episode, _), run in zip(verified, RUNS):
        all_rows.extend(diagnose(episode, run))
    output.mkdir(parents=True)
    sources = {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))}
    for relative in ('nso/view_quality_article_exposure_v3.py', 'tests/test_view_quality_article_exposure_v3.py'):
        sources[relative] = sha(ROOT/relative)
    for relative in sources:
        destination = output/'source_archive'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/relative).read_bytes())
    summary = dict(schema='article.same_center_exposure_diagnosis.v3',
        runs=list(RUNS), declared_paid_steps=list(STEPS), rows=all_rows,
        frozen_forecasts_reproduced=len(all_rows), elapsed_s=time.monotonic()-start,
        new_worlds=0, new_policy_runs=0, new_sensor_queries=0,
        new_tsdf_integrations=0, new_surface_evaluations=0,
        source_bindings=sources, source_episodes=[receipt for _, receipt in verified],
        conclusion_scope='conditional saved-candidate nominal-area diagnosis, not changed route or measured reconstruction benefit',
        selection='nine fixed direct replans selected to investigate repeated nominal gain, not all replans',
        history='all acquired frames through current paid step, including initial frame',
        proxy='deduplicate same-center nominal attempted visibility; not measured surface evidence')
    write(output/'diagnosis.json', summary)
    fields=['run_id', 'paid_step', 'support_points', 'same_center_distinct_paid_views',
            'old_expected_area_m2', 'deduplicated_expected_area_m2', 'overlap_fraction']
    with (output/'candidate_overlap.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows({k: row[k] for k in fields} for row in all_rows)
    manifest = dict(schema='article.exposure_diagnosis_files.v1',
        files={str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(output.rglob('*')) if p.is_file()},
        source_sha256=sources)
    write(output/'manifest.json', manifest)
    print(json.dumps(dict(output=str(output), reproduced=len(all_rows),
                          frozen_sources=[x[1]['frozen_source_count'] for x in verified],
                          elapsed_s=summary['elapsed_s']), sort_keys=True))


if __name__ == '__main__':
    main()
