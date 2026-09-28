#!/usr/bin/env python3
"""Audit a new evidence gate on six fixed saved paths, without planning/TSDF.

Each paid RGB-D packet is re-associated. Exact packet/pixel/discrete metadata
and world coordinates within 1e-12 metres bind the archived geometric residual
to the same measurement. BLAS projection roundoff is measured explicitly;
floating coordinate hashes are not claimed to match across runtimes.
Only the feedback gate changes; saved routes are imposed, not simulated as
counterfactual policy outcomes. No endpoint quality improvement is claimed.
"""
import csv
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1
from nso.semantic_reliability import SemanticReliabilityBelief
from nso.observed_instances_v41 import support_digest

BASE = ROOT / 'audit_results/semantic_development_acquisition_20260923/episodes'
OUT = ROOT / 'audit_results/article_stage_20260928/feedback_replay'
CASES = tuple(f'core_P{p:02d}_nom_S_b120_lexicographic' for p in range(6))
COORDINATE_ATOL_M = 1e-12


def compare_associations(current, archived):
    """Permit only declared floating projection roundoff, never mask changes."""
    if len(current) != len(archived):
        raise ValueError('Number of associated instances changed')
    maximum_error = 0.
    changed_hashes = 0
    ignored = {'article_feedback_eligibility', 'geometry_feedback_eligible',
               'marker_anchor_world_m', 'points_world_m', 'support_sha256'}
    for new, old in zip(current, archived):
        if ({k:v for k,v in new.items() if k not in ignored}
                != {k:v for k,v in old.items() if k not in ignored}):
            raise ValueError('Discrete association or measurement identity changed')
        for key in ('marker_anchor_world_m', 'points_world_m'):
            a, b = np.asarray(new[key], dtype=float), np.asarray(old[key], dtype=float)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Invalid associated coordinate arrays')
            error = float(np.max(np.abs(a-b))) if a.size else 0.
            if error > COORDINATE_ATOL_M:
                raise ValueError('Associated coordinates exceed declared tolerance')
            maximum_error = max(maximum_error, error)
        for item in (new, old):
            if support_digest(np.asarray(item['points_world_m'], dtype=float)) != item['support_sha256']:
                raise ValueError('Associated support hash is internally invalid')
        changed_hashes += int(new['support_sha256'] != old['support_sha256'])
    return maximum_error, changed_hashes


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked(root, manifest, name):
    content = (root / name).read_bytes()
    expected = manifest['files'][name]
    if len(content) != expected['bytes'] or sha(content) != expected['sha256']:
        raise ValueError('Saved artifact does not match manifest: ' + name)
    return content


def evaluate_case(case):
    root = BASE / case
    manifest = read(root / 'artifact_manifest.json')
    public = json.loads(checked(root, manifest, 'public_spec.json'))
    workspace = json.loads(checked(root, manifest, 'public_workspace.json'))
    result = json.loads(checked(root, manifest, 'result.json'))
    ledger = ObservedInstancesArticleV1(palette=workspace['marker_palette'],
        structure_names=public['structure_prior']['abstract_structures'],
        class_structure_prior=public['structure_prior']['probability_by_category'],
        maximum_instances=8)
    belief = SemanticReliabilityBelief(geometry_prior=ledger.geometry_prior,
        class_structure_priors=ledger.class_priors, share_across_instances=True)
    rows = []
    old_applied = new_applied = rescued = equal_associations = changed_support_hashes = 0
    maximum_coordinate_error_m = 0.
    began = time.monotonic()
    for step in range(result['acquired_and_saved_packets']):
        packet_name = f'packets/{step:03d}_rgbd.npz'
        content = checked(root, manifest, packet_name)
        import io
        with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in PaidRGBDObservationV40.__dataclass_fields__}
            for name in ('frame_id', 'paid_step'):
                values[name] = values[name].item()
        observation = PaidRGBDObservationV40.from_mapping(values)
        saved = json.loads(gzip.decompress(checked(root, manifest, f'steps/{step:03d}.json.gz')))
        evidence = saved['controller_evidence']
        assert observation.sha256() == evidence['observation_sha256']
        current = ledger.observe(observation)
        old = evidence['association']['accepted']
        try:
            coordinate_error, changed_hashes = compare_associations(current['accepted'], old)
        except ValueError as exc:
            raise ValueError(f'Actual measured association changed in {case} at {step}: {exc}') from exc
        maximum_coordinate_error_m = max(maximum_coordinate_error_m, coordinate_error)
        changed_support_hashes += changed_hashes
        equal_associations += len(old)
        original_associations = {r['instance_id']:r for r in old}
        previous = {r['instance_id']: r for r in evidence['geometry_feedback']}
        for residual in evidence['observed_residual']['results']:
            if not residual['accepted']:
                continue
            key = residual['instance_id']
            source_association = original_associations[key]
            if (residual['observation_sha256'] != observation.sha256()
                    or residual['support_sha256'] != source_association['support_sha256']
                    or residual['frame_id'] != observation.frame_id):
                raise ValueError('Archived residual is not bound to current saved measurement')
            original = previous[key]
            updated = ledger.apply_geometry_feedback(key, frame_id=observation.frame_id,
                observation_sha256=observation.sha256(), log_likelihoods=residual['log_likelihoods'])
            old_applied += int(original['applied'])
            new_applied += int(updated['applied'])
            was_rescued = updated['applied'] and updated['article_feedback_eligibility']['cache_rescue_eligible']
            rescued += int(was_rescued)
            rows.append(dict(case=case, paid_step=step, instance_id=key,
                informative=bool(residual['informative']), old_applied=bool(original['applied']),
                new_applied=bool(updated['applied']), rescued=bool(was_rescued),
                new_reason=updated['reason'], **updated['article_feedback_eligibility']))
        for instance in ledger.snapshot()['instances']:
            key = instance['instance_id']
            label = instance['observed_class'] if (instance['semantic_conditioning_used']
                    and not instance['association_uncertain']) else None
            if key not in belief.instance_ids:
                belief.register(key, label)
            else:
                belief.set_class(key, label)
            belief.replace_log_evidence(key, instance['geometry_log_scores'])
    return dict(case=case, saved_frames=result['acquired_and_saved_packets'],
                measured_associations_numerically_equivalent=equal_associations,
                changed_support_hashes=changed_support_hashes,
                maximum_coordinate_error_m=maximum_coordinate_error_m, old_applied=old_applied,
                new_applied=new_applied, rescued=rescued, elapsed_s=time.monotonic()-began,
                saved_manifest_sha256=sha((root/'artifact_manifest.json').read_bytes()),
                final_new_ledger=ledger.snapshot(), final_new_belief=belief.snapshot()), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Retain each diagnostic execution; do not overwrite it')
    source_names = ['scripts/replay_article_feedback_20260928.py',
        'nso/observed_instances_article_v1.py', 'nso/observed_instances_local.py',
        'nso/observed_instances_v41.py', 'nso/instance_belief_v40.py',
        'nso/exact_support_distance.py', 'nso/semantic_reliability.py']
    source_hashes = {p: sha((ROOT/p).read_bytes()) for p in source_names}
    output.mkdir(parents=True)
    declaration = dict(schema='article.feedback_replay.v1', cases=list(CASES),
        source_sha256=source_hashes, source='saved actual RGB-D and bound archived residuals',
        new_worlds=0, new_policy_trajectories=0, new_tsdf_integrations=0,
        coordinate_tolerance_m=COORDINATE_ATOL_M,
        exact_fields='observation SHA; pixel masks; all discrete association metadata',
        quality_evaluated=False, causal_scope='evidence eligibility on imposed saved routes')
    (output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    results = []
    all_rows = []
    for case in CASES:
        try:
            result, rows = evaluate_case(case)
        except Exception as exc:
            (output/'failure.json').write_text(json.dumps(dict(status='failed', case=case,
                completed_cases=[r['case'] for r in results], error=str(exc)), indent=2)+'\n')
            raise
        results.append(result); all_rows.extend(rows)
        (output/(case+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if not k.startswith('final_')}), flush=True)
    assert source_hashes == {p: sha((ROOT/p).read_bytes()) for p in source_names}
    with (output/'feedback_events.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    summary = dict(**declaration, status='complete', cases_completed=len(results),
        frames=sum(r['saved_frames'] for r in results),
        old_applied=sum(r['old_applied'] for r in results),
        new_applied=sum(r['new_applied'] for r in results),
        rescued=sum(r['rescued'] for r in results))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({p.name:sha(p.read_bytes())
        for p in sorted(output.iterdir()) if p.is_file()},indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
