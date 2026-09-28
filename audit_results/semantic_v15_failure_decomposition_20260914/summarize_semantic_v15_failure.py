#!/usr/bin/env python3
"""Descriptive metric decomposition after the frozen V15 information screen."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_semantic_v15_feedback_observability import sha, write, require


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for root in (a.source, a.analysis):
        require(json.loads((root / 'manifest.json').read_text())['status'] == 'complete', 'complete inputs required')
        for n, expected in json.loads((root / 'artifact_hashes.json').read_text()).items():
            require(sha(root / n) == expected, 'artifact changed')
    analysis = json.loads((a.analysis / 'result.json').read_text())
    rows = json.loads((a.source / 'partial.json').read_text())
    require(len(rows) == 46 == analysis['candidate_outcomes'], 'candidate count mismatch')
    measures, choices = [], []
    for ref in analysis['reference_results']:
        rid = ref['reference_seed']
        for state in ref['states']:
            optima = []
            for wi, seed in enumerate(state['structure_seeds']):
                values = state['rewards'][wi]
                optima.append([cid for cid, v in zip(state['candidate_ids'], values) if v == max(values)])
            common = sorted(set.intersection(*(set(x) for x in optima)))
            choices.append(dict(reference_seed=rid, action_id=state['action_id'],
                                per_world_optimal_candidates=optima, shared_world_optimal_candidates=common))
        for r in rows:
            m = r['after'][str(rid)]
            precision, recall = m['precision_05cm'], m['recall_05cm']
            current_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            require(abs(current_f1 - m['f1_05cm']) < 1e-12, 'F1 decomposition differs')
            perfect_precision_f1 = 2 * recall / (1 + recall)
            measures.append(dict(reference_seed=rid, structure_seed=r['structure_seed'],
                action_id=r['action_id'], candidate_id=r['candidate_id'],
                candidate_role=r['intervention']['option']['group'], first_target_reached=r['first_target_reached'],
                coverage_2d=m['coverage_2d'], precision_05cm=precision, recall_05cm=recall,
                f1_05cm=m['f1_05cm'], joint_05cm=m['joint_05cm'],
                surface_error_mean_m=m['surface_error_mean_m'],
                complex_recall_05cm=m['complex_recall_05cm'], new_visible_surface_m2=r['new_visible_surface_m2'],
                total_paid_actions=r['total_paid_actions'], total_path_distance_m=r['total_path_distance_m'],
                precision_only_relative_f1_ceiling=(perfect_precision_f1 / current_f1 - 1.) if current_f1 else None))
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / 'physical_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measures[0])); writer.writeheader(); writer.writerows(measures)
    write(a.output / 'result.json', dict(status='complete', per_reference_state_optima=choices,
        all_worlds_share_an_optimal_candidate_in_every_state_and_reference=all(x['shared_world_optimal_candidates'] for x in choices),
        precision_05cm_range=[min(x['precision_05cm'] for x in measures), max(x['precision_05cm'] for x in measures)],
        maximum_precision_only_relative_f1_ceiling=max(x['precision_only_relative_f1_ceiling'] for x in measures),
        physical_candidate_records=len(rows), reference_metric_rows=len(measures), parent_layouts=1,
        first_target_not_reached=[{k: r[k] for k in ('structure_seed', 'action_id', 'candidate_id')} for r in rows if not r['first_target_reached']],
        role='post-screen descriptive diagnosis; no new metric gate or favorable outcome selection',
        semantic_efficacy_proven=False, training_allowed=False))
    (a.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(a.output / 'manifest.json', dict(status='complete', source_sha256=sha(Path(__file__)),
        source_inventory_sha256=sha(a.source / 'artifact_hashes.json'),
        analysis_inventory_sha256=sha(a.analysis / 'artifact_hashes.json')))
    write(a.output / 'artifact_hashes.json', {q.name: sha(q) for q in sorted(a.output.iterdir())
        if q.is_file() and q.name != 'artifact_hashes.json'})


if __name__ == '__main__': main()
