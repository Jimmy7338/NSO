#!/usr/bin/env python3
"""Publish all candidate/group rewards only after the frozen screen completes."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.audit_semantic_v15_feedback_observability import sha, write, require


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = json.loads((args.analysis / 'result.json').read_text())
    require(result['status'] == 'complete' and result['candidate_outcomes'] == 46,
            'complete frozen 46-outcome analysis required')
    for name, expected in json.loads((args.analysis / 'artifact_hashes.json').read_text()).items():
        require(sha(args.analysis / name) == expected, 'analysis artifact differs')
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = sha(Path(__file__))
    reference_results = result['reference_results']
    steps = [s['action_id'] for s in reference_results[0]['states']]
    require(all([s['action_id'] for s in r['states']] == steps for r in reference_results),
            'reference states do not match')
    figure, axes = plt.subplots(1, len(steps), figsize=(11, 4.5), squeeze=False)
    rows, oracle_rows = [], []
    for si, step in enumerate(steps):
        ax = axes[0, si]
        base = reference_results[0]['states'][si]
        for gi, group in enumerate(base['groups']):
            color = ('#245578', '#b06128', '#487347', '#86568e')[gi % 4]
            means = np.asarray([r['states'][si]['groups'][gi]['mean_candidate_rewards'] for r in reference_results])
            members = [base['structure_seeds'][i] for i in group['world_indices']]
            ax.plot(base['candidate_ids'], means[0], marker='o', markersize=4, color=color,
                label=f"Observed group {group['observed_group']}: " + ', '.join(map(str, members)))
            ax.fill_between(base['candidate_ids'], means.min(axis=0), means.max(axis=0), color=color, alpha=.12)
        for ref in reference_results:
            state = ref['states'][si]
            for wi, seed in enumerate(state['structure_seeds']):
                group = next(g['observed_group'] for g in state['groups'] if wi in g['world_indices'])
                for ci, candidate in enumerate(state['candidate_ids']):
                    rows.append(dict(parent='D12-00', action_id=step, structure_seed=seed,
                        observed_group=group, candidate_id=candidate, reference_seed=ref['reference_seed'],
                        delta_coverage_times_f1_05cm=state['rewards'][wi][ci]))
            oracle_rows.append(dict(action_id=step, reference_seed=ref['reference_seed'],
                geometry_blind_oracle=state['geometry_blind_oracle'],
                observed_semantic_oracle=state['observed_semantic_oracle'],
                latent_world_oracle=state['latent_world_oracle'],
                information_value=state['information_value'], relative_information_value=state['relative_information_value']))
        relative = base['relative_information_value']
        label = 'undefined' if relative is None else f'{100 * relative:.4f}%'
        ax.set_title(f'Decision after action {step}\nObserved information value: {label}')
        ax.set_xlabel('Shared candidate ID')
        ax.set_ylabel('Increment in 2D coverage fraction × global F1@5cm')
        ax.set_xticks(base['candidate_ids'])
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=7, loc='best')
    figure.suptitle('V15 finite-pool diagnostic — one development parent', fontsize=12)
    figure.text(.5, .01, 'Lines: reference seed 2026. Bands: range across three reference samplings, not confidence intervals.\n'
                'Group means use frozen world weights. These are candidate outcomes, not trained semantic-policy scores.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .085, 1, .91))
    figure.savefig(args.output / 'candidate_group_rewards.png', dpi=180)
    figure.savefig(args.output / 'candidate_group_rewards.svg')
    plt.close(figure)
    for name, records in [('candidate_rewards.csv', rows), ('information_values.csv', oracle_rows)]:
        with (args.output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    require(source_hash == sha(Path(__file__)), 'plot source changed')
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(args.output / 'manifest.json', dict(status='complete', source_sha256=source_hash,
        analysis_inventory_sha256=sha(args.analysis / 'artifact_hashes.json'),
        candidate_reference_rows=len(rows), parent_groups=1, semantic_efficacy_proven=False,
        shaded_bands_are_confidence_intervals=False))
    write(args.output / 'artifact_hashes.json', {q.name: sha(q) for q in sorted(args.output.iterdir())
        if q.is_file() and q.name != 'artifact_hashes.json'})


if __name__ == '__main__': main()
