#!/usr/bin/env python3
"""Export the complete separately declared clean-consumption six-run matrix."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'audit/sampling_rollouts/summary.json'


def main():
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'completed_independent_six_run_evaluation'
    assert source['design']['new_runs_planned'] == source['design']['new_runs_completed'] == 6
    assert source['pending'] == source['failures'] == []
    rows, texrows = [], []
    for layout, label in (('shelf_left','Left'), ('shelf_right','Right')):
        for arm in ('G','S','S_no_measured_consumption'):
            r = source['rows'][layout][arm]
            m = r['surface_metrics']
            assert r['actual_template_point_cap_activations'] == 0
            assert r['history_zero_weight_full_basis_verified']
            row = dict(layout=layout, arm=arm, coverage_percent=100*r['final_coverage_fraction'],
                first_100_coverage_action=r['first_coverage_action'].get('1.0'),
                paid_actions=r['paid_actions'], path_length_m=r['path_length_m'],
                F1_5cm_percent=100*m['f1_05cm'], F1_10cm_percent=100*m['f1_10cm'],
                open31_recall_5cm_percent=100*m['target_surface_metrics']['open31']['recall_05cm'],
                closed118_recall_5cm_percent=100*m['target_surface_metrics']['closed118']['recall_05cm'],
                collisions=r['collisions'], return_to_anchor=r['final_pose_equals_initial_anchor'],
                former_cap_exceeding_calls=r['S_calls_exceeding_former_cap'],
                trajectory_sha256=r['trajectory_sha256'], native_mesh_sha256=r['native_TSDF_mesh_sha256'])
            rows.append(row)
            mode = arm if arm in ('G','S') else r'S$^{-m}$'
            threshold = '--' if row['first_100_coverage_action'] is None else str(row['first_100_coverage_action'])
            texrows.append(f"{label if arm=='G' else ''} & {mode} & "
                f"{row['coverage_percent']:.2f} & {threshold} & {row['paid_actions']} & "
                f"{row['path_length_m']:.1f} & {row['F1_5cm_percent']:.2f} & "
                f"{row['F1_10cm_percent']:.2f} & {row['open31_recall_5cm_percent']:.2f} & "
                f"{row['closed118_recall_5cm_percent']:.2f} " + r'\\')
        if layout=='shelf_left': texrows.append(r'\midrule')
    with (HERE/'sampling_policy_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = r'''% Generated only after all six declared runs pass independent evaluation.
\begin{table}[!htbp]
\centering\small\setlength{\tabcolsep}{3.1pt}
\caption{Complete full-quadrature policy comparison and measured-consumption ablation. S$^{-m}$ disables only measured-surface reward consumption; valid-free rejection and B2 visibility remain. $C$ is final reachable-cell coverage, $a_{100}$ its first 100\% action, $n_a$ paid actions and $d$ traveled distance. F1 and object recall use the unchanged native-mesh evaluator; $R_o$/$R_c$ denote open-shelf/closed-cabinet recall at 5~cm. These are six deterministic reruns on the same two permutations, not new independent scenes. Historical C/J controls are reused from Table~S10.}
\label{tab:clean-consumption-policy}
\begin{tabular}{llrrrrrrrr}
\toprule
Layout & Mode & $C$ (\%) & $a_{100}$ & $n_a$ & $d$ (m) & F1$_5$ (\%) & F1$_{10}$ (\%) & $R_o$ (\%) & $R_c$ (\%) \\
\midrule
''' + '\n'.join(texrows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (HERE/'sampling_policy_table.tex').write_text(text)
    receipt=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        all_six_runs=True,generated_by=str(Path(__file__).resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'policy_table_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'rows':len(rows), 'source_sha256':receipt['source_sha256']}))


if __name__=='__main__':main()
