#!/usr/bin/env python3
"""Export the complete six-state staged diagnostic; refuses partial results."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / 'dual_template_instance_20261120/audit/staged_summary.json'


def main():
    source = json.loads(SOURCE.read_text())
    assert source['completed_states'] == source['declared_states'] == 6
    assert not source['pending_states']
    assert source['status'] == 'complete_instance_six_state_diagnostic_summary'
    assert sum(x['candidates'] for x in source['states']) == 66
    variant_labels = {'B2': 'B2', 'sampling': 'F',
                      'geometry': 'F+M', 'geometry_free': 'F+M+E',
                      'pooling': 'P', 'pooling_geometry': 'P+M',
                      'pooling_applicability': 'P+A', 'pooling_geometry_applicability': 'P+M+A'}
    fields = ['variant', 'model', 'mean_predicted_weight', 'mean_supported_fraction',
              'mean_supported_weight', 'mean_realized_weight', 'mean_rank_rho',
              'rank_defined_states', 'mean_selected_object_net_m2',
              'mean_selected_global_net_m2', 'mean_selected_coverage_delta_pp',
              'mean_selected_action_count', 'G_S_different_winner_states']
    rows, texrows = [], []
    for variant, label in variant_labels.items():
        group = source['equal_state_weight_aggregate'][variant]
        for model in ('G', 'S'):
            r = group['models'][model]
            selected = r['mean_selected']
            row = dict(variant=variant, model=model,
                mean_predicted_weight=r['mean_route_predicted_weight_m2'],
                mean_supported_fraction=r['pooled_route_source_support_fraction'],
                mean_supported_weight=r['mean_route_source_supported_weight_m2'],
                mean_realized_weight=r['mean_route_realized_supported_weight_m2'],
                mean_rank_rho=r['template_rank_rho_vs_object_net'],
                rank_defined_states=r['rank_defined_states'],
                mean_selected_object_net_m2=selected['actual_object_net_new_m2'],
                mean_selected_global_net_m2=selected['actual_global_net_new_m2'],
                mean_selected_coverage_delta_pp=selected['coverage_delta_percentage_points'],
                mean_selected_action_count=selected['completed_actions'],
                G_S_different_winner_states=group['G_S_different_winner_states'])
            rows.append(row)
            rho = '--' if row['mean_rank_rho'] is None else f"{row['mean_rank_rho']:.3f}"
            support = '--' if row['mean_supported_fraction'] is None else f"{100*row['mean_supported_fraction']:.2f}"
            texrows.append(f"{label if model=='G' else ''} & {model} & "
                f"{row['mean_predicted_weight']:.3f} & {support} & "
                f"{row['mean_supported_weight']:.3f} & {row['mean_realized_weight']:.3f} & "
                f"{rho} & {row['mean_selected_object_net_m2']:.3f} & "
                f"{row['mean_selected_action_count']:.1f} " + r"\\")
        if variant != 'pooling_geometry_applicability':
            texrows.append(r'\midrule')
    with (HERE/'staged_diagnostic_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    content = r'''% Generated from all six declared development states; no policy reruns implied.
\begin{table}[!htbp]
\centering\small\setlength{\tabcolsep}{3.2pt}
\caption{All staged fixed-state predictions using the same acquired candidate outcomes. F retains the full quadrature basis; M estimates the measured envelope; E adds relative free-space evidence. P pools acquired instance fragments on the full basis. A includes relative evidence and absolute mass attenuation. Each statistic is first computed within a state, then averaged with equal weight over six dependent states from two furniture permutations. $W$ is predicted weighted template area; $P_s$ is its fraction within 5~cm of source triangles; $W_s$ and $W_r$ are source-supported and subsequently matched, previously unmatched template weights. They are not unique physical areas. $\rho$ is the within-state Spearman association between template gain and actual object net surface gain. $\Delta A_o$ is the true-reference object surface gain estimated for the candidate selected by the unchanged full score; $n_a$ is its actual action count. All 66 fixed candidate branches are retained and reused across versions; no additional branch or independent scene is counted.}
\label{tab:staged-template-geometry}
\begin{tabular}{llrrrrrrr}
\toprule
Stage & Model & $W$ & $P_s$ (\%) & $W_s$ & $W_r$ & $\rho$ & $\Delta A_o$ (m$^2$) & $n_a$ \\
\midrule
''' + '\n'.join(texrows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (HERE/'staged_diagnostic_table.tex').write_text(content)
    (HERE/'table_receipt.json').write_text(json.dumps(dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        all_six_states=True, all_66_candidates=True,
        generated_by=str(Path(__file__).resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    print(json.dumps({'rows':len(rows), 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
