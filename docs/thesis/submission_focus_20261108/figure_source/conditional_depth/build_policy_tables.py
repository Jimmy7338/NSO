#!/usr/bin/env python3
"""Publish tables only after all ten frozen B8 runs are independently evaluated."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ARMS = ('G_fixed','S_fixed','G_uncertainty','S_uncertainty','S_mean_collapse')
LABELS = {'G_fixed':'G fixed','S_fixed':'S fixed','G_uncertainty':'G distribution',
          'S_uncertainty':'S distribution','S_mean_collapse':'S mean collapse'}

def read(p): return json.loads(p.read_text())
def resource(p): return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def csv_write(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    pp=BASE/'audit/policy_rollouts/summary.json';rp=BASE/'audit/particle_replay/summary.json'
    policy=read(pp);replay=read(rp)
    assert policy['status']=='completed_independent_ten_run_evaluation'
    assert not policy['pending'] and not policy['failures']
    assert policy['design']['new_runs_completed']==10
    assert replay['status']=='complete_six_state_five_arm_summary' and replay['completed_states']==6
    assert not replay['pending']
    result=[]
    for layout in ('shelf_left','shelf_right'):
        assert set(policy['rows'][layout])==set(ARMS)
        for arm in ARMS:
            r=policy['rows'][layout][arm];m=r['surface_metrics']
            source=BASE/'rollouts/joint_depth/layouts'/layout/'runs'/arm/'attempt_001/result.json'
            actor=read(source)
            assert actor['status']=='completed_execution' and actor['trajectory_sha256']==r['trajectory_sha256']
            assert r['status']=='completed_independently_evaluated'
            result.append(dict(layout=layout,arm=arm,paid_actions=r['paid_actions'],
                initial_coverage_fraction=r['initial_coverage_fraction'],final_coverage_fraction=r['final_coverage_fraction'],
                free_correct_fraction=r['free_correct_fraction'],first_full_coverage_action=r['first_coverage_action']['1.0'],
                path_length_m=r['path_length_m'],f1_5cm_percent=100*m['f1_05cm'],
                shelf_recall_5cm_percent=100*m['target_surface_metrics']['open31']['recall_05cm'],
                cabinet_recall_5cm_percent=100*m['target_surface_metrics']['closed118']['recall_05cm'],
                f1_10cm_percent=100*m['f1_10cm'],elapsed_s=actor['elapsed_s'],
                decision_wall_s_total=actor['decision_wall_s_total'],global_planning_calls=r['global_planning_calls'],
                collisions=r['collisions'],returned=r['final_pose_equals_initial_anchor'],
                trajectory_sha256=r['trajectory_sha256'],native_TSDF_mesh_sha256=r['native_TSDF_mesh_sha256']))
    csv_write('joint_depth_policy_table.csv',result)
    table=[r'\begin{table}[tbp]',r'\centering\small',r'\setlength{\tabcolsep}{3.5pt}',
        r'\caption{Complete matched conditional-depth policies. All ten runs begin at 42.606\% correct-free coverage, finish 256 logical actions with 100\% coverage, return and incur zero simulated collisions. $A_{100}$ is the first full-coverage action; surface scores use native measured TSDFs at 5\,cm. Elapsed seconds are host computation under three concurrent runs with numerical-library thread limits set to one, not robot execution time. These are two furniture permutations of one inspected floorplan, without stochastic repeats.}',
        r'\label{tab:joint-depth-policy}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
        r'Layout & Policy & $A_{100}$ & Path (m) & F1 (\%) & Shelf (\%) & Cabinet (\%) & Time (s) \\',r'\midrule']
    last=None
    for r in result:
        if last is not None and r['layout']!=last:table.append(r'\midrule')
        layout=('Left' if r['layout']=='shelf_left' else 'Right') if last!=r['layout'] else ''
        table.append(f"{layout} & {LABELS[r['arm']]} & {r['first_full_coverage_action']} & {r['path_length_m']:.1f} & "
            f"{r['f1_5cm_percent']:.3f} & {r['shelf_recall_5cm_percent']:.3f} & {r['cabinet_recall_5cm_percent']:.3f} & {r['elapsed_s']:.1f} \\\\")
        last=r['layout']
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (HERE/'joint_depth_policy_table.tex').write_text('\n'.join(table)+'\n')
    rows=[]
    for arm in ARMS:
        r=replay['equal_state_weight_aggregate'][arm];c=r['mean_candidate'];s=r['selected_means']
        rows.append(dict(arm=arm,predicted_weight_m2=c['predicted_weight_m2'],
            supported_weight_m2=c['true_object_supported_weight_m2'],realized_weight_m2=c['realized_true_object_weight_m2'],
            support_fraction=c['true_object_support_fraction'],rho=r['template_rank_rho_vs_object_net'],
            selected_object_net_new_m2=s['actual_object_net_new_m2'],selected_global_net_new_m2=s['actual_global_net_new_m2'],
            selected_completed_actions=s['completed_actions'],selected_path_length_m=s['path_length_m'],
            selected_coverage_delta_pp=s['coverage_delta_percentage_points']))
    csv_write('joint_depth_replay_table.csv',rows)
    table=[r'\begin{table}[tbp]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
        r'\caption{Five-arm prediction replay on the same six acquired development states and 66 original branches (330 forecasts, no new branch executions). Forecast columns average candidates within each state, then states equally. Supported and realized weights are weighted template expectations in m$^2$, not unique physical surface area. Rank is within-state correlation with actual object net surface gain. The final columns describe actual gains and costs of the chosen branches.}',
        r'\label{tab:joint-depth-replay}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'Policy & Supported & Realized & $\rho$ & Object net & Global net & Actions \\',r'\midrule']
    for r in rows:
        table.append(f"{LABELS[r['arm']]} & {r['supported_weight_m2']:.4f} & {r['realized_weight_m2']:.4f} & {r['rho']:.3f} & "
            f"{r['selected_object_net_new_m2']:.4f} & {r['selected_global_net_new_m2']:.3f} & {r['selected_completed_actions']:.2f} \\\\")
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (HERE/'joint_depth_replay_table.tex').write_text('\n'.join(table)+'\n')
    receipt=dict(status='all_ten_completed_independent_results_only',policy_rows=10,replay_rows=5,
        sources=[resource(pp),resource(rp)],build_script=resource(Path(__file__).resolve()),
        outputs=[resource(HERE/name)for name in ('joint_depth_policy_table.csv','joint_depth_policy_table.tex','joint_depth_replay_table.csv','joint_depth_replay_table.tex')])
    (HERE/'policy_table_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],policy_rows=len(result),replay_rows=len(rows))))

if __name__=='__main__':main()
