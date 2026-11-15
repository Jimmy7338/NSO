#!/usr/bin/env python3
"""Read only actual frozen calibration events; no geometry replay or ROS calls.

Scores the stored pre-action posterior and the existing disabled-feedback
probability (.5) on the SAME measured prediction-mask outcomes. This is an
off-policy prediction comparison, not a counterfactual exploration trial.
"""
import argparse
import ast
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path('/home/ouc/nso_robot')
WORK = ROOT / 'workspaces/nso_hybrid_20261029/nso_bridge'
EPISODES = [
    ('S3', 'NSO_primitive_feedback_fresh_S_20261115_01'),
    ('G3', 'NSO_primitive_feedback_fresh_G_20261115_01'),
    ('S26', 'NSO_primitive_optimized_S_20261115_01'),
    ('L19', 'NSO_exterior_optimized_S_20261115_01'),
]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def exact_original_calibrator():
    """Compile just preserved classes; never import model/device/robot code."""
    namespace = dict(np=np, deepcopy=deepcopy, hashlib=hashlib)
    sources = []
    for path, name in [
        (WORK / 'joint_vendor/nso/observed_gain_calibration_v10_1.py',
         'ObservedGainCalibration'),
        (WORK / 'nso_bridge/exploration_joint_policy.py',
         'ExplorationObservedGainCalibration'),
    ]:
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(path), 'exec'), namespace)
        sources.append(dict(path=str(path), sha256=sha(path), class_name=name))
    return namespace['ExplorationObservedGainCalibration'], sources


def load_records(report):
    choices = [(p, json.loads(p.read_text()))
               for p in (report / 'joint/primitive_feedback').rglob('records.json')]
    path, records = max(choices, key=lambda item: (len(item[1]), str(item[0])))
    assert [r['action_id'] for r in records] == list(range(len(records)))
    # The latest manifest is a prefix extension; do not multiply repeated copies.
    assert all(rows == records[:len(rows)] for _, rows in choices)
    return path, records, len(choices)


def load_verified_arrays(item, checks, role, action_id):
    path = Path(item['path'])
    digest = sha(path)
    assert digest == item['sha256'], (role, action_id, str(path))
    checks.append(dict(role=role, action_id=action_id, path=str(path), sha256=digest))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def metric(rows, sensor):
    values = [row['channels'][sensor] for row in rows]
    nonempty = [v for v in values if v['predicted_cells'] > 0]
    total_n = sum(v['predicted_cells'] for v in values)
    total_s = sum(v['realized_predicted_cells'] for v in values)
    outside = sum(v['unpredicted_realized_cells'] for v in values)
    answer = dict(
        completed_actions=len(values), nonempty_prediction_actions=len(nonempty),
        zero_prediction_action_ids=[r['action_id'] for r in rows
                                    if r['channels'][sensor]['predicted_cells'] == 0],
        predicted_cell_events=total_n, realized_predicted_cell_events=total_s,
        unrealized_predicted_cell_events=total_n-total_s,
        unpredicted_realized_cell_events=outside,
        actual_new_cell_events=total_s+outside,
        realized_in_prediction_fraction=(total_s / (total_s+outside)
                                         if total_s+outside else None),
        event_counts_are_not_independent_samples=True,
        prediction_metrics_have_nonempty_support=bool(nonempty),
        units=dict(logloss='natural logarithm, nats per predicted cell event',
                   gain_error='cell events within frozen predicted mask'),
    )
    for name in ['online', 'fixed_half']:
        sum_brier = sum(v[name]['brier_sum'] for v in values)
        sum_nll = sum(v[name]['logloss_sum'] for v in values)
        answer[name] = dict(
            cell_event_weighted_brier=sum_brier/total_n if total_n else None,
            cell_event_weighted_logloss=sum_nll/total_n if total_n else None,
            macro_action_brier_nonempty=(sum(v[name]['brier_sum']/v['predicted_cells']
                                            for v in nonempty)/len(nonempty)
                                         if nonempty else None),
            macro_action_logloss_nonempty=(sum(v[name]['logloss_sum']/v['predicted_cells']
                                              for v in nonempty)/len(nonempty)
                                           if nonempty else None),
            gain_count_MAE_all_completed=(sum(v[name]['gain_abs_error'] for v in values)
                                          /len(values)),
            gain_count_MSE_all_completed=(sum(v[name]['gain_square_error'] for v in values)
                                          /len(values)),
            gain_count_MAE_nonempty=(sum(v[name]['gain_abs_error'] for v in nonempty)
                                      /len(nonempty) if nonempty else None),
            gain_count_MSE_nonempty=(sum(v[name]['gain_square_error'] for v in nonempty)
                                      /len(nonempty) if nonempty else None),
            total_predicted_in_mask_gain=sum(v[name]['predicted_gain'] for v in values),
            absolute_total_gain_count_error=abs(sum(v[name]['predicted_gain'] for v in values)-total_s),
            squared_total_gain_count_error=(sum(v[name]['predicted_gain'] for v in values)-total_s)**2,
        )
    answer['relative_reduction_online_vs_fixed_half'] = {}
    for key, baseline in answer['fixed_half'].items():
        if key == 'total_predicted_in_mask_gain':
            continue
        value = answer['online'][key]
        answer['relative_reduction_online_vs_fixed_half'][key] = (
            1-value/baseline if baseline is not None and baseline != 0 else None)
    return answer


def evaluate_episode(label, report, cls):
    record_path, records, manifest_count = load_records(report)
    checks = []
    for record in records:
        digest = sha(record['original_path'])
        assert digest == record['original_sha256']
        checks.append(dict(role='raw_original', action_id=record['action_id'],
                           path=record['original_path'], sha256=digest))
    shape = tuple(json.loads((report/'joint_configuration.json').read_text())['configuration']['shape'])
    gain = cls(shape, feedback_enabled=True, consume_predicted_attempts=False)
    rows = []
    for record in records[1:]:
        aid = record['action_id']
        outcome = record['original_prediction_and_actual_feedback']
        p_item, a_item = record['frozen_prediction_inputs'], record['frozen_actual_feedback_inputs']
        before = load_verified_arrays(p_item, checks, 'frozen_prediction', aid)
        after = load_verified_arrays(a_item, checks, 'frozen_actual_feedback', aid)
        assert set(before) == {'radar_prediction', 'camera_prediction', 'known_before', 'camera_before'}
        assert set(after) == {'belief_after', 'camera_seen_after'}
        assert all(a.shape == shape for a in [*before.values(), *after.values()])
        assert all(a.dtype == bool for a in before.values())
        assert after['camera_seen_after'].dtype == bool
        expected_p = gain.snapshot()['posterior_mean']
        assert outcome['predicted_gain']['posterior_before'] == expected_p
        assert p_item['posterior_before'] == expected_p
        assert p_item['original_source_before_sha256'] == records[aid-1]['original_sha256']
        assert outcome['frozen_prediction_inputs'] == p_item
        assert outcome['frozen_actual_feedback_inputs'] == a_item
        # Execute the actual preserved observe definition on saved masks.
        gain._pending[aid] = dict(radar=before['radar_prediction'],
                                 camera=before['camera_prediction'],
                                 known_before=before['known_before'],
                                 camera_before=before['camera_before'])
        actual = gain.observe(SimpleNamespace(belief=after['belief_after'],
                                             camera_seen=after['camera_seen_after']), action_id=aid)
        assert actual == outcome['actual_gain_event']
        assert gain.snapshot() == outcome['gain_snapshot']
        row = dict(action_id=aid, action=outcome['action'], ros_status=outcome['ros_status'],
                   primitive_arrival_verified=outcome['primitive_arrival_verified'],
                   reason=outcome['reason'], channels={})
        for sensor in ['radar', 'camera']:
            event = actual[sensor]
            n, s, f = (event[k] for k in ['predicted_cells', 'realized_predicted_cells',
                                         'unrealized_predicted_cells'])
            assert n == s+f
            assert n == outcome['predicted_gain']['predicted_'+sensor+'_cells']
            # Domain certificate: predicted coverage cells were unknown before.
            assert not np.any(before[sensor+'_prediction'] & before['known_before'])
            channel = dict(**event, posterior_before=expected_p[sensor],
                           posterior_after=actual['posterior_after'][sensor],
                           realized_predicted_fraction=s/n if n else None)
            for tag, p in [('online', expected_p[sensor]), ('fixed_half', .5)]:
                assert 0 < p < 1
                error = p*n-s
                channel[tag] = dict(probability=p, predicted_gain=p*n,
                    brier_sum=s*(1-p)**2+f*p*p,
                    logloss_sum=-s*math.log(p)-f*math.log(1-p),
                    gain_abs_error=abs(error), gain_square_error=error*error)
            row['channels'][sensor] = channel
        rows.append(row)
    pending = []; accepted_observations = []
    with (report/'joint/decisions.jsonl').open() as stream:
        for line in stream:
            entry = json.loads(line)
            request, response = entry.get('request') or {}, entry.get('response') or {}
            if request.get('method') == 'observe_primitive_result' and response.get('ok'):
                response_result = response.get('result') or {}
                if response_result.get('accepted'):
                    accepted_observations.append(dict(action_id=request['params']['action_id'],
                        actual_primitive_attempts=response_result.get('summary',{}).get('actual_primitive_attempts'),
                        verified=response_result.get('primitive_outcome',{}).get('primitive_arrival_verified')))
            if request.get('method') == 'next_action' and response.get('ok'):
                item = response['result'].get('pending')
                if item and item['action_id'] >= len(records):
                    pending.append(dict(action_id=item['action_id'], action=item['action'],
                                         measured_completed_feedback=False))
    config = json.loads((report/'joint_configuration.json').read_text())
    summary = json.loads((report/'summary.json').read_text())
    assert [o['action_id'] for o in accepted_observations] == list(range(len(records)))
    return dict(label=label, report=str(report), records_manifest=str(record_path),
        records_manifest_sha256=sha(record_path), repeated_prefix_manifests_not_counted=manifest_count-1,
        actual_observation_frames=len(records), bootstrap_frames_not_scored=1,
        completed_primitive_attempts=len(rows),
        ROS_failed_action_ids=[r['action_id'] for r in rows if r['ros_status'] != 3],
        true_arrivals=sum(r['primitive_arrival_verified'] for r in rows),
        pending_without_completed_feedback=pending,
        calibration_prior=dict(alpha=1, beta=1, initial_probability=.5),
        final_posterior=gain.snapshot(),
        geometry_profile=('optimized_history_after_frozen_feedback' if label in ['S26','L19']
                          else 'bootstrap_frozen_prediction_epoch'),
        completed_worker_observations=accepted_observations,
        count_reconciliation=dict(supervisor_primitive_attempts=summary['primitive_attempts'],
            committed_worker_attempts=len(rows),
            supervisor_count_minus_committed_worker=summary['primitive_attempts']-len(rows),
            raw_supervisor_summary_preserved=True,
            note='Calibration uses every actual worker-committed frozen mask event; a supervisor cleanup count may lag an accepted in-flight worker response.'),
        source_initialization_scope=config['calibration'].get('initialization_scope'),
        physical_summary=summary,
        metrics={sensor: metric(rows, sensor) for sensor in ['radar','camera']},
        actions=rows, verified_source_artifacts=checks,
        verification=dict(all_archive_and_original_hashes_match=True,
                          exact_original_observe_reproduces_all_event_fields=True,
                          every_posterior_before_equals_prior_plus_only_preceding_outcomes=True,
                          every_frozen_before_source_matches_previous_original_source=True,
                          all_final_Beta_events_and_snapshots_exact=True,
                          no_bootstrap_or_pending_action_scored=True))


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.8), gridspec_kw={'width_ratios':[1.1,1.4,1.4]})
    ax = axes[0]
    x = np.arange(4); width = .34
    baseline = [e['metrics']['radar']['fixed_half']['cell_event_weighted_brier'] for e in result['episodes']]
    online = [e['metrics']['radar']['online']['cell_event_weighted_brier'] for e in result['episodes']]
    ax.bar(x-width/2, baseline, width, label='Fixed $p=0.5$', color='#A6A9AE', edgecolor='white')
    ax.bar(x+width/2, online, width, label='Online, past events only', color='#167C80', edgecolor='white')
    ax.set_xticks(x, ['S3','G3','S26','Later L19']);ax.set_ylabel('In-mask Brier score (lower is better)')
    ax.set_ylim(0,.37);ax.set_title('(a) All recorded episodes', loc='left', fontsize=10)
    for at, val in zip(x+width/2, online):ax.text(at,val+.007,f'{val:.3f}',ha='center',fontsize=8)
    ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.0))
    ax.text(.5,-.23,'3 + 3 + 26 + 19 dependent action observations\nAll four recorded episodes retained',
            transform=ax.transAxes,ha='center',va='top',fontsize=8,color='#555555')
    for ax, ep, panel in zip(axes[1:],result['episodes'][2:],['b','c']):
        rows=ep['actions'];x=np.array([r['action_id'] for r in rows])
        vals=[r['channels']['radar'] for r in rows]
        realized=np.array([v['realized_predicted_fraction'] if v['predicted_cells'] else np.nan for v in vals])
        probability=np.array([v['posterior_before'] for v in vals])
        ax.axhline(.5,color='#8A8E94',ls='--',lw=1.3,label='Fixed $p=0.5$')
        ax.plot(x,probability,'o-',color='#167C80',markersize=3,lw=1.5,label='Online, earlier outcomes only')
        ax.scatter(x,realized,s=22,color='#C5762B',marker='s',label='Measured in-mask fraction',zorder=4)
        zeros=x[np.array([v['predicted_cells']==0 for v in vals])]
        ax.scatter(zeros,np.full(len(zeros),-.055),marker='v',s=22,color='#777777',label='Empty predicted mask')
        failed=[r['action_id'] for r in rows if r['ros_status']!=3]
        for aid in failed:ax.axvspan(aid-.35,aid+.35,color='#C5762B',alpha=.12,lw=0)
        if ep['label']=='L19':
            ax.axvline(10.5,ls=':',color='#777777',lw=1)
            ax.text(10.7,.035,'After v1 bundle freeze',fontsize=7,color='#666666',rotation=90,va='bottom')
        ax.set_ylim(-.10,1.04);ax.set_xlim(.4,max(x)+.6)
        ax.set_xticks([1,5,10,15,20,25,26]if ep['label']=='S26'else[1,5,10,15,19])
        ax.set_xlabel('Completed primitive attempt')
        ax.set_ylabel('Probability / measured in-mask fraction')
        ax.set_title(f'({panel}) '+('Optimized S26'if ep['label']=='S26'else'Later episode L19'),loc='left',fontsize=10)
        ax.legend(frameon=False,fontsize=7.0,loc='upper left',ncol=1)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.055,right=.99,bottom=.24,top=.88,wspace=.35)
    fig.savefig(output/'prequential_radar_yield.pdf',metadata={'Title':'Actual frozen-event online yield calibration'})
    fig.savefig(output/'prequential_radar_yield.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cls,sources=exact_original_calibrator()
    freeze=json.loads((Path(__file__).resolve().parent/'prior_freeze_receipt.json').read_text())
    for name,spec in freeze['prior_files'].items():assert sha(spec['path'])==spec['sha256']
    result=dict(schema='actual_frozen_prequential_yield/2',analysis_script_sha256=sha(__file__),
        prior_three_episode_bundle=freeze,
        read_only_source_analysis=True,new_robot_actions=0,new_feedback_events=0,
        source_method_definitions=sources,
        baseline=dict(probability=.5,mechanism='Preserved ObservedGainCalibration starts Beta(1,1); feedback_enabled=False keeps alpha/beta fixed. Only its probability prediction is compared on identical actual frozen events; not a physical no-feedback-policy run.'),
        equations=dict(n='count(predicted_mask)',
            radar_actual='(belief_after != -1) & ~known_before',
            camera_actual='camera_seen_after & ~camera_before & ~known_before',
            s='count(predicted_mask & actual)',f='n-s',
            outside='count(actual & ~predicted_mask)',
            p_online='(1+sum previous s)/(2+sum previous n), stored posterior_before checked exactly',
            Brier='[s*(1-p)^2 + (n-s)*p^2]/n',
            logloss='-[s*ln(p)+(n-s)*ln(1-p)]/n',
            expected_conditional_gain='p*n; target is s, not s+outside'),
        unit_of_analysis='Four recorded episodes: one short S3/G3 functional pair, one optimized S26 continuation sharing the G3-created RTAB map, and one later analyzed L19 sequence seeded by the S26 RTAB map. Not independent scenes; no significance claim from action/cell counts.',
        limitations=[
            'The channel named radar tests new known occupancy cells after combined RGB-D/scan fusion, not isolated lidar detection accuracy.',
            'Frozen before/after prediction-epoch arrays are used; optimized-history map-area growth is not relabelled as observed gain.',
            'Same-event retrospective baseline predictions do not represent a new no-feedback trajectory or prove exploration efficiency.',
            'Predicted-mask yield excludes unpredicted realized cells; outside counts are reported separately.',
            'All completed actions retained, including empty predicted masks and ROS4 failures; S26 pending action27 has no completed feedback and no invented label.',
            'No per-cell or per-action independence assumption, significance test, confidence interval, or generalization claim.',
            'Camera prediction masks are empty throughout; no camera gain learning benefit is established.',
        ],episodes=[evaluate_episode(label,ROOT/'reports/nso-hybrid-20261115'/name,cls) for label,name in EPISODES])
    previous=json.loads(Path(freeze['prior_files']['results.json']['path']).read_text())
    for prior,now in zip(previous['episodes'],result['episodes'][:3]):
        assert prior['actions']==now['actions']and prior['metrics']==now['metrics']
    later=result['episodes'][-1]
    records=json.loads(Path(later['records_manifest']).read_text())
    acquisition={r['action_id']:r['geometry']['rgbd']['sensor_stamp_ns']for r in records[1:]}
    results_cutoff=freeze['prior_files']['results.json']['mtime_ns']
    bundle_cutoff=freeze['prior_manifest_mtime_ns']
    after_results=[r for r in later['actions']if acquisition[r['action_id']]>results_cutoff]
    after_bundle=[r for r in later['actions']if acquisition[r['action_id']]>bundle_cutoff]
    later['held_out_interpretation']=dict(
        role='Previously unanalyzed later episode; metric definitions unchanged, no outcome-based tuning',
        wholly_prospectively_collected_after_v1_freeze=False,
        collection_overlapped_v1_analysis=True,
        original_result_freeze_unix_ns=results_cutoff, original_bundle_freeze_unix_ns=bundle_cutoff,
        per_action_actual_RGBD_acquisition_unix_ns=acquisition,
        action_ids_acquired_after_result_freeze=[r['action_id']for r in after_results],
        action_ids_acquired_after_whole_bundle_freeze=[r['action_id']for r in after_bundle],
        after_result_freeze_metrics={s:metric(after_results,s)for s in ['radar','camera']},
        after_whole_bundle_freeze_metrics={s:metric(after_bundle,s)for s in ['radar','camera']},
        suffix_does_not_reset_or_import_Beta='Stored same-episode posterior_before retained; not a new independent episode',
        efficacy='Later Brier/logloss do not improve; S26 benefit is episode-specific')
    result['prior_actions_and_metrics_preserved_exactly']=True
    result['action_count_total']=sum(e['completed_primitive_attempts']for e in result['episodes'])
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with(args.output/'action_metrics.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=['episode','action_id','action','ros_status','arrival','sensor','n','success','failure','outside','p_before','online_brier','fixed_brier','online_expected_in_mask_gain','actual_in_mask_gain'])
        writer.writeheader()
        for ep in result['episodes']:
            for row in ep['actions']:
                for sensor,v in row['channels'].items():
                    n=v['predicted_cells'];writer.writerow(dict(episode=ep['label'],action_id=row['action_id'],action=row['action'],ros_status=row['ros_status'],arrival=row['primitive_arrival_verified'],sensor=sensor,n=n,success=v['realized_predicted_cells'],failure=v['unrealized_predicted_cells'],outside=v['unpredicted_realized_cells'],p_before=v['posterior_before'],online_brier=v['online']['brier_sum']/n if n else '',fixed_brier=.25 if n else '',online_expected_in_mask_gain=v['online']['predicted_gain'],actual_in_mask_gain=v['realized_predicted_cells']))
    if args.plot:plot(result,args.output)
    print(json.dumps([{k:e[k]for k in ['label','completed_primitive_attempts','ROS_failed_action_ids','pending_without_completed_feedback','metrics']}for e in result['episodes']],indent=2))


if __name__=='__main__':main()
