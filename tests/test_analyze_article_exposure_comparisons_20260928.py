"""Synthetic sealed-input and paired-analysis checks; never create a World."""
from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import analyze_article_exposure_comparisons_20260928 as analysis
from tests.test_analyze_article_ground_comparisons_20260928 import metrics, trace


def rows():
    return [analysis.blank_row(arm,arm+'_'+scene+'_'+method,
        dict(scene_id=scene,method=method,budget=160,noise_seed=92801),None)
        for arm in analysis.ARMS for scene in analysis.SCENES for method in analysis.METHODS]


def forecasts(*,arm='ExposureV3',repeated=False,fallback=False):
    names=['planar','recessed','louvered','composite'];p=[.1,.2,.3,.4]
    components=[];new=[]
    for i,name in enumerate(names):
        before=0. if fallback else float(i+1);excluded=before*.25 if arm=='ExposureV3' else 0.
        new.append(before-excluded)
        for scale in (.85,1.,1.15):
            c=dict(structure=name,scale=scale,visible_area_m2=before+2.,already_observed_area_m2=2.,new_surface_area_m2=before-excluded)
            if arm=='ExposureV3':c.update(new_surface_area_before_exposure_m2=before,
                excluded_previously_exposed_unseen_area_m2=excluded,exposure_exclusion_is_not_measured_surface=True)
            components.append(c)
    effective=[0.]*4 if repeated else new
    candidate=dict(view_id='target:0',components=[] if fallback else components,fallback=fallback,
        repeated_view_excluded=repeated,structure_new_surface_area_m2=effective,
        expected_new_surface_area_m2=sum(a*b for a,b in zip(p,effective)),
        unexcluded_structure_new_surface_area_m2=new,unexcluded_expected_new_surface_area_m2=sum(a*b for a,b in zip(p,new)))
    return dict(instance_id='instance',structure_names=names,structure_probabilities=p,candidates=[candidate]),candidate


class Fixture:
    def __init__(self,root):
        self.root=root
        self.source=self.write('fixture_source.py',b'# synthetic read-only source\n')
        self.pins={str(self.source.relative_to(analysis.ROOT)):analysis.old.digest(self.source.read_bytes())}

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else analysis.old.canonical(value));return path

    def seal(self,root,files,**fields):
        records={}
        for name,value in files.items():
            path=self.write(str(root.relative_to(self.root)/name),value)
            records[name]=dict(sha256=analysis.old.digest(path.read_bytes()),bytes=path.stat().st_size)
        return self.write(str(root.relative_to(self.root)/'manifest.json'),dict(files=records,**fields))

    def protocols(self):
        archive=self.root/'sources.zip'
        with zipfile.ZipFile(archive,'x') as out:
            for name in self.pins:out.writestr(name,self.source.read_bytes())
        asset=self.write('assets/manifest.json',dict(fixture=True));reference=self.write('reference/manifest.json',dict(fixture_reference=True))
        common=dict(status='frozen',phase='ablation',evaluation=dict(threshold_m=.05),
            metric_preprocessing=dict(schema='numeric_face_preprocessing'),mapper=dict(voxel=.1),
            asset_root=str(asset.parent.relative_to(analysis.ROOT)),asset_manifest_sha256=analysis.old.digest(asset.read_bytes()),
            numerical_runtime=dict(runtime='synthetic'),
            references={scene:dict(root=str(reference.parent.relative_to(analysis.ROOT)),manifest_sha256=analysis.old.digest(reference.read_bytes())) for scene in analysis.SCENES},
            source_sha256=self.pins,source_archive=str(archive.relative_to(analysis.ROOT)),source_archive_sha256=analysis.old.digest(archive.read_bytes()))
        baseline=dict(deepcopy(common),schema='article.experiment_protocol.ground_v2',controller=dict(ground=True),
            output_relative_path=str((self.root/'ground').relative_to(analysis.ROOT)),
            slots={'ground_'+scene+'_'+method:dict(scene_id=scene,method=method,budget=160,noise_seed=92801,paired_baseline_run_id='old_'+scene+'_'+method)
                for scene in analysis.SCENES for method in analysis.METHODS})
        bp=self.write('baseline.json',baseline)
        protocol=dict(deepcopy(common),schema='article.experiment_protocol.exposure_v3',controller=dict(ground=True,same_center_exposure_dedup=True),
            output_relative_path=str((self.root/'exposure').relative_to(analysis.ROOT)),
            paired_baseline_protocol=dict(path=str(bp.relative_to(analysis.ROOT)),sha256=analysis.old.digest(bp.read_bytes())),
            slots={'exposure_'+name:dict(slot,paired_baseline_run_id=name) for name,slot in baseline['slots'].items()})
        return protocol,baseline,self.write('protocol.json',protocol)

    def terminal(self,*,arm='ExposureV3',failed=False):
        protocol,baseline,_=self.protocols();source=protocol if arm=='ExposureV3' else baseline
        run,slot=next(iter(source['slots'].items()));phase=analysis.ROOT/source['output_relative_path']
        episode=phase/'episodes'/run;reviewroot=self.root/'reviews';protocolsha='p'*64
        entry=dict(run_id=run,status='attempt_failed' if failed else 'controller_stop',qualified=not failed)
        report=dict(schema=analysis.REVIEW_SCHEMAS[arm][1],run_id=run,online_status=entry['status'],
            protocol_sha256=protocolsha,reviewer_source_sha256=self.pins,all_checks_passed=True,qualified=not failed,
            method=slot['method'],scene_id=slot['scene_id'],phase='ablation')
        manifestsha=None
        if failed:
            failure=dict(error=dict(type='SyntheticFailure',message='retained'))
            path=self.write(str((episode/'attempt_failure.json').relative_to(self.root)),failure)
            entry['result_sha256']=analysis.old.digest(path.read_bytes());report.update(status='failed_attempt_preserved',failure=failure)
        else:
            result=dict(actions=[],timings={},elapsed_s=1.)
            evaluation=dict(metrics=metrics(),metric_version=analysis.VERSION)
            artifacts={}
            for name,value in [('evaluation.json',evaluation),('result.json',result),('raw.bin',b'original sensor bytes')]:
                path=self.write(str((episode/name).relative_to(self.root)),value)
                artifacts[name]=dict(sha256=analysis.old.digest(path.read_bytes()),bytes=path.stat().st_size)
            manifestpath=self.write(str((episode/'artifact_manifest.json').relative_to(self.root)),
                dict(files=artifacts,source_sha256=source['source_sha256'],protocol_sha256=protocolsha))
            manifestsha=analysis.old.digest(manifestpath.read_bytes())
            entry.update(artifact_manifest_sha256=manifestsha,result_sha256=artifacts['result.json']['sha256'])
            report.update(status='reviewed',input_manifest_sha256=manifestsha,metrics=metrics(),metric_version=analysis.VERSION,
                metric_settings=source['evaluation'],physics_metrics=dict(paid_actions=2,budget=160,collisions=0,
                    returned_xy_and_yaw=True,rgbd_frames=3,translation_m=.5,action_counts=dict(forward=2)),
                paired_baseline=dict(run_id=slot['paired_baseline_run_id'],protocol=protocol['paired_baseline_protocol']),exposure_audit=dict(all_candidates=0))
        self.seal(reviewroot/run,{'review.json':report},schema=analysis.REVIEW_SCHEMAS[arm][0],
            protocol_sha256=protocolsha,reviewer_source_sha256=self.pins,input_episode_manifest_sha256=manifestsha)
        return phase,reviewroot,analysis.blank_row(arm,run,slot,entry),entry,source,protocolsha,episode


class ExposureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic_exposure_analysis_',dir=analysis.ROOT/'tmp')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.fixture=Fixture(self.root)

    def test_all_slots_pairs_and_no_means_for_unstarted(self):
        inventory=rows();pairs,witnesses=analysis.pair_rows(inventory,{})
        self.assertEqual(len(inventory),24);self.assertEqual(len(pairs),24)
        self.assertEqual(sum(p['pair_kind']=='across_arm' for p in pairs),12)
        self.assertTrue(all(r['J_nav'] is None for r in inventory));self.assertFalse(witnesses)
        self.assertTrue(all(s['all_layouts_mean_delta']['J_nav'] is None for s in analysis.ground.layout_summary(pairs).values()))

    def test_negative_effect_remains_and_motion_is_independent_gate(self):
        inventory=rows()
        for row in inventory:
            if row['scene_id']==analysis.SCENES[0] and row['method']=='S':
                analysis.ground.apply_metrics(row,metrics(.4 if row['arm']=='ExposureV3' else .6),
                    version=analysis.VERSION,motion=True,mode='synthetic')
        pairs,_=analysis.pair_rows(inventory,{})
        item=next(p for p in pairs if p['comparison']=='ExposureV3-GroundV2/S' and p['scene_id']==analysis.SCENES[0])
        self.assertLess(item['delta_J_nav'],0.)
        group=analysis.ground.layout_summary(pairs)['ExposureV3-GroundV2/S']
        self.assertEqual(group['available_layout_units'],1);self.assertIsNone(group['all_layouts_mean_delta']['J_nav'])
        inventory[-10]['motion_completion_verified']=False
        pairs,_=analysis.pair_rows(inventory,{})
        item=next(p for p in pairs if p['comparison']=='ExposureV3-GroundV2/S' and p['scene_id']==analysis.SCENES[0])
        self.assertEqual(item['quality_status'],'quality_available_motion_gate_not_met')

    def test_reference_or_version_mismatch_cannot_be_scored(self):
        inventory=rows();selected=[r for r in inventory if r['scene_id']==analysis.SCENES[0] and r['method']=='G']
        for r in selected:analysis.ground.apply_metrics(r,metrics(),version=analysis.VERSION,motion=True,mode='synthetic')
        for key,value in [('reference_fingerprint','other'),('metric_version','other')]:
            changed=deepcopy(inventory);next(r for r in changed if r['run_id']==selected[0]['run_id'])[key]=value
            pair=analysis.pair_rows(changed,{})[0][0]
            self.assertEqual(pair['quality_status'],'incompatible_metric_or_reference');self.assertNotIn('delta_J_nav',pair)

    def test_area_stages_are_distinct_including_exact_camera_gate(self):
        forecast,candidate=forecasts(repeated=True);receipt=analysis.area_receipt(forecast,candidate,'ExposureV3')
        self.assertAlmostEqual(receipt['expected_before_exposure_m2'],3.)
        self.assertAlmostEqual(receipt['expected_excluded_exposure_m2'],.75)
        self.assertAlmostEqual(receipt['expected_after_exposure_m2'],2.25)
        self.assertEqual(receipt['effective_new_surface_m2'],0.)
        self.assertAlmostEqual(receipt['exact_repeated_view_excluded_m2'],2.25)
        forecast,candidate=forecasts(arm='GroundV2');receipt=analysis.area_receipt(forecast,candidate,'GroundV2')
        self.assertEqual(receipt['expected_excluded_exposure_m2'],0.)
        self.assertEqual(receipt['expected_before_exposure_m2'],receipt['expected_after_exposure_m2'])

    def test_area_rejects_bad_partition_probability_and_inherited_field_semantics(self):
        for mutate in [lambda f,c:c['components'][0].update(new_surface_area_m2=.99),
            lambda f,c:f.update(structure_probabilities=[float('nan')]*4),
            lambda f,c:f.update(structure_probabilities=[-.1,.2,.3,.6]),
            lambda f,c:c.update(unexcluded_expected_new_surface_area_m2=3.),
            lambda f,c:c.update(structure_new_surface_area_m2=[]),
            lambda f,c:c['components'][0].update(visible_area_m2=float('inf'))]:
            forecast,candidate=forecasts();mutate(forecast,candidate)
            with self.assertRaises(ValueError):analysis.area_receipt(forecast,candidate,'ExposureV3')
        forecast,candidate=forecasts(fallback=True)
        self.assertEqual(analysis.area_receipt(forecast,candidate,'ExposureV3')['effective_new_surface_m2'],0.)

    def test_physics_completion_is_not_original_qualification(self):
        p=dict(paid_actions=160,budget=160,collisions=0,returned_xy_and_yaw=True,rgbd_frames=161)
        self.assertTrue(analysis.motion_complete(p))
        for key,value in [('collisions',1),('returned_xy_and_yaw',False),('rgbd_frames',160),('paid_actions',161)]:
            self.assertFalse(analysis.motion_complete(dict(p,**{key:value})))
        self.assertIsNone(analysis.motion_complete({}))

    def test_protocol_rejects_main_partial_pairs_and_covariate_drift(self):
        p,b,_=self.fixture.protocols();analysis.validate_pairing(p,b)
        for mutate in [lambda x:x.update(phase='main'),lambda x:x['slots'].pop(next(iter(x['slots']))),
            lambda x:x['controller'].update(semantic_bonus=5),
            lambda x:next(iter(x['slots'].values())).update(noise_seed=2),
            lambda x:x['evaluation'].update(threshold_m=.1)]:
            altered=deepcopy(p);mutate(altered)
            with self.assertRaises(ValueError):analysis.validate_pairing(altered,b)

    def test_only_replaced_ground_cli_can_leave_new_source_closure(self):
        old={'shared.py':'same','scripts/run_article_ground_experiment_20260928.py':'old'}
        analysis.verify_source_relationship({'shared.py':'same','new.py':'new'},old)
        for new in [{'shared.py':'changed'},{'new.py':'new'}]:
            with self.assertRaises(ValueError):analysis.verify_source_relationship(new,old)

    def test_review_and_every_artifact_pin_are_required(self):
        phase,reviews,row,entry,protocol,pin,episode=self.fixture.terminal()
        ready=analysis.load_terminal(analysis.old.Inputs(),phase,reviews,row,entry,protocol,pin)
        self.assertEqual(ready[0],episode);self.assertTrue(row['quality_measurement_available']);self.assertTrue(row['motion_completion_verified'])
        self.assertEqual(row['input_evaluation_sha256'],ready[1]['files']['evaluation.json']['sha256'])
        (episode/'raw.bin').write_bytes(b'tampered raw')
        with self.assertRaises(ValueError):analysis.load_terminal(analysis.old.Inputs(),phase,reviews,row,entry,protocol,pin)

    def test_failed_attempt_is_preserved_without_zero_metrics_or_trace(self):
        phase,reviews,row,entry,protocol,pin,_=self.fixture.terminal(failed=True)
        self.assertIsNone(analysis.load_terminal(analysis.old.Inputs(),phase,reviews,row,entry,protocol,pin))
        self.assertEqual(row['status'],'failed_attempt_reviewed');self.assertEqual(row['online_status'],'attempt_failed')
        self.assertIsNone(row['J_nav']);self.assertFalse(row['quality_measurement_available']);self.assertIsNone(row['motion_completion_verified'])

    def test_complete_synthetic_snapshot_keeps_running_slot_and_source_ledger_pins(self):
        p,b,path=self.fixture.protocols();run=next(iter(p['slots']))
        self.fixture.write('exposure/start_ledger.json',dict(protocol_sha256=analysis.old.digest(path.read_bytes()),slots=p['slots'],
            entries=[dict(run_id=run,status='reserved')]))
        output=self.root/'snapshot'
        with patch.object(analysis,'STAGE',self.root):
            summary=analysis.analyze(path,output,self.root/'v3reviews',self.root/'v2reviews')
        self.assertEqual(summary['arm_records'],24);self.assertEqual(summary['paired_cells'],12)
        self.assertEqual(summary['status_counts']['ExposureV3'],{'running_reserved':1,'unstarted':11})
        self.assertEqual(summary['new_worlds'],0);self.assertEqual(summary['new_quality_evaluations'],0)
        manifest=json.loads((output/'manifest.json').read_bytes())
        self.assertEqual(len(manifest['source_sha256']),3)
        self.assertTrue(any(r['mutable_snapshot'] for r in manifest['input_sha256'].values()))
        for name in ('slots.csv','paired_effects.csv'):
            with (output/name).open() as stream:self.assertEqual(len(list(csv.DictReader(stream))),24)
        self.assertLess(sum(p.stat().st_size for p in output.iterdir()),100000)
        with patch.object(analysis,'STAGE',self.root),self.assertRaises(ValueError):
            analysis.analyze(path,output,self.root/'v3reviews',self.root/'v2reviews')

    def test_common_prefix_limits_macro_peer_score_and_action_witnesses(self):
        inventory=rows();left=trace();right=trace(right=True)
        for i in range(3):
            left['steps'][i]['macro_target']='same'
            right['steps'][i]['macro_target']='changed' if i>=1 else 'same'
            for state,original,peer in [(left['steps'][i],'first_id','left_peer'),(right['steps'][i],'other_id','right_peer')]:
                state['observed_anchors'][peer]=[2.,0.,1.]
                state['beliefs'][peer]=deepcopy(state['beliefs'][original]);state['beliefs'][peer]['peers']=[]
            if i>=1:right['steps'][i]['beliefs']['other_id']['peers']=['right_peer']
        by={(r['arm'],r['scene_id'],r['method']):r for r in inventory}
        lr=by['ExposureV3',analysis.SCENES[0],'G'];rr=by['GroundV2',analysis.SCENES[0],'G']
        pairs,witnesses=analysis.pair_rows(inventory,{lr['run_id']:left,rr['run_id']:right});pair=pairs[0]
        self.assertEqual(pair['first_action_divergence_paid_step'],2)
        self.assertEqual(pair['first_changed_score_paid_step'],1)
        self.assertEqual(pair['first_macro_target_divergence_paid_step'],1)
        self.assertEqual(pair['first_common_peer_set_divergence_paid_step'],1)
        self.assertEqual(witnesses[pair['pair_id']]['first_common_peer_set']['right_peer_ids_in_left_identity'],['left_peer'])
        right['steps'][1]['macro_target']='same';right['steps'][1]['beliefs']['other_id']['peers']=[]
        pair=analysis.pair_rows(inventory,{lr['run_id']:left,rr['run_id']:right})[0][0]
        self.assertIsNone(pair['first_macro_target_divergence_paid_step']);self.assertIsNone(pair['first_common_peer_set_divergence_paid_step'])

    def test_trace_details_use_post_feedback_belief_and_report_only_selected_view_areas(self):
        episode=self.root/'episode';manifest=dict(files={});compact=dict(steps={})
        for step in range(3):
            forecast,candidate=forecasts();other=deepcopy(candidate);other['view_id']='not_selected:0';forecast['candidates'].append(other)
            association=[dict(instance_id=key,association_uncertain=False,class_conflict=False,observed_class='cabinet',
                distinct_class_supports=dict(cabinet=2),geometry_log_scores=[0.,0.,0.,0.]) for key in ('instance','peer')]
            beliefs=[dict(instance_id=key,geometry_log_evidence=[1.,0.,0.,0.] if step and key=='peer' else [0.]*4,
                peer_instance_ids=['peer'] if step and key=='instance' else []) for key in ('instance','peer')]
            record=dict(controller_evidence=dict(association=dict(instances=association),structure_belief=dict(instances=beliefs)),
                decision=dict(macro_id=0 if step==0 else 1,macro_target='target' if step else None,
                    reason='normal',global_replanned=bool(step),global_selection=None if not step else dict(
                        selected=dict(kind='direct',target=dict(node='target',heading=0),score=.5,expected_gain=1.,total_cost=2),
                        remaining=160-step,forecasts=[forecast])))
            path=self.fixture.write('episode/steps/'+f'{step:03d}.json.gz',gzip.compress(analysis.old.canonical(record),mtime=0))
            manifest['files'][f'steps/{step:03d}.json.gz']=dict(sha256=analysis.old.digest(path.read_bytes()),bytes=path.stat().st_size)
            compact['steps'][step]={}
        detail=analysis.trace_details(analysis.old.Inputs(),episode,manifest,'ExposureV3',compact)
        self.assertEqual(detail['macro_count'],1);self.assertEqual(detail['observed_instance_count'],2)
        self.assertEqual(detail['first_recorded_peer_step'],1);self.assertEqual(detail['first_informative_peer_step'],1)
        self.assertEqual(len(detail['selected_exposure_receipts']),2)
        self.assertTrue(all(r['view_id']=='target:0' for r in detail['selected_exposure_receipts']))
        self.assertEqual(detail['replans'][0]['forecast_candidates'],2)
        self.assertFalse(detail['observed_instance_count_is_gt_discovery'])
        self.assertFalse(detail['replans'][0]['area_is_selection_utility'])

    def test_actual_trajectory_length_matches_review_without_synthesized_points(self):
        inventory=rows();row=inventory[0];row['path_length_m']=.25
        trace=dict(trajectory=[dict(paid_step=0,x_m=0.,y_m=0.,yaw_rad=0.),dict(paid_step=1,x_m=.25,y_m=0.,yaw_rad=0.)])
        pairs,witnesses=analysis.pair_rows(inventory,{})
        output=self.root/'trajectory_snapshot'
        with patch.object(analysis,'STAGE',self.root):
            analysis.save_snapshot(output,inventory,{row['run_id']:trace},{},pairs,witnesses,analysis.old.Inputs(),[])
        with (output/'trajectories.csv').open() as stream:result=list(csv.DictReader(stream))
        self.assertEqual(len(result),2);self.assertEqual(float(result[1]['cumulative_path_m']),.25)
        row['path_length_m']=10.
        with patch.object(analysis,'STAGE',self.root),self.assertRaises(ValueError):
            analysis.save_snapshot(self.root/'bad_path',inventory,{row['run_id']:trace},{},pairs,witnesses,analysis.old.Inputs(),[])

    def test_verified_costs_remain_separate_when_quality_is_unavailable(self):
        inventory=rows()
        for row in inventory:
            if row['scene_id']==analysis.SCENES[0] and row['method']=='G':
                row.update(paid_actions=160,path_length_m=20. if row['arm']=='ExposureV3' else 24.)
        pair=analysis.pair_rows(inventory,{})[0][0]
        self.assertEqual(pair['quality_status'],'unavailable');self.assertEqual(pair['delta_path_length_m'],-4.)
        self.assertEqual(pair['cost_status'],'reviewed_recorded_costs_available');self.assertNotIn('delta_J_nav',pair)

    def test_gate_export_preserves_six_positions_missing_latch_and_source_pins(self):
        inventory=rows();selected={r['method']:r for r in inventory
            if r['arm']=='ExposureV3' and r['scene_id']==analysis.SCENES[0] and r['method'] in ('S','B')}
        traces={selected['S']['run_id']:trace(),selected['B']['run_id']:trace(right=True)}
        details={}
        for run in traces:
            details[run]=dict(gate_steps={i:dict(paid_step=i,macro_id=1,macro_target=dict(node='target',heading=0),
                selected=dict(kind='diagnose_then_observe',target=dict(node='target',heading=0)),global_replanned=i==0,
                return_latched_recorded=None,decision_guard=dict(allowed=True),remaining_budget=160-i,
                return_reserve_margin_after_action=100-i,planning_evidence=dict(instances={},recorded_peer_links=[])) for i in range(3)},
                episode_relative_path='synthetic/'+run,episode_file_pins={
                    f'steps/{i:03d}.json.gz':dict(bytes=10,sha256=str(i)*64) for i in range(3)})
        pairs,_=analysis.pair_rows(inventory,traces)
        result=analysis.main_gate_witnesses(inventory,traces,details,pairs)
        self.assertEqual(len(result['pairs']),6);self.assertFalse(result['automatic_gate_approval']);self.assertEqual(result['new_main_runs'],0)
        target=next(r for r in result['pairs'] if r['comparison']=='ExposureV3/S-B' and r['scene_id']==analysis.SCENES[0])
        self.assertEqual(target['status'],'evidence_ready_for_manual_review');self.assertIsNone(target['automatic_gate_pass'])
        self.assertEqual(target['first_actual_action_divergence_paid_step'],2)
        for side in target['sides'].values():
            self.assertEqual(side['macro_commit_paid_step'],0);self.assertTrue(side['macro_commit_within_common_prefix'])
            self.assertIsNone(side['current_decision']['return_latched_recorded'])
            self.assertEqual(len(side['supporting_file_pins']),3)
        self.assertEqual(sum(r['status']=='unavailable' for r in result['pairs']),5)

    def test_gate_absent_or_unpaired_action_is_not_inferred_as_effect(self):
        inventory=rows();selected={r['method']:r for r in inventory
            if r['arm']=='ExposureV3' and r['scene_id']==analysis.SCENES[0] and r['method'] in ('S','B')}
        for case in ('same','ended','sensor_diverged'):
            left=trace();right=trace(right=True,different_actions=case=='sensor_diverged')
            if case=='ended':right['result']['actions']=right['result']['actions'][:1]
            if case=='sensor_diverged':right['steps'][0]['observation_content_sha']='earlier sensor change'
            traces={selected['S']['run_id']:left,selected['B']['run_id']:right}
            pairs,_=analysis.pair_rows(inventory,traces)
            result=analysis.main_gate_witnesses(inventory,traces,{k:{} for k in traces},pairs)
            target=next(r for r in result['pairs'] if r['comparison']=='ExposureV3/S-B' and r['scene_id']==analysis.SCENES[0])
            self.assertEqual(target['status'],dict(same='no_actual_action_divergence',ended='one_execution_ended_before_action_pair',
                sensor_diverged='action_difference_after_sensor_divergence')[case])
            self.assertNotIn('sides',target);self.assertIsNone(target['automatic_gate_pass'])

    def test_gate_uses_post_feedback_instance_count_and_real_peer_qualification(self):
        instances=[dict(instance_id=k,observed_class='cabinet',association_uncertain=False,class_conflict=False,
            distinct_class_supports=dict(cabinet=2),geometric_feedback_frames=5) for k in ('self','peer')]
        beliefs=[dict(instance_id=k,observed_class='cabinet',geometry_log_evidence=[0.,-1.,-2.,-3.],
            peer_instance_ids=['peer'] if k=='self' else [],rho=.6,structure_probabilities=[.4,.3,.2,.1]) for k in ('self','peer')]
        evidence=dict(association=dict(instances=instances),structure_belief=dict(instances=beliefs),
            geometry_feedback=[dict(applied=True,instance_id='peer',instance=dict(instances[1],geometric_feedback_frames=6))])
        state=analysis.planning_evidence(evidence)
        self.assertEqual(state['instances']['peer']['geometric_feedback_frames'],6)
        self.assertTrue(state['recorded_peer_links'][0]['same_qualified_category'])
        self.assertTrue(state['recorded_peer_links'][0]['peer_geometry_informative'])
        decision=analysis.gate_decision(dict(controller_evidence=evidence,decision={}),first_recorded_return_step=None)
        self.assertIsNone(decision['return_latched_recorded']);self.assertIsNone(decision['remaining_budget'])
        self.assertIsNone(decision['committed_macro_budget_feasible']);self.assertIsNone(decision['return_reserve_margin_after_action'])


if __name__=='__main__':unittest.main()
