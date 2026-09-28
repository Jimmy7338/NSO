#!/usr/bin/env python3
"""Read-only descriptive summary of the three already-seen-asset integration slots.

No controller, mapper, World, sensor or numerical surface evaluator is imported.
Logged EVI is a planning proxy; cross-instance net value is not inferred merely
from a feature flag or from a positive total EVI.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

RUNS=('integration_A_G','integration_A_B','integration_A_S')
COMPLETE={'controller_stop','controller_blocked','budget_exhausted','stopped_without_confirmed_return'}
EPS=1e-12
MINIMUM_SEMANTIC_VIEWS=2  # Fixed frontend setting in this integration protocol.


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError('nonfinite JSON: '+value)))


def _different_vector(left,right):
    if left is None:return False
    if len(left)!=len(right):return True
    return any((a is None)!=(b is None) or (a is not None and b is not None and abs(a-b)>EPS)
               for a,b in zip(left,right))


def parse_steps(records):
    counts=Counter();plane_reasons=Counter();fallback_reasons=Counter();residual_reasons=Counter()
    instances=defaultdict(lambda:dict(counts=Counter(),paid_steps=set(),poses=set(),
        forecast_view_ids=set(),last_belief=None,observed_classes=set(),rho_values=[]))
    posterior_previous={};log_previous={};rho_previous={};selected=[];decisions=[];poses=[]
    first={};evi_values=[];selected_evi=[];eligible_ever=defaultdict(set)
    max_same_class=0;max_same_class_visible=0;first_same_class=None;first_same_class_visible=None
    for index,step,receipt in records:
        evidence,decision=step['controller_evidence'],step['decision']
        counts['saved_decisions']+=1
        pose=receipt['execution']['pose_xyyaw_rad'];poses.append(pose)
        rounded_pose=tuple(0. if abs(float(x))<1e-9 else round(float(x),9) for x in pose)
        decisions.append(dict(paid_step=index,action=decision.get('action'),macro_target=decision.get('macro_target'),
            macro_id=decision.get('macro_id'),global_replanned=decision.get('global_replanned',False)))
        counts['macro_observe_submitted']+=int(decision.get('macro_observe_submitted',False))
        counts['completed_macro_receipts']+=int(evidence.get('completed_macro_id') is not None)
        association=evidence.get('association',{})
        accepted_ids={row['instance_id'] for row in association.get('accepted',[])}
        qualified_by_class=defaultdict(list)
        for row in association.get('instances',[]):
            key=row['instance_id'];label=row.get('observed_class');item=instances[key]
            if label is not None:item['observed_classes'].add(label)
            support=row.get('distinct_class_supports',{}).get(label,0)
            qualified=(label is not None and not row.get('association_uncertain',False)
                and not row.get('class_conflict',False) and support>=MINIMUM_SEMANTIC_VIEWS)
            counts['frontend_qualified_instance_frames']+=int(qualified)
            item['counts']['frontend_qualified_frames']+=int(qualified)
            if qualified:
                eligible_ever[label].add(key)
                qualified_by_class[label].append(dict(instance_id=key,distinct_class_supports=support))
        current_same=max((len(group) for group in qualified_by_class.values()),default=0)
        max_same_class=max(max_same_class,current_same)
        counts['frames_with_same_class_qualified_coexisting_instances']+=int(current_same>=2)
        visible_same=0
        for label,group in sorted(qualified_by_class.items()):
            visible=[row for row in group if row['instance_id'] in accepted_ids]
            visible_same=max(visible_same,len(visible))
            if len(group)>=2 and first_same_class is None:
                first_same_class=dict(paid_step=index,observed_class=label,instances=group)
            if len(visible)>=2 and first_same_class_visible is None:
                first_same_class_visible=dict(paid_step=index,observed_class=label,instances=visible)
        max_same_class_visible=max(max_same_class_visible,visible_same)
        counts['frames_with_same_class_qualified_current_view_instances']+=int(visible_same>=2)
        for row in association.get('accepted',[]):
            key=row['instance_id'];item=instances[key]
            item['counts']['accepted_paid_associations']+=1;item['paid_steps'].add(index);item['poses'].add(rounded_pose)
            item['counts']['novel_support_associations']+=int(row.get('novel_support_voxels',0)>0)
        for row in evidence.get('view_evidence',{}).get('results',[]):
            fit=row.get('current_plane_fit');item=instances[row['instance_id']]
            if fit is not None:
                counts['plane_fit_attempts']+=1;plane_reasons[fit.get('reason','unknown')]+=1
                counts['accepted_plane_fits']+=int(fit.get('accepted',False))
                item['counts']['accepted_plane_fits']+=int(fit.get('accepted',False))
                if fit.get('accepted'):first.setdefault('valid_observed_label_plane_paid_step',index)
        for row in evidence.get('observed_residual',{}).get('results',[]):
            accepted=bool(row.get('accepted'));counts['accepted_residuals']+=int(accepted)
            counts['informative_residuals']+=int(accepted and row.get('informative',False))
            residual_reasons[row.get('reason','unknown')]+=1
            if accepted and row.get('informative'):first.setdefault('informative_paid_geometry_residual_step',index)
        for row in evidence.get('geometry_feedback',[]):
            applied=bool(row.get('applied'));counts['frontend_geometry_updates_applied']+=int(applied)
            instances[row['instance_id']]['counts']['frontend_geometry_updates_applied']+=int(applied)
        belief=evidence.get('structure_belief',{})
        counts['frames_with_structure_belief']+=int('instances' in belief)
        counts['frames_with_cross_instance_belief_enabled']+=int(belief.get('share_across_instances',False))
        for row in belief.get('instances',[]):
            key=row['instance_id'];item=instances[key];item['last_belief']=row
            if row.get('observed_class') is not None:item['observed_classes'].add(row['observed_class'])
            semantic=bool(row.get('semantic_conditioning_used'))
            counts['semantic_conditioned_instance_frames']+=int(semantic)
            item['counts']['semantic_conditioned_frames']+=int(semantic)
            if semantic:first.setdefault('semantic_conditioning_paid_step',index)
            posterior=row.get('structure_probabilities',[]);logs=row.get('geometry_log_evidence',[])
            if _different_vector(posterior_previous.get(key),posterior):counts['posterior_changes_between_paid_frames']+=1
            if _different_vector(log_previous.get(key),logs):counts['planner_geometry_evidence_changes']+=1
            posterior_previous[key]=posterior;log_previous[key]=logs
            nonzero=any(value is None or abs(value)>EPS for value in logs)
            counts['planner_nonzero_geometry_evidence_instance_frames']+=int(nonzero)
            item['counts']['planner_nonzero_geometry_evidence_frames']+=int(nonzero)
            peers=row.get('peer_instance_ids',[]);rho=row.get('rho')
            counts['instance_frames_with_same_class_peers']+=int(bool(peers))
            shifted=bool(peers) and rho is not None and abs(rho-.5)>EPS
            counts['peer_adjusted_reliability_instance_frames']+=int(shifted)
            if shifted:first.setdefault('peer_adjusted_reliability_paid_step',index)
            if rho is not None:
                item['rho_values'].append(float(rho))
                if key in rho_previous and abs(rho-rho_previous[key])>EPS:counts['rho_changes_between_paid_frames']+=1
                rho_previous[key]=rho
        global_selection=decision.get('global_selection')
        if not global_selection:continue
        counts['global_replans']+=1
        pool=global_selection.get('candidate_pool',{})
        counts['public_candidates_across_replans']+=pool.get('selected',len(pool.get('candidates',[])))
        direct=global_selection.get('direct_options',[]);diagnostic=global_selection.get('diagnostic_options',[])
        counts['direct_options_scored']+=len(direct);counts['diagnostic_options_scored']+=len(diagnostic)
        counts['positive_direct_options']+=sum(row.get('score',0)>EPS for row in direct)
        choice=global_selection.get('selected')
        if choice:
            target=choice.get('target',{});view_id=f"{target.get('node')}:{target.get('heading')}"
            chosen_forecasts=[dict(instance_id=forecast['instance_id'],
                expected_new_surface_area_m2=row.get('expected_new_surface_area_m2',0))
                for forecast in global_selection.get('forecasts',[]) for row in forecast.get('candidates',[])
                if row.get('view_id')==view_id and not row.get('fallback',False)
                and row.get('expected_new_surface_area_m2',0)>EPS]
            selected.append(dict(paid_step=index,macro_id=decision.get('macro_id'),kind=choice.get('kind'),
                target=choice.get('target'),instance_id=choice.get('instance_id'),score=choice.get('score'),evi=choice.get('evi'),
                positive_instance_forecasts_at_selected_target=chosen_forecasts))
            counts['selected_'+choice.get('kind','unknown')]+=1
            counts['selected_direct_with_positive_instance_forecast']+=int(choice.get('kind')=='direct' and bool(chosen_forecasts))
            if choice.get('kind')=='diagnose_then_observe':
                first.setdefault('selected_diagnostic_paid_step',index)
                selected_evi.append(float(choice.get('evi',0)))
                counts['selected_diagnostics_with_positive_evi']+=int(choice.get('evi',0)>EPS)
        for row in diagnostic:
            evi=float(row.get('evi',0));evi_values.append(evi)
            positive=evi>EPS;cross=bool(row.get('cross_instance_future_used',False))
            counts['diagnostic_options_with_positive_evi']+=int(positive)
            counts['diagnostic_options_with_cross_future_enabled']+=int(cross)
            counts['positive_total_evi_options_with_cross_future_enabled']+=int(positive and cross)
            counts['diagnostic_options_with_future_information_enabled']+=int(row.get('future_information_used',False))
            if positive:first.setdefault('positive_diagnostic_evi_paid_step',index)
        for forecast in global_selection.get('forecasts',[]):
            key=forecast['instance_id'];item=instances[key];counts['instance_forecasts']+=1
            counts['semantic_conditioned_instance_forecasts']+=int(forecast.get('semantic_conditioning_used',False))
            for row in forecast.get('candidates',[]):
                counts['instance_view_forecasts']+=1;item['counts']['view_forecasts']+=1
                item['forecast_view_ids'].add(row.get('view_id'))
                fallback=bool(row.get('fallback'));counts['view_forecast_fallbacks']+=int(fallback)
                counts['view_forecast_nonfallbacks']+=int(not fallback)
                item['counts']['forecast_fallbacks']+=int(fallback)
                if fallback:fallback_reasons[row.get('fallback_reason','unknown')]+=1
                counts['repeated_paid_view_exclusions']+=int(row.get('repeated_view_excluded',False))
                counts['positive_instance_view_forecasts']+=int(row.get('expected_new_surface_area_m2',0)>EPS)
    output_instances=[]
    for key,item in sorted(instances.items()):
        output_instances.append(dict(observed_instance_id=key,counts=dict(item['counts']),
            accepted_paid_frames=len(item['paid_steps']),distinct_paid_camera_poses=len(item['poses']),
            pose_count_rounding_decimals=9,unique_forecast_view_ids=len(item['forecast_view_ids']),
            observed_classes=sorted(item['observed_classes']),last_belief=item['last_belief'],
            rho_min=min(item['rho_values']) if item['rho_values'] else None,
            rho_max=max(item['rho_values']) if item['rho_values'] else None))
    return dict(counts=dict(counts),first_events=first,observed_instances=output_instances,
        observed_instance_count=len(instances),plane_fit_reasons=dict(plane_reasons),
        forecast_fallback_reasons=dict(fallback_reasons),residual_reasons=dict(residual_reasons),
        diagnostic_evi_max=max(evi_values,default=0.),selected_diagnostic_evi_max=max(selected_evi,default=0.),
        same_class_opportunity=dict(source='shared frontend association.instances; not method-filtered planning labels',
            minimum_distinct_class_supports=MINIMUM_SEMANTIC_VIEWS,
            qualification='Observed class present, association not uncertain, no class conflict, minimum distinct class supports satisfied.',
            qualified_instance_ids_ever_by_class={key:sorted(value) for key,value in sorted(eligible_ever.items())},
            qualified_unique_instance_count=len(set().union(*eligible_ever.values())) if eligible_ever else 0,
            max_same_class_qualified_coexisting_instances=max_same_class,
            same_class_qualified_coexistence_observed=max_same_class>=2,
            first_coexistence_witness=first_same_class,
            max_same_class_qualified_current_view_instances=max_same_class_visible,
            first_current_view_witness=first_same_class_visible,
            coexistence_is_ledger_membership_not_same_camera_view=True),
        cross_instance_future_net_gain_nonzero=None,
        cross_instance_future_net_gain_identified=False,
        cross_instance_identification_limit='Receipts record total EVI and a cross-instance feature flag, not the matched no-cross counterfactual score. Their conjunction does not identify net cross-instance information value.',
        frontend_updates_are_not_automatically_planner_feedback=True,
        selected_direct_instance_forecast_is_not_causal_revisit_attribution=True,
        forecast_area_is_not_measured_terminal_quality=True,
        selected_global_options=selected,_decisions=decisions,_poses=poses)


def _verify_manifest(episode,entry):
    manifest=read(episode/'artifact_manifest.json');files=manifest['files']
    if manifest.get('schema')!='semantic.experiment.artifacts.v1':raise ValueError('new integration episode required')
    if manifest['source_sha256']!=entry['metadata']['source_sha256']:raise ValueError('durable reservation source mismatch')
    actual={str(p.relative_to(episode)) for p in episode.rglob('*') if p.is_file()}
    if actual!=set(files)|{'artifact_manifest.json'}:raise ValueError('episode inventory mismatch')
    for name,row in files.items():
        path=episode/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(episode):
            raise ValueError('plain contained episode artifact required')
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('artifact changed: '+name)
    if sha(episode/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('archived protocol mismatch')
    return manifest


def _records(episode,result,manifest,protocol):
    count=result['acquired_and_saved_packets'];encoding=read(episode/'encoding.json')['steps']
    if len(encoding)!=count:raise ValueError('incomplete saved step compression ledger')
    for index,row in enumerate(encoding):
        name=f'steps/{index:03d}.json.gz'
        if (row['artifact']!=name or row['stored_bytes']!=manifest['files'][name]['bytes']
                or not 0<row['uncompressed_bytes']<=protocol['maximum_file_bytes']):
            raise ValueError('step encoding binding mismatch')
        with gzip.open(episode/name,'rb') as stream:content=stream.read(protocol['maximum_file_bytes']+1)
        if len(content)!=row['uncompressed_bytes']:raise ValueError('bounded decompression length differs')
        yield index,json.loads(content),read(episode/f'packets/{index:03d}_receipt.json')


def _verify_review(review,run_id,manifest_sha,manifest,result):
    """Validate replay evidence without invoking a policy or numerical backend."""
    runtime=review.get('runtime',{});replay=review.get('replay',{})
    if (review.get('schema')!='semantic.experiment.review.v1' or review.get('run_id')!=run_id
            or review.get('phase_id')!='semantic_integration_20260923'
            or review.get('status') not in ('experiment_reviewed','experiment_replay_verified','experiment_review_failed')
            or (review.get('status')!='experiment_review_failed' and review.get('error') is not None)
            or review.get('episode_manifest_sha256')!=manifest_sha
            or review.get('source_sha256')!=manifest['source_sha256']
            or review.get('current_sources_match') is not True or review.get('current_source_differences')
            or runtime.get('no_new_world_or_sensor_action') is not True
            or 'before' not in runtime or runtime['before']!=runtime.get('after')):
        raise ValueError('review episode/source/runtime binding differs')
    if replay.get('status')!='verified':return False
    rr=replay.get('runtime',{})
    if (replay.get('verification_kind')!='same_driver_saved_packet_policy_and_TSDF_replay'
            or replay.get('prediction_verified') is not True or replay.get('occupancy_exact') is not True
            or replay.get('frames_verified')!=result['acquired_and_saved_packets']
            or replay.get('terminal_status_verified')!=result['status']
            or replay.get('mesh_order_invariant_tolerance_m')!=1e-9
            or replay.get('counterfactual_trajectory') is not False
            or any(replay.get(key)!=0 for key in ('new_worlds','new_sensor_packets','physical_actions'))
            or 'before' not in rr or rr['before']!=rr.get('after')):
        raise ValueError('claimed complete policy/TSDF replay is not fully bound')
    return True


def _check_evaluation(value,protocol,result):
    reference=protocol.get('reference_manifest_sha256')
    expected=result['status']=='controller_stop' and result['sensor_status'].get('returned_xy_and_yaw') is True
    if (value.get('reference_manifest_sha256')!=reference
            or value.get('task_success')!=expected or value.get('original_episode_status')!=result['status']
            or value.get('all_task_instances_in_macro_denominator') is not True
            or value.get('prediction_roi_cropped') is not False
            or any(value.get(key)!=0 for key in ('new_worlds','new_sensor_packets','new_tsdf_integrations'))):
        raise ValueError('evaluation reference, target status or complete-prediction contract differs')
    metrics=value['metrics'];coverage=value['coverage'];config=protocol['evaluation']
    if (metrics['prediction_seed']!=config['seed'] or metrics['prediction_sample_spacing_m']!=config['sample_spacing_m']
            or metrics['threshold_m']!=config['threshold_m'] or metrics['C_nav']!=coverage['C_nav']
            or metrics['J_nav']!=metrics['C_nav']*metrics['Q']):
        raise ValueError('evaluation metric configuration or reported product differs')


def _numeric_supplement(path,episode,manifest_sha,manifest,result,protocol,review_path,review):
    value=read(path);runtime=value.get('runtime',{})
    if (value.get('schema')!='semantic.endpoint.numeric_supplement.v1'
            or value.get('status')!='semantic_endpoint_evaluation_completed' or value.get('error') is not None
            or value.get('episode_manifest_sha256')!=manifest_sha
            or value.get('run_id')!=review['run_id'] or Path(value.get('episode_root','')).resolve()!=episode
            or value.get('original_review_sha256')!=sha(review_path)
            or Path(value.get('original_review_path','')).resolve()!=review_path.resolve()
            or value.get('numerical_evaluation_recomputed') is not True or value.get('original_replay_rerun') is not False
            or review.get('status')!='experiment_review_failed'
            or review.get('error')!={'type':'ValueError','message':'nonfinite or degenerate triangle rejected'}
            or not _verify_review(review,review['run_id'],manifest_sha,manifest,result)
            or runtime.get('no_new_world_or_sensor_action') is not True
            or 'before' not in runtime or runtime['before']!=runtime.get('after')):
        raise ValueError('numeric supplement must bind original diagnosed failure and independent replay')
    sources=value['numerical_source_sha256'];archive=Path(value['numerical_source_root'])
    extras={'nso/complete_surface_evaluation.py','scripts/evaluate_semantic_integration_endpoint.py'}
    if set(sources)!=set(manifest['source_sha256'])|extras or any(sources[k]!=v for k,v in manifest['source_sha256'].items()):
        raise ValueError('numeric supplement source closure differs')
    for name,expected in sources.items():
        relative=Path(name);source=archive/relative
        if (relative.is_absolute() or '..' in relative.parts or source.is_symlink()
                or not source.resolve().is_relative_to(archive.resolve()) or sha(source)!=expected):
            raise ValueError('numeric supplemental source archive changed')
    diagnosis=value['diagnosis'];metrics=value['evaluation']['metrics']
    if (diagnosis['mesh_sha256']!=sha(episode/'prediction/mesh.npz')
            or diagnosis.get('all_faces_preserved') is not True or diagnosis.get('zero_area_faces')!=0
            or not diagnosis.get('finite_vertices') or not diagnosis.get('positive_faces_below_old_floor',0)>0
            or metrics.get('prediction_mesh_validation')!='strict_positive_area_without_absolute_area_floor'
            or metrics.get('positive_subthreshold_faces_preserved')!=diagnosis['positive_faces_below_old_floor']):
        raise ValueError('numeric supplement mesh/positive-face diagnosis differs')
    _check_evaluation(value['evaluation'],protocol,result)
    return value


def _review_info(root,run_id,episode,manifest_sha,manifest,result,protocol,ledger):
    reviews=root/'reviews'
    candidates=[reviews/f'{run_id}_review.json',reviews/f'{run_id}_replay.json',reviews/f'{run_id}.json']
    present=[path for path in candidates if path.is_file()]
    info=dict(available=bool(present),saved_policy_replay_verified=False,evaluation_available=False,
        numerical_evaluation_recomputed=False,evaluation_execution='not_available')
    if not present:return info
    checked=[]
    for path in present:
        value=read(path);verified=_verify_review(value,run_id,manifest_sha,manifest,result)
        if verified and value['replay'].get('replayed_tsdf_integrations')!=read(episode/'prediction/mapper.json')['tsdf_integration_count']:
            raise ValueError('review replay fusion count differs from saved mapper')
        checked.append((path,value,verified))
    path,review,verified=checked[0]
    info.update(path=str(path),sha256=sha(path),status=review['status'],error=review.get('error'),
        saved_policy_replay_verified=verified,saved_policy_replay_frames=review.get('replay',{}).get('frames_verified'),
        original_review_receipts=[dict(path=str(p),sha256=sha(p),status=v['status'],replay_verified=ok) for p,v,ok in checked])
    if review['status']=='experiment_reviewed':
        if not verified:raise ValueError('numerical review requires its independent replay')
        _check_evaluation(review['evaluation'],protocol,result)
        info.update(evaluation_available=True,evaluation=review['evaluation'],numerical_evaluation_recomputed=True,
            evaluation_execution='original_complete_review')
    numeric_path=reviews/f'{run_id}_positive_surface/receipt.json'
    reused_path=reviews/f'{run_id}_evaluation_reused.json'
    if numeric_path.is_file() and reused_path.is_file():
        raise ValueError('ambiguous original numeric and reuse receipts for one integration slot')
    if numeric_path.is_file():
        numeric=_numeric_supplement(numeric_path,episode,manifest_sha,manifest,result,protocol,path,review)
        info.update(evaluation_available=True,evaluation=numeric['evaluation'],numerical_evaluation_recomputed=True,
            evaluation_execution='original_positive_face_numeric_supplement',numerical_evaluation_elapsed_s=numeric['elapsed_s'],
            numeric_receipt_path=str(numeric_path),numeric_receipt_sha256=sha(numeric_path),
            original_failed_review_retained=True,positive_face_diagnosis=numeric['diagnosis'])
    if reused_path.is_file():
        reuse=read(reused_path);runtime=reuse.get('runtime',{})
        if (not verified or reuse.get('schema')!='semantic.experiment.identical_input_evaluation_reuse.v1'
                or reuse.get('status')!='experiment_endpoint_evaluation_reused' or reuse.get('run_id')!=run_id
                or reuse.get('episode_manifest_sha256')!=manifest_sha
                or reuse.get('source_sha256')!=manifest['source_sha256']
                or Path(reuse.get('episode_root','')).resolve()!=episode
                or reuse.get('target_review_sha256')!=sha(path)
                or Path(reuse.get('target_review_path','')).resolve()!=path.resolve()
                or reuse.get('target_replay')!=review['replay']
                or reuse.get('numerical_evaluation_recomputed') is not False
                or reuse.get('independent_replay_performed_here') is not False
                or runtime.get('no_new_world_or_sensor_action') is not True
                or 'before' not in runtime or runtime['before']!=runtime.get('after')):
            raise ValueError('reuse receipt target episode/replay binding differs')
        source_run=reuse['source_run_id']
        entries=[row for row in ledger['entries'] if row['run_id']==source_run]
        if source_run not in RUNS or len(entries)!=1 or source_run==run_id:
            raise ValueError('reuse source must be another uniquely recorded integration episode')
        source_episode=Path(entries[0]['metadata']['output']).resolve()
        sm=_verify_manifest(source_episode,entries[0]);source_pin=sha(source_episode/'artifact_manifest.json')
        source_result=read(source_episode/'result.json');source_protocol=read(source_episode/'protocol.json')
        source_review_path=Path(reuse['source_review_path']);source_review=read(source_review_path)
        if (reuse['source_episode_manifest_sha256']!=source_pin or sha(source_review_path)!=reuse['source_review_sha256']
                or sm['source_sha256']!=manifest['source_sha256'] or source_protocol!=protocol
                or not _verify_review(source_review,source_run,source_pin,sm,source_result)
                or reuse['source_replay']!=source_review['replay']):
            raise ValueError('reuse source review, episode or protocol differs')
        supplemental=reuse['source_numeric_supplement'];source_numeric_path=Path(supplemental['path'])
        if sha(source_numeric_path)!=supplemental['sha256']:
            raise ValueError('reuse source numerical supplement pin differs')
        numeric=_numeric_supplement(source_numeric_path,source_episode,source_pin,sm,source_result,
            source_protocol,source_review_path,source_review)
        proof=reuse['equivalence_proof']
        if (proof.get('complete_source_and_target_saved_policy_TSDF_replays_verified') is not True
                or proof.get('identical_complete_source_closure') is not True
                or proof.get('evaluation_configuration')!=protocol['evaluation']
                or sha(Path(reuse['supplemental_source_copy']))!=reuse['supplemental_source_sha256']):
            raise ValueError('reuse comparison/source proof differs')
        for name in ('prediction/mesh.npz','prediction/occupancy.npz'):
            row=proof['exact_original_arrays'][name]
            if (row['source_file_sha256']!=sha(source_episode/name) or row['target_file_sha256']!=sha(episode/name)
                    or not all(item.get('values_dtype_shape_and_bits_identical') is True for item in row['arrays'].values())):
                raise ValueError('reuse exact-array proof no longer binds complete predictions')
        copied=deepcopy(numeric['evaluation'])
        copied.update(task_success=result['status']=='controller_stop' and result['sensor_status'].get('returned_xy_and_yaw') is True,
            original_episode_status=result['status'],numerical_evaluation_recomputed=False,independent_replay_performed_here=False)
        if reuse['evaluation']!=copied:raise ValueError('reused metrics differ from original numerical source')
        _check_evaluation(copied,protocol,result)
        info.update(evaluation_available=True,evaluation=copied,numerical_evaluation_recomputed=False,
            evaluation_execution='exact_input_numeric_reuse',numeric_receipt_path=str(reused_path),
            numeric_receipt_sha256=sha(reused_path),numerical_reuse_elapsed_s=reuse['elapsed_s'],
            reused_from_run_id=source_run,reused_numeric_source_path=str(source_numeric_path),
            reused_numeric_source_sha256=sha(source_numeric_path))
    return info


def summarize_run(root,run_id,ledger):
    rows=[row for row in ledger['entries'] if row['run_id']==run_id]
    if not rows:return dict(run_id=run_id,status='not_started',complete_episode=False,terminal_attempt=False)
    if len(rows)!=1:raise ValueError('duplicate start reservation')
    entry=rows[0];episode=Path(entry['metadata']['output']).resolve()
    if entry['status']=='reserved_before_factory':
        return dict(run_id=run_id,status='reserved_or_running',complete_episode=False,terminal_attempt=False)
    terminal=episode/('attempt_failure.json' if entry['status']=='experiment_attempt_failed' else 'result.json')
    if sha(terminal)!=entry['result_sha256']:raise ValueError('terminal result differs from durable ledger')
    result=read(terminal)
    if result['status'] not in COMPLETE or not (episode/'artifact_manifest.json').is_file():
        return dict(run_id=run_id,status=result['status'],complete_episode=False,terminal_attempt=True,
            world_created=entry.get('world_created'),terminal_sha256=sha(terminal),failure=result,
            interpretation='Retained failed attempt; no missing endpoint is replaced or treated as success.')
    manifest=_verify_manifest(episode,entry);manifest_sha=sha(episode/'artifact_manifest.json')
    protocol=read(episode/'protocol.json')
    if manifest['protocol_sha256']!=ledger['protocol_sha256'] or protocol['slots']!=ledger['slots']:
        raise ValueError('episode and finite protocol ledger disagree')
    if result.get('error') is not None or result.get('finalization_errors'):raise ValueError('hidden terminal acquisition error')
    runtime=read(episode/'runtime.json');count=result['acquired_and_saved_packets']
    expected=dict(worlds_created=1,rgbd_frames=count,scans=count,paid_actions=count-1,blocked_before_world_creation=0)
    if any(runtime['after'][key]-runtime['before'][key]!=value for key,value in expected.items()):
        raise ValueError('saved acquisition counts differ from live runtime receipt')
    parsed=parse_steps(_records(episode,result,manifest,protocol))
    actions=[row['controller_action'] for row in result['actions']]
    if len(actions)!=count-1:raise ValueError('saved paid action count differs from frames')
    review_info=_review_info(root,run_id,episode,manifest_sha,manifest,result,protocol,ledger)
    return dict(run_id=run_id,method=protocol['slots'][run_id]['method'],status=result['status'],
        complete_episode=True,terminal_attempt=True,episode_root=str(episode),episode_manifest_sha256=manifest_sha,
        result_sha256=sha(terminal),executed_paid_actions=result['executed_paid_actions'],saved_packets=count,
        collisions=result['collisions'],returned_xy_and_yaw=result['sensor_status'].get('returned_xy_and_yaw'),
        execution_elapsed_s=result['elapsed_s'],timings=result['timings'],
        episode_bytes=sum(p.stat().st_size for p in episode.rglob('*') if p.is_file()),
        mesh_file_sha256=sha(episode/'prediction/mesh.npz'),occupancy_file_sha256=sha(episode/'prediction/occupancy.npz'),
        action_sequence_sha256=digest(actions),pose_sequence_sha256=digest(parsed['_poses']),
        review=review_info,**parsed,_actions=actions)


def pair_summary(left,right):
    result=dict(left=left['run_id'],right=right['run_id'],comparable_complete_trajectories=False)
    if not left.get('complete_episode') or not right.get('complete_episode'):return result
    result.update(comparable_complete_trajectories=True,
        identical_action_sequences=left['_actions']==right['_actions'],
        identical_pose_sequences=left['_poses']==right['_poses'],
        identical_endpoint_mesh_bytes=left['mesh_file_sha256']==right['mesh_file_sha256'],
        identical_endpoint_occupancy_bytes=left['occupancy_file_sha256']==right['occupancy_file_sha256'])
    divergence=None
    for index in range(max(len(left['_actions']),len(right['_actions']))):
        a=left['_actions'][index] if index<len(left['_actions']) else None
        b=right['_actions'][index] if index<len(right['_actions']) else None
        if a!=b:
            divergence=dict(executed_paid_step=index+1,left_action=a,right_action=b,
                preceding_decision_paid_step=index,
                left_prior_decision=left['_decisions'][index] if index<len(left['_decisions']) else None,
                right_prior_decision=right['_decisions'][index] if index<len(right['_decisions']) else None)
            break
    result['first_executed_action_divergence']=divergence
    macro=None
    for index,(a,b) in enumerate(zip(left['_decisions'],right['_decisions'])):
        if a['macro_target']!=b['macro_target']:
            macro=dict(paid_step=index,left_target=a['macro_target'],right_target=b['macro_target']);break
    result['first_macro_target_divergence']=macro
    lmetrics=left.get('review',{}).get('evaluation',{});rmetrics=right.get('review',{}).get('evaluation',{})
    if lmetrics and rmetrics:
        lv=lmetrics['metrics']['J_nav'];rv=rmetrics['metrics']['J_nav']
        result.update(observed_J_nav_difference_right_minus_left=rv-lv,
            observed_relative_J_nav_difference=(rv-lv)/lv if lv else None,
            difference_is_single_seen_scene_descriptive_only=True)
    return result


def summarize(root,*,allow_partial=False):
    root=Path(root).resolve();ledger=read(root/'start_ledger.json')
    if ledger.get('schema')!='semantic.phase_start_ledger.v1' or set(ledger['slots'])!=set(RUNS):
        raise ValueError('the three declared integration slots G/B/S are required')
    rows=[summarize_run(root,run,ledger) for run in RUNS]
    terminal=all(row['terminal_attempt'] for row in rows)
    if not terminal and not allow_partial:raise ValueError('all three attempts must finish before final descriptive summary')
    pairs=[pair_summary(rows[i],rows[j]) for i,j in ((0,1),(0,2),(1,2))]
    for row in rows:
        for key in ('_actions','_poses','_decisions'):row.pop(key,None)
    return dict(schema='semantic.integration.descriptive_summary.v1',source_sha256=sha(Path(__file__)),
        phase_id=ledger['phase_id'],protocol_sha256=ledger['protocol_sha256'],
        ledger_sha256=sha(root/'start_ledger.json'),all_declared_attempts_terminal=terminal,
        all_declared_episodes_complete=all(row['complete_episode'] for row in rows),
        all_complete_policy_replays_verified=all(row.get('review',{}).get('saved_policy_replay_verified',False) for row in rows),
        all_endpoint_numerical_results_available=all(row.get('review',{}).get('evaluation_available',False) for row in rows),
        numerical_evaluations_recomputed=sum(row.get('review',{}).get('numerical_evaluation_recomputed',False) for row in rows),
        numerical_evaluations_reused=sum(row.get('review',{}).get('evaluation_execution')=='exact_input_numeric_reuse' for row in rows),
        episodes=rows,pairs=pairs,summary_zero_tolerance=EPS,
        new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,new_planner_executions=0,
        primary_experiment_started=False,independent_parent_scenes=1,
        semantic_information_value_established=False,algorithm_increment_established=False,
        limitations=[
            'Three integration/cost trajectories on already-seen DEV_A_00, not the six-parent primary comparison or held-out confirmation.',
            'Observed instance IDs are frontend associations, not ground-truth facility identities; counts can include fragmentation or misses.',
            'Forecast view counts are score evaluations; accepted paid association views are reported separately.',
            'Same-class eligibility uses the shared frontend ledger even for G, which deliberately hides labels from its planning belief; coexisting ledger instances need not be visible in the same camera view.',
            'A lack of qualified same-class peers on these paid trajectories does not establish that the physical scene lacks repeated-category facilities; DEV_A has four facilities including two cabinets.',
            'A selected direct target can mix coverage and predicted instance gain; a positive instance forecast at that target does not prove that instance gain caused the choice.',
            'Frontend geometry corrections may exist while a method intentionally disables them in the planning belief.',
            'Positive total EVI and cross-future enablement do not identify the net cross-instance increment; a matched no-cross ablation is required.',
            'Positive proxy scores, belief changes and action divergence do not alone prove measured quality improvement.',
            'Any displayed J difference is descriptive for this single seen scene and is not a development-gate decision.'
        ])


def self_test():
    # Explicitly synthetic parser fixtures only; never saved as experiment data.
    choice=dict(kind='diagnose_then_observe',target={'node':'d','heading':0},score=.12,evi=.02,
                future_information_used=True,cross_instance_future_used=True)
    evidence=dict(association={'accepted':[{'instance_id':'synthetic_i','novel_support_voxels':3}],
        'instances':[dict(instance_id=key,observed_class='synthetic_class',distinct_class_supports={'synthetic_class':support},
            association_uncertain=uncertain,class_conflict=False) for key,support,uncertain in
            [('synthetic_i',2,False),('synthetic_peer',2,False),('synthetic_unqualified',1,False),('synthetic_uncertain',3,True)]]},
        view_evidence={'results':[{'instance_id':'synthetic_i','current_plane_fit':{'accepted':True,'reason':'synthetic'}}]},
        observed_residual={'results':[{'accepted':True,'informative':True,'reason':'synthetic'}]},
        geometry_feedback=[{'instance_id':'synthetic_i','applied':True}],
        structure_belief={'share_across_instances':True,'instances':[dict(instance_id='synthetic_i',
            observed_class='synthetic_class',structure_probabilities=[.8,.2],geometry_log_evidence=[0.,-1.],
            rho=.8,peer_instance_ids=['synthetic_peer'],semantic_conditioning_used=True)]})
    selection=dict(candidate_pool={'selected':2},direct_options=[],diagnostic_options=[choice],selected=choice,
        forecasts=[dict(instance_id='synthetic_i',semantic_conditioning_used=True,candidates=[dict(
            view_id='d:0',fallback=False,repeated_view_excluded=False,expected_new_surface_area_m2=1.)])])
    step=dict(controller_evidence=evidence,decision=dict(action='left',macro_id=1,macro_target=choice['target'],
        global_replanned=True,global_selection=selection))
    parsed=parse_steps([(0,step,{'execution':{'pose_xyyaw_rad':[0.,0.,0.]}})])
    assert parsed['counts']['selected_diagnostics_with_positive_evi']==1
    assert parsed['counts']['peer_adjusted_reliability_instance_frames']==1
    assert parsed['cross_instance_future_net_gain_nonzero'] is None
    assert parsed['observed_instances'][0]['distinct_paid_camera_poses']==1
    assert parsed['same_class_opportunity']['max_same_class_qualified_coexisting_instances']==2
    assert parsed['same_class_opportunity']['max_same_class_qualified_current_view_instances']==1
    assert parsed['same_class_opportunity']['qualified_unique_instance_count']==2
    left=dict(run_id='synthetic_left',complete_episode=True,_actions=['left'],_poses=[[0.,0.,0.]],
        _decisions=parsed['_decisions'],mesh_file_sha256='x',occupancy_file_sha256='y')
    right=deepcopy(left);right.update(run_id='synthetic_right',_actions=['right'])
    pair=pair_summary(left,right)
    assert pair['first_executed_action_divergence']['executed_paid_step']==1
    manifest=dict(source_sha256={'synthetic.py':'x'})
    original=dict(schema='semantic.experiment.review.v1',run_id='integration_A_G',phase_id='semantic_integration_20260923',
        status='experiment_replay_verified',episode_manifest_sha256='pin',source_sha256=manifest['source_sha256'],
        current_sources_match=True,current_source_differences=[],runtime=dict(before={},after={},no_new_world_or_sensor_action=True),
        replay=dict(status='verified',verification_kind='same_driver_saved_packet_policy_and_TSDF_replay',
            prediction_verified=True,occupancy_exact=True,frames_verified=1,terminal_status_verified='controller_stop',
            mesh_order_invariant_tolerance_m=1e-9,counterfactual_trajectory=False,new_worlds=0,
            new_sensor_packets=0,physical_actions=0,runtime=dict(before={},after={})))
    terminal=dict(status='controller_stop',acquired_and_saved_packets=1)
    assert _verify_review(original,'integration_A_G','pin',manifest,terminal)
    for field,value in (('episode_manifest_sha256','different'),('source_sha256',{})):
        changed=deepcopy(original);changed[field]=value
        try:_verify_review(changed,'integration_A_G','pin',manifest,terminal)
        except ValueError:pass
        else:raise AssertionError('changed review binding accepted')
    changed=deepcopy(original);changed['replay']['occupancy_exact']=False
    try:_verify_review(changed,'integration_A_G','pin',manifest,terminal)
    except ValueError:pass
    else:raise AssertionError('incomplete replay accepted')
    return dict(status='parser_self_test_passed',input_kind='synthetic_in_memory_only',new_worlds=0,
        experiment_data_written=False,checks=['logged_proxy_activation','unknown_cross_net_increment',
            'paid_view_separation','qualified_same_class_coexistence_vs_current_view','first_executed_action_divergence',
            'strict_review_episode_and_source_pin','complete_replay_contract'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('audit_results/semantic_integration_20260923'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),sort_keys=True))
    else:
        if args.output is None:parser.error('--output is required for a saved summary')
        value=summarize(args.root,allow_partial=args.allow_partial)
        if args.output.resolve().is_relative_to((args.root/'episodes').resolve()):
            parser.error('summary output must be outside immutable episodes')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({key:val for key,val in value.items() if key not in ('episodes','pairs')},sort_keys=True))
