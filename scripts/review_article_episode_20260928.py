#!/usr/bin/env python3
"""Read-only receipt audit of a completed article episode; no policy or TSDF.

An active reservation is pending, never a failed episode. Review outputs are
exclusive-created outside the episode. Saved sensor arrays are decoded only
to verify hashes, paid-motion accounting and graph return reserves.
"""
import argparse
from collections import Counter, deque
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import SensorStepV41, return_pose_matches, runtime_counts_v41
from nso.episode_driver_v43 import ACTION_TO_SENSOR_V43, validate_step_v43
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_mapper_v42 import _scan_copy
from nso.primitive_navigation_v41 import PrimitiveStateV41, PublicPrimitiveGraphV41
from utils.rgbd_contract import PlanarScan

METHODS={'NBV':'G','G':'G','B':'bayes_semantic','S':'shared_semantic',
    'S_no_feedback':'shared_no_feedback','S_no_future':'shared_no_future',
    'S_no_cross_future':'shared_no_cross_future'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def safe_file(root,name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('relative artifact name required')
    target=root/path
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError('artifact escapes immutable input root')
    return target


class Checks:
    def __init__(self):
        self.count=0;self.failures=[]

    def check(self,condition,name,detail=None):
        self.count+=1
        if not condition:
            self.failures.append(dict(check=name,detail=detail))
        return bool(condition)

    def require(self,condition,name,detail=None):
        if not self.check(condition,name,detail):
            raise ValueError(name)


def verify_files(root,files,checks,label):
    for name,row in files.items():
        p=safe_file(root,name)
        checks.require(p.is_file(),label+':exists:'+name)
        checks.require(p.stat().st_size==row['bytes'],label+':bytes:'+name)
        checks.require(sha(p)==row['sha256'],label+':sha:'+name)


def packet_at(episode,index):
    prefix=f'packets/{index:03d}';record=read(episode/(prefix+'_receipt.json'))
    with np.load(episode/(prefix+'_rgbd.npz'),allow_pickle=False) as data:
        fields={key:data[key].copy() for key in PaidRGBDObservationV40.__dataclass_fields__}
    for key in ('frame_id','paid_step'):fields[key]=fields[key].item()
    observation=PaidRGBDObservationV40.from_mapping(fields)
    scan=PlanarScan.load(episode/(prefix+'_scan.npz'))
    return SensorStepV41(observation,scan,record['execution']),record


def distance_queries(graph):
    """Unweighted public-graph distances only; no controller/view selection."""
    cache={}
    def distance(a,b):
        if a not in cache:
            found={a:0};queue=deque([a])
            while queue:
                current=queue.popleft()
                for action in ('forward','left','right'):
                    try:target=graph.successor(current,action)
                    except ValueError:continue
                    if target not in found:
                        found[target]=found[current]+1;queue.append(target)
            cache[a]=found
        return cache[a].get(b)
    return distance


def state(value):
    return PrimitiveStateV41(value['node'],value['heading'])


def csv_bytes(rows):
    if not rows:return b''
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def review(episode,output):
    episode=Path(episode).resolve();output=Path(output).resolve()
    if output==episode or output.is_relative_to(episode):
        raise ValueError('review output must be separate from the immutable episode')
    before=runtime_counts_v41();checks=Checks();phase=episode.parent.parent
    ledger_path=phase/'start_ledger.json';ledger=read(ledger_path)
    matches=[r for r in ledger['entries'] if r['run_id']==episode.name]
    checks.require(len(matches)==1,'one ledger reservation')
    entry=matches[0]
    if entry['status']=='reserved':
        return dict(status='pending',run_id=episode.name,reason='active reservation; no terminal audit written',
                    new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0)
    checks.require(not output.exists(),'exclusive review output')
    summary=dict(schema='article.saved_episode_review.v1',run_id=episode.name,
                 episode=str(episode.relative_to(ROOT)) if episode.is_relative_to(ROOT) else str(episode),
                 online_status=entry['status'],new_worlds=0,new_sensor_queries=0,
                 new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0)
    paths=[];replans=[];events=[];source_pins={};manifest_sha=None
    def event(kind,index,instance=None,**fields):
        events.append(dict(kind=kind,paid_step=index,instance_id=instance,**fields))
    try:
        if not (episode/'artifact_manifest.json').exists():
            failure=episode/'attempt_failure.json'
            checks.require(failure.is_file(),'retained terminal failure record')
            checks.require(sha(failure)==entry['result_sha256'],'failure bound to terminal ledger')
            summary.update(status='failed_attempt_preserved',failure=read(failure),qualified=False,
                           limitation='no complete artifact manifest; no completed-episode quality claim')
        else:
            manifest_sha=sha(episode/'artifact_manifest.json')
            checks.require(manifest_sha==entry['artifact_manifest_sha256'],'manifest bound to terminal ledger')
            manifest=read(episode/'artifact_manifest.json');files=manifest['files']
            verify_files(episode,files,checks,'episode')
            checks.check({str(p.relative_to(episode)) for p in episode.rglob('*') if p.is_file()}
                         ==set(files)|{'artifact_manifest.json'},'complete closed artifact inventory')
            protocol=read(episode/'protocol.json');protocol_sha=sha(episode/'protocol.json')
            checks.require(protocol_sha==manifest['protocol_sha256']==ledger['protocol_sha256'],
                           'archived protocol bound to ledger and manifest')
            slot=protocol['slots'][episode.name];method=slot['method'];budget=slot['budget']
            started=read(episode/'started.json')
            checks.check(started['slot']==slot,'started slot matches protocol')
            source_pins=protocol['source_sha256']
            checks.require(source_pins==manifest['source_sha256']==started['source_sha256'],
                           'three source-pin copies agree')
            archive=safe_file(ROOT,protocol['source_archive'])
            checks.require(sha(archive)==protocol['source_archive_sha256'],'source archive hash')
            with zipfile.ZipFile(archive) as z:
                checks.require(set(z.namelist())==set(source_pins),'complete source archive inventory')
                for name,pin in source_pins.items():
                    checks.require(hashlib.sha256(z.read(name)).hexdigest()==pin,'source archive:'+name)
            summary['current_source_mismatches']=[name for name,pin in source_pins.items()
                if not (ROOT/name).is_file() or sha(ROOT/name)!=pin]
            checks.check(not summary['current_source_mismatches'],'current source closure still frozen')
            reference_entry=protocol['references'][slot['scene_id']]
            reference_root=safe_file(ROOT,reference_entry['root'])
            checks.require(sha(reference_root/'manifest.json')==reference_entry['manifest_sha256'],
                           'reference manifest pin')
            verify_files(reference_root,read(reference_root/'manifest.json'),checks,'reference')
            reference=read(reference_root/'reference.json')
            checks.check(reference['asset_manifest_sha256']==protocol['asset_manifest_sha256'],
                         'reference and scene asset binding')
            asset_root=safe_file(ROOT,protocol['asset_root'])
            checks.require(sha(asset_root/'manifest.json')==protocol['asset_manifest_sha256'],'scene bundle pin')
            asset_manifest=read(asset_root/'manifest.json')
            for name,pin in asset_manifest['artifact_sha256'].items():
                if name.startswith(slot['scene_id']+'/'):
                    checks.require(sha(safe_file(asset_root,name))==pin,'bound scene artifact:'+name)
            for saved,original in (('public_graph.json','graph.json'),
                                   ('public_workspace.json','public_workspace.json')):
                checks.check(read(episode/saved)==read(asset_root/slot['scene_id']/original),
                             'runtime public asset contents:'+saved)
            result=read(episode/'result.json');evaluation=read(episode/'evaluation.json')
            checks.require(sha(episode/'result.json')==entry['result_sha256'],'result ledger binding')
            checks.check(result['status']==entry['status'],'terminal status binding')
            final=read(episode/'controller_final.json');spec=read(episode/'public_spec.json')
            checks.check(spec['task']['max_actions']==budget,'runtime sensor paid budget')
            checks.check(final['article_method']==method and final['method']==METHODS[method],
                         'method and internal policy binding')
            configuration=final['configuration']
            checks.check(configuration['actual_structure_feedback']==(method!='S_no_feedback'),
                         'feedback ablation policy')
            checks.check(configuration['future_information']==(method!='S_no_future'),
                         'future ablation policy')
            graph=PublicPrimitiveGraphV41(read(episode/'public_graph.json'));distance=distance_queries(graph)
            count=result['acquired_and_saved_packets'];encoding=read(episode/'encoding.json')['steps']
            encoding={r['artifact']:r for r in encoding}
            checks.require(set(encoding)=={f'steps/{i:03d}.json.gz' for i in range(count)},
                           'complete consecutive saved steps')
            checks.require({n for n in files if n.startswith('packets/')}==
                {f'packets/{i:03d}_{suffix}' for i in range(count)
                 for suffix in ('rgbd.npz','scan.npz','receipt.json')},'complete paid packet inventory')
            previous_pose=None;previous_action=None;home=None;total_distance=0.;minimum_slack=budget
            action_counts=Counter();semantic_steps=set();feedback_steps=set();plane_steps=set()
            for i in range(count):
                step,packet_record=packet_at(episode,i);obs=step.rgbd
                name=f'steps/{i:03d}.json.gz';plain=gzip.decompress((episode/name).read_bytes())
                checks.require(len(plain)==encoding[name]['uncompressed_bytes'],'decoded step length:'+str(i))
                log=json.loads(plain);evidence=log['controller_evidence'];decision=log['decision']
                expected='initial_observation' if i==0 else ACTION_TO_SENSOR_V43[previous_action]
                verified=validate_step_v43(step,expected_step=i,expected_action=expected,
                                          expected_previous_pose=previous_pose)
                checks.check(verified==log['accounting'],'actual saved motion accounting:'+str(i))
                checks.check(packet_record['observation_sha256']==obs.sha256(),'packet logical SHA:'+str(i))
                if i:
                    executed=result['actions'][i-1]
                    checks.check(executed==dict(paid_step=i,controller_action=previous_action,
                        sensor_action=expected,observation_sha256=obs.sha256(),collision=step.receipt['collision']),
                        'terminal executed-action receipt:'+str(i))
                for kind in ('rgbd','scan'):
                    checks.check(packet_record[kind+'_artifact']==files[f'packets/{i:03d}_{kind}.npz'],
                                 kind+' nested artifact binding:'+str(i))
                _,scan_sha=_scan_copy(step.scan,None if i==0 else float(i-1))
                mapping=log['mapper']
                checks.check(mapping['observation_sha256']==obs.sha256() and mapping['scan_sha256']==scan_sha,
                             'map current RGBD and scan binding:'+str(i))
                checks.check(mapping['tsdf_integrated'] and mapping['tsdf_integration_count']==i+1
                             and mapping['semantic_or_prototype_fusion'] is False,'one measured TSDF update:'+str(i))
                checks.check(evidence['paid_step']==decision['paid_step']==i
                             and evidence['observation_sha256']==decision['source_observation_sha256']==obs.sha256(),
                             'decision current paid observation:'+str(i))
                checks.check(decision['article_method']==method and decision['method']==METHODS[method],
                             'per-step method:'+str(i))
                checks.check(decision['ground_truth_scene_input'] is False and
                             decision['actual_future_sensor_rendered'] is False,'no private/future sensor input:'+str(i))
                current=graph.state_from_observation(obs);pose=step.receipt['pose_xyyaw_rad']
                blocked=evidence['safety'].get('newly_blocked_edges',[])
                for removed in blocked:
                    graph.block_observed_edge(*removed['edge'])
                if blocked:distance=distance_queries(graph)
                if home is None:home=current;first_pose=list(pose)
                back=distance(current,home)
                checks.check(back is not None and back<=budget-i,'actual full-pose return reserve:'+str(i))
                slack=None if back is None else budget-i-back
                if slack is not None:minimum_slack=min(minimum_slack,slack)
                segment=0. if previous_pose is None else float(np.linalg.norm(np.asarray(pose[:2])-previous_pose[:2]))
                total_distance+=segment
                if i:action_counts[expected]+=1
                paths.append(dict(paid_step=i,sensor_action=expected,x_m=pose[0],y_m=pose[1],yaw_rad=pose[2],
                    segment_translation_m=segment,cumulative_translation_m=total_distance,
                    paid_remaining=budget-i,return_cost_actions=back,return_slack_actions=slack,
                    collision=step.receipt['collision'],next_controller_action=decision['action']))
                routing=decision.get('routing') or {}
                if decision['action'] in ACTION_TO_SENSOR_V43:
                    after=graph.successor(current,decision['action']);after_back=distance(after,home)
                    checks.check(after_back==routing.get('return_cost_after_action') and
                                 after_back is not None and 1+after_back<=budget-i,
                                 'selected primitive reserves full return:'+str(i))
                for row in routing.get('candidates',[]):
                    target=state(row);out=distance(current,target);ret=distance(target,home)
                    cost=None if out is None or ret is None else out+1+ret
                    checks.check(row['total_with_observation_and_return']==cost
                        and row['feasible']==(cost is not None and cost<=budget-i),'macro+observe+return:'+str(i))
                belief=evidence['structure_belief']
                checks.check(belief['share_across_instances']==method.startswith('S'),'class sharing permission:'+str(i))
                for instance in belief['instances']:
                    key=instance['instance_id']
                    if method in ('G','NBV'):
                        checks.check(instance['observed_class'] is None and not instance['semantic_conditioning_used'],
                                     'geometry method does not condition on category:'+str(i)+':'+key)
                    if method=='B':checks.check(not instance['peer_instance_ids'],'B has no peer transfer:'+str(i)+':'+key)
                    if method=='S_no_feedback':checks.check(np.all(np.asarray(instance['geometry_log_evidence'])==0),
                                                           'no-feedback planning evidence:'+str(i)+':'+key)
                    if instance['semantic_conditioning_used']:
                        semantic_steps.add(i);event('semantic_conditioning',i,key,observed_class=instance['observed_class'])
                for row in evidence['association']['accepted']:
                    for key in ('actual_scene_pose','actual_dimensions','world_aabb_m','ground_truth_owner',
                                'triangle_instance_id','local_solid_boxes','true_structure','template_depths'):
                        checks.check(key not in row,'association excludes '+key+':'+str(i))
                for row in evidence.get('geometry_feedback',[]):
                    if row['applied']:
                        feedback_steps.add(i);event('measured_feedback_applied',i,row['instance_id'],reason=row['reason'])
                for key in evidence.get('first_actual_reliable_planes',[]):
                    plane_steps.add(i);event('first_reliable_plane',i,key)
                for key in evidence['view_evidence'].get('article_multiview_supplemented_instances',[]):
                    event('multiview_plane_supplement',i,key)
                selection=decision.get('global_selection')
                if decision['global_replanned']:
                    checks.require(selection is not None,'replan receipt:'+str(i))
                    checks.check(selection['paid_step']==i and selection['remaining']==budget-i,
                                 'replan current budget:'+str(i))
                    checks.check(selection['future_sensor_rendered'] is False,'replan predicts only:'+str(i))
                    if method=='NBV':checks.check(not selection['diagnostic_options'],'NBV no diagnosis lookahead:'+str(i))
                    selected=selection.get('selected')
                    alternatives=selection.get('direct_options',[])+selection.get('diagnostic_options',[])
                    if selected and selected.get('kind')=='measurement_initialization':alternatives=[selected]
                    if not alternatives:alternatives=[dict(kind='no_feasible_target',target=None)]
                    for row in alternatives:
                        target=row.get('target') or {};kind=row['kind'];selected_flag=(row==selected)
                        total=row.get('total_cost');out=ret=None
                        if target:
                            out=distance(current,state(target));ret=distance(state(target),home)
                        if total is not None:
                            checks.check(out is not None and ret is not None and total==out+1+ret
                                         and total<=budget-i,'replan direct/initialization cost:'+str(i))
                        if kind=='direct':checks.check(math.isclose(row['score'],row['expected_gain']/total,
                            abs_tol=1e-10,rel_tol=1e-10),'direct score arithmetic:'+str(i))
                        for follow in row.get('followups',[]):
                            last=state(follow['target']);middle=distance(state(target),last);tail=distance(last,home)
                            checks.check(out is not None and middle is not None and tail is not None
                                and follow['total_cost']==out+1+middle+1+tail and follow['total_cost']<=budget-i,
                                'diagnostic two-paid-observation reserve:'+str(i))
                        replans.append(dict(paid_step=i,kind=kind,selected=selected_flag,
                            target_node=target.get('node'),target_heading=target.get('heading'),
                            score=row.get('score'),expected_gain=row.get('expected_gain'),evi=row.get('evi'),
                            scheduling_priority=row.get('scheduling_priority'),total_cost=total,
                            followup_count=len(row.get('followups',[])),remaining=budget-i))
                    if selected:event('replan_selected',i,selected.get('instance_id'),kind_selected=selected['kind'],
                                      target=selected['target'],score=selected.get('score'),total_cost=selected.get('total_cost'))
                previous_pose=list(pose);previous_action=decision['action']
            returned=return_pose_matches(previous_pose,first_pose)
            checks.check(result['sensor_status']['returned_xy_and_yaw']==returned,'return flag from actual initial/final pose')
            checks.check(result['executed_paid_actions']==count-1==sum(action_counts.values()),'paid action and RGBD/scan accounting')
            checks.check(result['mapper_frames']==result['mapper_tsdf_integrations']==count,'terminal one integration per frame')
            checks.check(result['collisions']==sum(r['collision'] for r in paths),'saved collision total')
            checks.check(result['sensor_status']['remaining_actions']==budget-(count-1),'remaining budget contract')
            checks.check(count-1<=budget,'episode within paid budget')
            runtime=read(episode/'runtime.json')
            delta={key:runtime['after'][key]-runtime['before'][key] for key in runtime['before']}
            checks.check(delta.get('worlds_created')==1 and delta.get('rgbd_frames')==count
                         and delta.get('scans')==count and delta.get('paid_actions')==count-1,'runtime sensor-cost counters')
            prediction=read(episode/'prediction_seal.json')
            checks.require(set(prediction)=={'mesh.npz','occupancy.npz','mapper.json'},'three frozen prediction artifacts')
            for name,pin in prediction.items():checks.require(sha(episode/'prediction'/name)==pin,'prediction seal:'+name)
            checks.check(evaluation['source_prediction_sha256']==prediction and
                evaluation['reference_manifest_sha256']==reference_entry['manifest_sha256'],'prediction/reference evaluation binding')
            metrics=evaluation['metrics']
            checks.check(metrics['reference_fingerprint']==reference['surface']['fingerprint'],'surface reference fingerprint')
            checks.check([r['instance_id'] for r in metrics['per_instance']]==[0,1,2,3],
                         'all four fixed instances remain in denominator')
            checks.check(math.isclose(metrics['Q'],sum(r['f1'] for r in metrics['per_instance'])/4,
                abs_tol=1e-12,rel_tol=0) and math.isclose(metrics['J_nav'],metrics['C_nav']*metrics['Q'],
                abs_tol=1e-12,rel_tol=0),'macro quality and joint arithmetic')
            checks.check(evaluation['semantic_weights_used'] is False and evaluation['no_roi_crop'] is True,
                         'unweighted whole prediction evaluation contract')
            qualified=result['status']=='controller_stop' and returned and not result['collisions'] and count-1<=budget
            checks.check(evaluation['qualified']==entry['qualified']==qualified,'task qualification independent of quality')
            summary.update(status='reviewed',method=method,scene_id=slot['scene_id'],phase=protocol['phase'],
                qualified=qualified,metrics=metrics,paid_actions=count-1,budget=budget,unused_actions=budget-(count-1),
                translation_m=total_distance,turns=action_counts['turn_left']+action_counts['turn_right'],
                extra_observes=action_counts['observe'],forward_actions=action_counts['forward'],
                rgbd_frames=count,scan_frames=count,collisions=result['collisions'],returned_xy_and_yaw=returned,
                minimum_return_slack_actions=minimum_slack,final_return_slack_actions=budget-(count-1),
                action_counts=dict(action_counts),replans=sum(1 for e in events if e['kind']=='replan_selected'),
                first_semantic_conditioning_step=min(semantic_steps,default=None),
                first_applied_feedback_step=min(feedback_steps,default=None),
                first_reliable_plane_step=min(plane_steps,default=None),
                input_manifest_sha256=manifest_sha,source_archive_sha256=protocol['source_archive_sha256'],
                protocol_sha256=protocol_sha,
                scope='Saved receipt and arithmetic audit; no policy replay, reintegration or quality remeasurement')
            checks.check(sha(episode/'artifact_manifest.json')==manifest_sha,'input manifest unchanged during review')
    except Exception as exc:
        summary.update(status='review_error',error=dict(type=type(exc).__name__,message=str(exc)))
        checks.check(False,'review execution completed',str(exc))
    checks.check(runtime_counts_v41()==before,'review creates no World or physical sensor action')
    summary.update(checks=checks.count,all_checks_passed=not checks.failures,failures=checks.failures)
    if checks.failures and summary['status']=='reviewed':summary['status']='review_findings'
    output.mkdir(parents=True,exist_ok=False)
    payloads={'review.json':canonical(summary),'events.json':canonical(events),
              'trajectory.csv':csv_bytes(paths),'replans.csv':csv_bytes(replans)}
    for name,content in payloads.items():
        with (output/name).open('xb') as stream:stream.write(content)
    manifest=dict(schema='article.episode_review_manifest.v1',input_episode_manifest_sha256=manifest_sha,
        reviewer_source_sha256=sha(Path(__file__)),files={name:dict(bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest()) for name,content in payloads.items()},
        new_worlds=0,new_tsdf_integrations=0,new_policy_runs=0)
    with (output/'manifest.json').open('xb') as stream:stream.write(canonical(manifest))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=review(args.episode,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    if result['status'] not in ('pending','reviewed','failed_attempt_preserved') or result.get('all_checks_passed') is False:
        raise SystemExit(1)
