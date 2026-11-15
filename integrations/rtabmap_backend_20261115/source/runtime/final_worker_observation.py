"""Post-stop, read-only reconciliation of one report's worker observation log.

Never issues RPC, changes acceptance, replaces joint_summary or authorizes work.
The reported summary source is the last validated ACCEPTED observe response,
not an invented observation or an assertion about when the caller received it.
"""
import hashlib
import json
from pathlib import Path


def final_worker_observation_audit(report, supervisor_summary, *, worker_closed, stop_confirmed):
    path=Path(report).resolve()/'joint/decisions.jsonl'
    receipt=dict(status='audit_error',source_path=str(path),source_sha256=None,
                 count_source='last validated accepted observe response in this report worker journal',
                 worker_closed=worker_closed,stop_confirmed=stop_confirmed,
                 supervisor_summary_preserved=True,accepted_observations=[],unaccepted_observations=[],
                 error=None,read_only=True,execution_authority=False)
    result=dict(final_worker_observation_receipt=receipt,final_worker_summary=None,
                supervisor_worker_count_difference=None)
    try:
        if worker_closed is not True or stop_confirmed is not True:
            raise ValueError('post-stop closed-worker evidence required')
        # A same-directory journal only: do not follow a journal symlink to a
        # different report or accept packets bound to a different report.
        if path.resolve().parent != Path(report).resolve()/'joint':
            raise ValueError('worker journal belongs to another directory')
        raw=path.read_bytes()
        receipt['source_sha256']=hashlib.sha256(raw).hexdigest()
        expected=None
        previous_id=0
        last=None
        last_id=None
        last_line=None
        for line_number,line in enumerate(raw.splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            request,response=row['request'],row['response']
            reqid=request['id']
            if type(reqid) is not int or response.get('id')!=reqid or type(response.get('id')) is not int:
                raise ValueError('request/response id mismatch at line '+str(line_number))
            method=request['method']
            if reqid==-1:
                if method!='shutdown':raise ValueError('invalid shutdown request')
            elif reqid<=previous_id:
                raise ValueError('nonmonotonic request id')
            else:previous_id=reqid
            if type(response.get('ok')) is not bool:raise ValueError('invalid response status')
            if method=='init':
                if expected is not None or response['ok'] is not True:
                    raise ValueError('invalid worker initialization')
                cfg=request['params']['configuration']
                expected=(cfg['scene_id'],cfg['episode_id'])
                continue
            if method!='observe':continue
            if expected is None:raise ValueError('observe precedes worker initialization')
            params=request['params'];aid=params['action_id'];packet=Path(params['path']).resolve()
            if (type(aid)is not int or aid<0 or packet.parent!=Path(report).resolve()/'joint_inputs'
                    or not packet.is_file()):
                raise ValueError('observe action/path does not belong to this report')
            if (aid==0 and params.get('action') is not None) or (aid>0 and params.get('action') not in ('forward','left','right')):
                raise ValueError('invalid observed action')
            if response['ok'] is not True:raise ValueError('observe worker response is an error')
            value=response['result']
            if type(value.get('accepted')) is not bool:raise ValueError('missing observe acceptance evidence')
            reference=dict(request_id=reqid,line_number=line_number,action_id=aid,
                           action=params.get('action'),packet_path=str(packet))
            if value['accepted'] is False:
                receipt['unaccepted_observations'].append(reference)
                continue
            state,summary=value['state'],value['summary']
            runtime=summary['runtime'];ledger=runtime['modules']['feedback']
            paid=summary['actual_paid_actions'];consumed=ledger['consumed_action_ids']
            pose=state.get('pose')
            if (type(paid)is not int or paid<0 or type(state.get('action_id'))is not int
                    or type(state.get('step'))is not int or state.get('action_id')!=aid or state.get('step')!=aid
                    or not isinstance(pose,list) or len(pose)!=3
                    or any(type(v)is not int for v in pose) or pose[2]not in range(4)
                    or paid!=aid or type(ledger.get('paid_actions'))is not int
                    or type(ledger.get('last_action_id'))is not int or ledger.get('paid_actions')!=paid
                    or ledger.get('last_action_id')!=aid
                    or not isinstance(consumed,list) or any(type(v)is not int for v in consumed)
                    or consumed!=list(range(1,paid+1))
                    or (runtime.get('scene_id'),runtime.get('episode_id'))!=expected
                    or packet.name!='joint-'+str(state['frame_id'])+'.npz'):
                raise ValueError('accepted observe/state/summary disagreement')
            if last is not None and paid<last['actual_paid_actions']:
                raise ValueError('accepted observation count regressed')
            reference.update(frame_id=state['frame_id'],worker_paid=paid,state=state)
            receipt['accepted_observations'].append(reference)
            last,last_id,last_line=summary,reqid,line_number
        if last is None:
            receipt.update(status='no_accepted_observation',error='no accepted observe summary; count unknown')
            return result
        supervisor=supervisor_summary['actual_paid_actions']
        supervisor_runtime=supervisor_summary['runtime']
        if (type(supervisor)is not int or supervisor<0 or
                (supervisor_runtime.get('scene_id'),supervisor_runtime.get('episode_id'))!=expected):
            raise ValueError('supervisor and worker episode identity differ')
        paid=last['actual_paid_actions']
        if supervisor>paid:raise ValueError('worker evidence behind supervisor; unresolved count')
        receipt.update(status='verified',worker_paid=paid,supervisor_paid=supervisor,
                       final_summary_request_id=last_id,final_summary_line=last_line,
                       supervisor_missing_paid_actions=list(range(supervisor+1,paid+1)))
        result['final_worker_summary']=last
        result['supervisor_worker_count_difference']=dict(supervisor_paid=supervisor,worker_paid=paid,
            worker_minus_supervisor=paid-supervisor,worker_source_request_id=last_id,
            count_source=receipt['count_source'])
    except Exception as exc:
        receipt['error']=type(exc).__name__+': '+str(exc)
        # Partial accepted rows remain diagnostic; never promote partial counts.
        result['final_worker_summary']=None
        result['supervisor_worker_count_difference']=None
    return result
