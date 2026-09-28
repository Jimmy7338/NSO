#!/usr/bin/env python3
"""Run only an explicit finite sequence; stop on technical, never quality, failure."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.semantic_experiment import COMPLETE,PhaseLedger,_repo_path,read_json
from nso.semantic_scene_experiment import file_sha256,run_experiment,validate_protocol


def check_available(protocol,run_ids,entries,output_root):
    limit=protocol.get('storage_plan',{}).get('maximum_active_batch_slots',1)
    if type(limit) is not int or not 1<=limit<=6:raise ValueError('declared active batch limit must be 1..6')
    if not 1<=len(run_ids)<=limit or len(set(run_ids))!=len(run_ids):
        raise ValueError('explicit distinct IDs within the declared finite batch cap required')
    if any(run_id not in protocol['slots'] for run_id in run_ids):raise ValueError('batch contains undeclared slot')
    attempted={row['run_id'] for row in entries}
    if attempted.intersection(run_ids):raise ValueError('entire batch rejected: a listed slot is already reserved/attempted')
    if any((output_root/run_id).exists() for run_id in run_ids):
        raise ValueError('entire batch rejected: retained output already exists')


def execute_sequence(run_ids,invoke,check_binding,emit):
    """The injected operations also permit a strictly in-memory logic fixture."""
    finished=[]
    for run_id in run_ids:
        invoked=False
        try:
            check_binding()
            emit(dict(event='slot_started',run_id=run_id))
            invoked=True
            result=invoke(run_id)
        except Exception as exc:
            result=dict(status='batch_invocation_failed',type=type(exc).__name__,message=str(exc),automatic_retry=False)
        row=dict(event='slot_finished',run_id=run_id,runner_invoked=invoked,result=result)
        emit(row);finished.append(row)
        if result.get('status') not in COMPLETE:
            return dict(status='batch_stopped_on_technical_or_resource_failure',finished=finished,
                unstarted_run_ids=run_ids[len(finished)-(not invoked):],automatic_retry=False)
    return dict(status='declared_batch_complete',finished=finished,unstarted_run_ids=[],automatic_retry=False)


def run_batch(protocol_path,run_ids,audit_dir):
    protocol_path=Path(protocol_path).resolve();pin=file_sha256(protocol_path)
    protocol=validate_protocol(read_json(protocol_path))
    if protocol['status']!='frozen':raise ValueError('explicit frozen protocol required')
    ledger=PhaseLedger(protocol,pin);ledger.check_existing()
    saved=read_json(ledger.path) if ledger.path.exists() else None
    if saved is not None and (saved.get('schema')!='semantic.phase_start_ledger.v1'
            or saved.get('phase_id')!=protocol['phase_id']
            or any(saved.get(key)!=value for key,value in ledger.binding.items())):
        raise ValueError('existing phase ledger does not match protocol')
    output_root=_repo_path(protocol['output_relative_path'])
    check_available(protocol,run_ids,[] if saved is None else saved['entries'],output_root)
    audit_dir=Path(audit_dir).resolve()
    if (not audit_dir.is_relative_to(ROOT/'audit_results') or 'episodes' in audit_dir.relative_to(ROOT).parts
            or audit_dir.is_relative_to(output_root) or output_root.is_relative_to(audit_dir) or audit_dir.exists()):
        raise ValueError('unique new audit directory outside all episode paths required')
    def check_binding():
        if file_sha256(protocol_path)!=pin:raise ValueError('protocol changed after finite batch binding')
    check_binding()
    audit_dir.mkdir(parents=True,exist_ok=False)
    protocol_bytes=protocol_path.read_bytes()
    (audit_dir/'protocol.json').write_bytes(protocol_bytes)
    if file_sha256(audit_dir/'protocol.json')!=pin:raise ValueError('protocol changed during batch archive')
    header=dict(schema='semantic.declared_batch.v1',phase_id=protocol['phase_id'],protocol_sha256=pin,
        protocol_path=str(protocol_path),run_ids=list(run_ids),slots={key:protocol['slots'][key] for key in run_ids},
        batch_script_sha256=file_sha256(__file__),started_unix_s=time.time(),
        no_implicit_extra_slots=True,no_automatic_retry=True,stop_depends_on_quality=False)
    with (audit_dir/'batch.json').open('x') as stream:
        json.dump(header,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    with (audit_dir/'events.jsonl').open('x') as stream:
        def emit(value):
            line=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
            stream.write(line+'\n');stream.flush();os.fsync(stream.fileno())
            print(line,flush=True)
        emit(dict(event='batch_started',protocol_sha256=pin,run_ids=run_ids,audit_dir=str(audit_dir)))
        result=execute_sequence(run_ids,lambda run_id:run_experiment(protocol_path,run_id),check_binding,emit)
        result.update(schema='semantic.declared_batch_result.v1',protocol_sha256=pin,
            run_ids=run_ids,finished_unix_s=time.time())
        with (audit_dir/'result.json').open('x') as terminal:
            json.dump(result,terminal,sort_keys=True,indent=2,allow_nan=False);terminal.write('\n')
            terminal.flush();os.fsync(terminal.fileno())
        emit(dict(event='batch_finished',status=result['status'],processed_entries=len(result['finished']),
            invoked_slots=sum(row['runner_invoked'] for row in result['finished']),
            unstarted_run_ids=result['unstarted_run_ids']))
    return result


def self_test():
    ids=['synthetic_A','synthetic_B','synthetic_C'];called=[];events=[]
    statuses=['stopped_without_confirmed_return','controller_blocked','budget_exhausted']
    def invoke(run_id):called.append(run_id);return dict(status=statuses[len(called)-1],fixture=True)
    result=execute_sequence(ids,invoke,lambda:None,events.append)
    assert result['status']=='declared_batch_complete' and called==ids
    called.clear();events.clear();statuses[1]='blocked_before_world_creation'
    result=execute_sequence(ids,invoke,lambda:None,events.append)
    assert called==ids[:2] and result['unstarted_run_ids']==ids[2:]
    called.clear()
    def changed():raise ValueError('synthetic protocol change')
    result=execute_sequence(ids,invoke,changed,events.append)
    assert not called and result['unstarted_run_ids']==ids
    protocol=dict(slots={key:{} for key in ids},storage_plan={'maximum_active_batch_slots':3})
    class Absent:
        def __truediv__(self,key):return self
        def exists(self):return False
    for selected,entries in ((ids,[{'run_id':ids[1]}]),([ids[0],ids[0]],[]),(['undeclared'],[])):
        try:check_available(protocol,selected,entries,Absent())
        except ValueError:pass
        else:raise AssertionError('invalid batch accepted')
    return dict(status='synthetic_batch_logic_passed',worlds_created=0,phase_slots_reserved=0,
        files_written=0,checks=['normal_unsuccessful_tasks_continue','technical_failure_stops',
            'binding_change_stops','whole_batch_existing_slot_rejection','duplicates_and_unknown_ids_rejected'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--run-id',nargs='+')
    parser.add_argument('--audit-dir',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        if any((args.protocol,args.run_id,args.audit_dir)):parser.error('self-test cannot be combined with real batch arguments')
        print(json.dumps(self_test(),sort_keys=True));return 0
    if args.protocol is None or not args.run_id or args.audit_dir is None:
        parser.error('--protocol, a finite --run-id list and unique --audit-dir are required')
    try:result=run_batch(args.protocol,args.run_id,args.audit_dir)
    except Exception as exc:
        print(json.dumps(dict(status='batch_entry_failed',type=type(exc).__name__,message=str(exc)),sort_keys=True),flush=True)
        return 2
    return 0 if result['status']=='declared_batch_complete' else 2


if __name__=='__main__':sys.exit(main())
