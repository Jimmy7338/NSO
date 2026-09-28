#!/usr/bin/env python3
"""Execute an explicitly declared finite queue, one <=3-slot batch at a time.

No slot discovery, resume, retry, scientific matrix generation, or worker pool.
Peer failure files are checked only between batches; running work is not killed.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.semantic_experiment import PhaseLedger,_repo_path,read_json
from nso.semantic_scene_experiment import file_sha256,validate_protocol
from scripts import run_semantic_declared_batch as batch_runner


SCHEMA='semantic.finite_queue.v1'
RESULT_SCHEMA='semantic.finite_queue.result.v1'
COMPLETE_STATUS='declared_finite_queue_complete'


def validate_schedule(protocol,batches,*,attempted_run_ids=(),existing_run_ids=()):
    """Pure finite-inventory validation; never infer or skip a pending slot."""
    if not isinstance(batches,list) or not 1<=len(batches)<=len(protocol['slots']):
        raise ValueError('nonempty finite explicit batch list required')
    active=protocol.get('storage_plan',{}).get('maximum_active_batch_slots',1)
    if type(active) is not int or not 1<=active<=6:
        raise ValueError('protocol active batch cap must be 1..6')
    seen=set();audits=set();ordered=[]
    for row in batches:
        if not isinstance(row,dict) or set(row)!={'run_ids','audit_dir'}:
            raise ValueError('each batch requires only explicit run_ids and audit_dir')
        ids=row['run_ids'];audit=row['audit_dir']
        if (not isinstance(ids,list) or not 1<=len(ids)<=min(3,active)
                or any(not isinstance(key,str) or key not in protocol['slots'] for key in ids)):
            raise ValueError('each batch must declare 1..3 known slots within protocol cap')
        if len(set(ids))!=len(ids) or seen.intersection(ids):
            raise ValueError('duplicate run ID anywhere in finite queue')
        if not isinstance(audit,str) or not audit or audit in audits:
            raise ValueError('each batch requires a unique explicit audit directory')
        seen.update(ids);audits.add(audit);ordered.extend(ids)
    if seen.intersection(attempted_run_ids) or seen.intersection(existing_run_ids):
        raise ValueError('entire queue rejected: listed slot already reserved/attempted or retained')
    return ordered


def execute_queue(batches,invoke,check_binding,peer_stop,emit):
    """Pure sequential control, with injected I/O for bounded memory tests."""
    finished=[]
    for index,batch in enumerate(batches):
        invoked=False
        try:
            check_binding()
            peer=peer_stop()
            if peer is not None:
                emit(dict(event='queue_stopped_at_peer_boundary',next_batch=index,peer=peer))
                return dict(status='queue_stopped_on_peer_signal',finished=finished,peer_stop=peer,
                    unstarted_run_ids=[key for row in batches[index:] for key in row['run_ids']],
                    unresolved_current_batch_run_ids=[],automatic_retry=False)
            emit(dict(event='batch_started',batch_index=index,**batch))
            invoked=True
            result=invoke(batch)
        except Exception as exc:
            result=dict(status='queue_batch_invocation_failed',type=type(exc).__name__,message=str(exc),
                automatic_retry=False)
        row=dict(event='batch_finished',batch_index=index,run_ids=list(batch['run_ids']),
            audit_dir=batch['audit_dir'],batch_invoked=invoked,result=result)
        emit(row);finished.append(row)
        if result.get('status')!='declared_batch_complete':
            # An exception after invocation can have left paid work behind. Do
            # not falsely call that whole batch unstarted or try to recover it.
            partial=result.get('unstarted_run_ids')
            unknown=invoked and partial is None
            unstarted=([] if unknown else list(partial or ([] if invoked else batch['run_ids'])))
            unstarted += [key for later in batches[index+1:] for key in later['run_ids']]
            return dict(status='queue_stopped_on_technical_or_resource_failure',finished=finished,
                unstarted_run_ids=unstarted,
                unresolved_current_batch_run_ids=list(batch['run_ids']) if unknown else [],automatic_retry=False)
    return dict(status=COMPLETE_STATUS,finished=finished,unstarted_run_ids=[],
        unresolved_current_batch_run_ids=[],automatic_retry=False)


def peer_stop_signal(paths):
    for path in paths:
        if not path.exists():continue
        try:
            if path.is_symlink() or not path.is_file():raise ValueError('plain peer terminal file required')
            pin=file_sha256(path);value=read_json(path)
            if file_sha256(path)!=pin:raise ValueError('peer terminal changed while reading')
            if (path.name=='result.json' and value.get('schema')==RESULT_SCHEMA
                    and value.get('status')==COMPLETE_STATUS):continue
            return dict(path=str(path),sha256=pin,status=value.get('status'),
                reason='peer_error_or_noncomplete_result')
        except Exception as exc:
            return dict(path=str(path),reason='invalid_peer_terminal',type=type(exc).__name__,message=str(exc))
    return None


def _path(value):
    if not isinstance(value,str) or not value:raise ValueError('explicit path string required')
    path=Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def _audit_path(value,output_root):
    path=_path(value)
    if (not path.is_relative_to(ROOT/'audit_results') or path==ROOT/'audit_results'
            or 'episodes' in path.relative_to(ROOT).parts or path.is_relative_to(output_root)
            or output_root.is_relative_to(path)):
        raise ValueError('audit directory must be outside all episode trees under audit_results')
    return path


def _terminal(path,value):
    """Publish a complete, exclusive terminal file for peer boundary readers."""
    temporary=path.with_name('.'+path.name+'.new')
    with temporary.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)  # atomic appearance; refuses an existing terminal
    temporary.unlink()
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def run_queue(queue_path):
    queue_path=Path(queue_path).resolve();queue_pin=file_sha256(queue_path);declaration=read_json(queue_path)
    required={'schema','protocol_path','protocol_sha256','audit_dir','batches'}
    if (not isinstance(declaration,dict) or not required<=set(declaration)
            or set(declaration)-required-{'peer_stop_paths'} or declaration['schema']!=SCHEMA):
        raise ValueError('explicit finite queue schema and fields required')
    pin=declaration['protocol_sha256'];protocol_path=_path(declaration['protocol_path'])
    if not isinstance(pin,str) or re.fullmatch('[0-9a-f]{64}',pin) is None or file_sha256(protocol_path)!=pin:
        raise ValueError('protocol differs from explicit queue SHA256 pin')
    protocol=validate_protocol(read_json(protocol_path))
    if protocol['status']!='frozen':raise ValueError('queue requires frozen protocol')
    ledger=PhaseLedger(protocol,pin);ledger.check_existing()
    saved=read_json(ledger.path) if ledger.path.exists() else None
    if saved is not None and (saved.get('schema')!='semantic.phase_start_ledger.v1'
            or saved.get('phase_id')!=protocol['phase_id']
            or any(saved.get(key)!=value for key,value in ledger.binding.items())):
        raise ValueError('existing phase ledger does not match protocol')
    output_root=_repo_path(protocol['output_relative_path'])
    # This whole-queue check occurs before any new batch reservation or World.
    ordered=validate_schedule(protocol,declaration['batches'],
        attempted_run_ids=[] if saved is None else [row['run_id'] for row in saved['entries']],
        existing_run_ids=[key for key in protocol['slots'] if (output_root/key).exists()])
    audit=_audit_path(declaration['audit_dir'],output_root)
    batches=[dict(run_ids=list(row['run_ids']),audit_dir=str(_audit_path(row['audit_dir'],output_root)))
        for row in declaration['batches']]
    validate_schedule(protocol,batches)  # reject path aliases of the same audit directory
    batch_paths=[Path(row['audit_dir']) for row in batches]
    if (audit.exists() or any(path.exists() or audit.is_relative_to(path) for path in batch_paths)
            or any(left.is_relative_to(right) or right.is_relative_to(left)
                   for i,left in enumerate(batch_paths) for right in batch_paths[i+1:])):
        raise ValueError('queue/batch audit directories must be new, unique and nonoverlapping')
    peer_values=declaration.get('peer_stop_paths',[])
    if not isinstance(peer_values,list) or len(peer_values)>8:raise ValueError('at most eight explicit peer terminal paths')
    peers=[_path(value) for value in peer_values]
    if (len(set(peers))!=len(peers) or any(path.name not in ('result.json','error.json')
            or not path.is_relative_to(ROOT/'audit_results') or 'episodes' in path.relative_to(ROOT).parts
            or path.is_relative_to(audit) or any(path.is_relative_to(p) for p in batch_paths) for path in peers)):
        raise ValueError('distinct external peer queue result/error paths required')
    own_source=Path(__file__).read_bytes();own_sha=file_sha256(__file__)
    batch_path=Path(batch_runner.__file__);batch_source=batch_path.read_bytes();batch_sha=file_sha256(batch_path)
    protocol_bytes=protocol_path.read_bytes();queue_bytes=queue_path.read_bytes()
    def check_binding():
        if (file_sha256(queue_path)!=queue_pin or file_sha256(protocol_path)!=pin
                or file_sha256(__file__)!=own_sha or file_sha256(batch_path)!=batch_sha):
            raise ValueError('queue declaration, protocol or orchestration sources changed after binding')
    check_binding()
    audit.mkdir(parents=True,exist_ok=False)
    try:
        for name,data in (('queue.json',queue_bytes),('protocol.json',protocol_bytes),
                ('queue.source.py',own_source),('batch.source.py',batch_source)):
            with (audit/name).open('xb') as stream:stream.write(data)
        if any(file_sha256(audit/name)!=expected for name,expected in
                (('queue.json',queue_pin),('protocol.json',pin),('queue.source.py',own_sha),('batch.source.py',batch_sha))):
            raise ValueError('queue archive differs from input/source binding')
        header=dict(schema='semantic.finite_queue.started.v1',phase_id=protocol['phase_id'],
            queue_path=str(queue_path),queue_sha256=queue_pin,protocol_path=str(protocol_path),protocol_sha256=pin,
            queue_script_sha256=own_sha,batch_script_sha256=batch_sha,run_ids=ordered,batches=deepcopy(batches),
            peer_stop_paths=[str(path) for path in peers],started_unix_s=time.time(),
            no_implicit_extra_slots=True,no_automatic_retry=True,stop_depends_on_quality=False,
            maximum_slots_per_batch=3,maximum_simultaneous_batches_in_this_queue=1)
        _terminal(audit/'started.json',header)
        with (audit/'events.jsonl').open('x') as stream:
            def emit(value):
                text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
                stream.write(text+'\n');stream.flush();os.fsync(stream.fileno());print(text,flush=True)
            def invoke(row):
                result=batch_runner.run_batch(protocol_path,row['run_ids'],Path(row['audit_dir']))
                if result.get('protocol_sha256')!=pin or result.get('run_ids')!=row['run_ids']:
                    raise ValueError('completed batch does not bind declared protocol and slot order')
                return result
            result=execute_queue(batches,invoke,check_binding,lambda:peer_stop_signal(peers),emit)
        result.update(schema=RESULT_SCHEMA,phase_id=protocol['phase_id'],queue_sha256=queue_pin,
            protocol_sha256=pin,queue_script_sha256=own_sha,batch_script_sha256=batch_sha,
            run_ids=ordered,finished_unix_s=time.time(),stop_depends_on_quality=False,
            no_implicit_extra_slots=True)
        _terminal(audit/'result.json',result)
        return result
    except Exception as exc:
        error=dict(schema='semantic.finite_queue.error.v1',status='queue_entry_or_audit_failed',
            type=type(exc).__name__,message=str(exc),queue_sha256=queue_pin,protocol_sha256=pin,
            automatic_retry=False,finished_unix_s=time.time())
        _terminal(audit/'error.json',error)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True,type=Path)
    args=parser.parse_args()
    try:result=run_queue(args.queue)
    except Exception as exc:
        error=dict(schema='semantic.finite_queue.error.v1',status='queue_entry_failed',
            type=type(exc).__name__,message=str(exc),automatic_retry=False)
        # Publish even an availability/pin preflight failure when the explicit
        # declaration still supplies a safe, unused audit location. Malformed
        # paths and retained outputs never justify overwriting another audit.
        try:
            declaration=read_json(args.queue)
            protocol=read_json(_path(declaration['protocol_path']))
            audit=_audit_path(declaration['audit_dir'],_repo_path(protocol['output_relative_path']))
            if not audit.exists():
                audit.mkdir(parents=True,exist_ok=False)
                with (audit/'queue.json').open('xb') as stream:stream.write(args.queue.read_bytes())
                with (audit/'queue.source.py').open('xb') as stream:stream.write(Path(__file__).read_bytes())
                _terminal(audit/'error.json',error)
        except Exception as publication_error:
            error['failure_marker_publication_error']=str(publication_error)
        print(json.dumps(error,sort_keys=True),flush=True)
        return 2
    print(json.dumps(dict(status=result['status'],unstarted_run_ids=result['unstarted_run_ids']),sort_keys=True),flush=True)
    return 0 if result['status']==COMPLETE_STATUS else 2


if __name__=='__main__':sys.exit(main())
