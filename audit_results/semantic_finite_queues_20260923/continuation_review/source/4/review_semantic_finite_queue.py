#!/usr/bin/env python3
"""Wait for explicit World queues, then review their finite saved outputs only.

Declaration semantic.finite_queue_review.v1 contains protocol_path/sha256,
queues [{path, sha256, live_process?}], numeric_sources (explicit original full
review descriptors), existing_receipts (descriptors with optional reuse path/
pin), max_wait_s and poll_s. A review descriptor contains run_id,
manifest_sha256, review_path, review_sha256. No live process is needed for an
already completed queue; a pending queue requires pid/start_ticks/cmdline_sha256/
boot_id, verified against /proc and its actual --queue argument. No lock is used.

Targets are processed in fixed batch-index/declared-queue/within-batch order as
soon as a complete slot_finished event binds their final manifest. A
single worker waits only for the next listed target; all specified World queues
must remain live or successfully complete at each review boundary. Different
numeric inputs choose full review BEFORE target replay. A proven equal input
chooses replay-only then strict reuse; later proof failure stops without retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.semantic_experiment import COMPLETE
from nso.semantic_scene_experiment import review_experiment, validate_protocol
from scripts import reuse_semantic_scene_endpoint_evaluation as proof
from scripts import run_semantic_finite_queue as world_queue
from scripts import summarize_semantic_scene_results as summary


SCHEMA = 'semantic.finite_queue_review.v1'
SOURCE_FILES = ('scripts/review_semantic_finite_queue.py',
    'scripts/reuse_semantic_scene_endpoint_evaluation.py',
    'scripts/run_semantic_finite_queue.py', 'scripts/run_semantic_declared_batch.py',
    'scripts/summarize_semantic_scene_results.py')
ORDER_RULE = 'batch_index_then_declared_queue_order_then_within_batch_slot_order'


def plain_path(value):
    path = Path(value)
    if not path.is_absolute(): path = ROOT/path
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('plain path without symlink required')
    return path.resolve()


def bound_json(path, pin):
    proof._pin(pin)
    if proof.file_sha256(path) != pin: raise ValueError('explicit input pin mismatch: '+str(path))
    result = proof.read_json(path)
    if proof.file_sha256(path) != pin: raise ValueError('input changed while reading')
    return result


def interleaved_slots(items):
    ordered = []
    for batch_index in range(max(len(item['batches']) for item in items)):
        for item in items:
            if batch_index < len(item['batches']):
                ordered.extend(item['batches'][batch_index]['run_ids'])
    if len(set(ordered)) != len(ordered): raise ValueError('duplicate slot across declared queues')
    return ordered


def completed_slot_event(batch, run_id, protocol_pin):
    """Ignore an unfinished final line; a complete event follows ledger.finish.

    run_experiment writes/final-checks the manifest and ledger before returning
    its pin. Only then run_semantic_declared_batch emits slot_finished and
    flushes/fsyncs the line. The manifest's mere existence is never readiness.
    """
    audit = plain_path(batch['audit_dir']); path = audit/'events.jsonl'
    if not path.exists(): return None
    raw = path.read_bytes(); complete = raw[:raw.rfind(b'\n')+1]
    offset = 0; matches = []
    for line in complete.splitlines(keepends=True):
        row = json.loads(line)
        if row.get('event') == 'slot_finished' and row.get('run_id') == run_id:
            matches.append((row, offset, line))
        offset += len(line)
    if not matches: return None
    if len(matches) != 1: raise ValueError('duplicate slot_finished receipt')
    row, offset, line = matches[0]; result = row.get('result', {})
    if (row.get('runner_invoked') is not True or result.get('status') not in COMPLETE
            or result.get('run_id') != run_id):
        raise ValueError('slot_finished is a technical failure, not a completed task')
    pin = result.get('artifact_manifest_sha256'); proof._pin(pin)
    header = proof.read_json(audit/'batch.json')
    if (header.get('schema') != 'semantic.declared_batch.v1'
            or header.get('protocol_sha256') != protocol_pin or header.get('run_ids') != batch['run_ids']
            or proof.file_sha256(audit/'protocol.json') != protocol_pin
            or header.get('batch_script_sha256') != proof.file_sha256(ROOT/'scripts/run_semantic_declared_batch.py')):
        raise ValueError('completed slot event lacks matching batch/protocol/source binding')
    return dict(manifest_sha256=pin, event_path=str(path), byte_offset=offset,
        complete_line_bytes=len(line), complete_line_sha256=hashlib.sha256(line).hexdigest(),
        batch_header_sha256=proof.file_sha256(audit/'batch.json'),
        publication_rule='complete slot_finished line after run_experiment returned and ledger.finish completed')


def process_identity(pid, queue_path, *, proc_root=Path('/proc')):
    """Read real process identity, including boot/start ticks to reject PID reuse."""
    if type(pid) is not int or pid <= 1: raise ValueError('explicit external positive PID required')
    process = proc_root/str(pid)
    def stat():
        value = (process/'stat').read_text()
        fields = value[value.rfind(')')+2:].split()
        if fields[0] in ('Z','X'): raise ValueError('queue process is not live')
        return int(fields[19])
    start = stat(); cmdline = (process/'cmdline').read_bytes()
    args = [part.decode() for part in cmdline.split(b'\0') if part]
    cwd = (process/'cwd').resolve()
    def argument_path(value):
        path = Path(value); return (path if path.is_absolute() else cwd/path).resolve()
    if (not any(argument_path(arg) == ROOT/'scripts/run_semantic_finite_queue.py' for arg in args)
            or args.count('--queue') != 1 or args.index('--queue')+1 >= len(args)
            or argument_path(args[args.index('--queue')+1]) != queue_path):
        raise ValueError('live PID command line does not execute the explicit World queue')
    if start != stat(): raise ValueError('process identity changed while reading')
    return dict(pid=pid, start_ticks=start, cmdline_sha256=hashlib.sha256(cmdline).hexdigest(),
        boot_id=(proc_root/'sys/kernel/random/boot_id').read_text().strip())


def queue_terminal(item, protocol_pin):
    audit = item['audit']; error = audit/'error.json'; result_path = audit/'result.json'
    if error.exists():
        raise ValueError('explicit World queue has retained error marker: '+str(error))
    if not result_path.exists(): return None
    pin = proof.file_sha256(result_path); result = bound_json(result_path, pin)
    if (result.get('schema') != world_queue.RESULT_SCHEMA
            or result.get('status') != world_queue.COMPLETE_STATUS
            or result.get('queue_sha256') != item['sha256']
            or result.get('protocol_sha256') != protocol_pin
            or result.get('run_ids') != item['run_ids']
            or result.get('unstarted_run_ids') != [] or result.get('unresolved_current_batch_run_ids') != []):
        raise ValueError('World queue failed or terminal binding is inconsistent')
    started = proof.read_json(audit/'started.json')
    if (started.get('queue_sha256') != item['sha256'] or started.get('protocol_sha256') != protocol_pin
            or started.get('run_ids') != item['run_ids']
            or proof.file_sha256(audit/'queue.json') != item['sha256']
            or proof.file_sha256(audit/'protocol.json') != protocol_pin
            or proof.file_sha256(audit/'queue.source.py') != result.get('queue_script_sha256')
            or proof.file_sha256(audit/'batch.source.py') != result.get('batch_script_sha256')):
        raise ValueError('World queue terminal lacks its matching initial/source archive')
    finished = result.get('finished', [])
    if len(finished) != len(item['batches']): raise ValueError('incomplete World batch accounting')
    for row, batch in zip(finished, item['batches']):
        child = row.get('result', {})
        if (row.get('run_ids') != batch['run_ids'] or row.get('batch_invoked') is not True
                or child.get('status') != 'declared_batch_complete'
                or child.get('protocol_sha256') != protocol_pin or child.get('run_ids') != batch['run_ids']
                or [v.get('run_id') for v in child.get('finished', [])] != batch['run_ids']
                or any(v.get('runner_invoked') is not True or v.get('result', {}).get('status') not in COMPLETE
                    for v in child.get('finished', []))):
            raise ValueError('World queue child batch did not complete every explicitly declared slot')
    return dict(path=str(result_path), sha256=pin, run_ids=item['run_ids'])


def wait_for_seal(run_id, *, sealed, check_queues, deadline, poll_s, emit,
        clock=time.monotonic, sleep=time.sleep):
    """Stream only the next explicit target; no scan or quality-based choice."""
    announced = False
    while True:
        check_queues()
        value = sealed(run_id)
        if value is not None: return value
        remaining = deadline-clock()
        if remaining <= 0: raise TimeoutError('bounded wait expired for declared unsealed target: '+run_id)
        if not announced:
            emit(dict(event='waiting_for_declared_target_seal', run_id=run_id)); announced = True
        sleep(min(poll_s, remaining))


def descriptor(value, protocol, *, existing=False):
    required = {'run_id','manifest_sha256','review_path','review_sha256'}
    allowed = required | ({'reuse_path','reuse_sha256'} if existing else set())
    if (not isinstance(value, dict) or not required <= set(value) or set(value)-allowed
            or value['run_id'] not in protocol['slots']
            or ('reuse_path' in value) != ('reuse_sha256' in value)):
        raise ValueError('explicit existing review descriptor and pins required')
    result = dict(value, review_path=plain_path(value['review_path']))
    proof._pin(value['manifest_sha256']); proof._pin(value['review_sha256'])
    if 'reuse_path' in value:
        result['reuse_path'] = plain_path(value['reuse_path']); proof._pin(value['reuse_sha256'])
    return result


def checked_source(value, protocol, episode_root):
    episode = proof._checked_episode(episode_root/value['run_id'], value['manifest_sha256'])
    if episode['protocol'] != protocol: raise ValueError('source protocol differs')
    review = proof._checked_review(value['review_path'], value['review_sha256'], episode, complete_numerical=True)
    if summary.review_metrics(review, episode) is None: raise ValueError('original full numerical source required')
    surface, _, record = proof._load_reference(episode)
    evaluation, metrics = review['evaluation'], review['evaluation']['metrics']
    if (metrics.get('reference_fingerprint') != surface.fingerprint
            or metrics.get('prediction_mesh_validation') != 'strict_positive_area_without_absolute_area_floor'
            or metrics.get('threshold_m') != protocol['evaluation']['threshold_m']
            or metrics.get('prediction_sample_spacing_m') != protocol['evaluation']['sample_spacing_m']
            or metrics.get('prediction_seed') != protocol['evaluation']['seed']
            or evaluation.get('coverage', {}).get('denominator') != record['coverage']
            or evaluation.get('task_success') != proof._task_success(episode)):
        raise ValueError('full numerical source reference/evaluation binding is invalid')
    return episode, review, record


def same_inputs(source, target, record):
    """Normal value differences return False; corrupt inventory/pins still fail."""
    if (source['protocol'] != target['protocol']
            or {k:v for k,v in source['slot'].items() if k != 'method'}
                != {k:v for k,v in target['slot'].items() if k != 'method'}
            or source['reference'] != target['reference']
            or source['manifest']['source_sha256'] != target['manifest']['source_sha256']
            or source['manifest']['input_sha256'] != target['manifest']['input_sha256']): return False
    for name in ('public_workspace.json','public_spec.json','public_graph.json'):
        if proof.file_sha256(source['root']/name) != proof.file_sha256(target['root']/name): return False
    sm, tm = (proof.read_json(row['root']/'prediction/mapper.json') for row in (source,target))
    if any(key not in sm or key not in tm for key in proof.MAPPER_FIELDS):
        raise ValueError('mapper numerical/history field missing')
    if any(sm[key] != tm[key] for key in proof.MAPPER_FIELDS): return False
    if (sm['backend_poisoned'] is not False
            or sm['frames'] != source['result']['acquired_and_saved_packets']
            or tm['frames'] != target['result']['acquired_and_saved_packets']
            or any(sm[key] != record['coverage'][key] for key in ('shape','resolution_m','origin_xy_m','grid_convention'))):
        raise ValueError('source or target mapper is not correctly bound')
    try: arrays = proof._equal_arrays(source['root'], target['root'])
    except ValueError as error:
        if str(error).startswith('numerical input array differs exactly:'): return False
        raise
    if arrays['prediction/occupancy.npz']['arrays']['belief']['array_sha256'] != sm['occupancy_sha256']:
        raise ValueError('mapper occupancy digest differs')
    return True


def choose_source(target, sources, protocol, episode_root):
    for value in sources:
        if value['run_id'] == target['started']['run_id']: continue
        slot = protocol['slots'][value['run_id']]
        if {k:v for k,v in slot.items() if k != 'method'} != {k:v for k,v in target['slot'].items() if k != 'method'}:
            continue
        source, _, record = checked_source(value, protocol, episode_root)
        if same_inputs(source, target, record): return value
    return None


def review_one(target, *, sources, existing, protocol, episode_root, phase_root, emit):
    """Decide equal/full before replay; never repeat an existing target replay."""
    run_id = target['started']['run_id']; pin = target['manifest_sha256']
    review_path = phase_root/'reviews'/f'{run_id}.json'
    reuse_path = phase_root/'evaluation_reuse'/f'{run_id}.json'
    previous = existing.get(run_id); replay_pin = None
    if previous:
        if previous['manifest_sha256'] != pin: raise ValueError('existing target manifest pin differs')
        review_path = previous['review_path']; replay_pin = previous['review_sha256']
        review = proof._checked_review(review_path, replay_pin, target, complete_numerical=False)
        if 'reuse_path' in previous:
            reused = bound_json(previous['reuse_path'], previous['reuse_sha256'])
            numeric = summary.reused_evaluation(reused, target)
            if (Path(numeric['target_review_path']) != review_path
                    or numeric['target_review_sha256'] != replay_pin):
                raise ValueError('declared existing replay and reuse chain disagree')
            return dict(run_id=run_id, mode='existing_reuse_validated', review_path=str(review_path),
                review_sha256=replay_pin, reuse_path=str(previous['reuse_path']),
                reuse_sha256=previous['reuse_sha256'], metrics=numeric['metrics'])
        if review['status'] == 'experiment_reviewed':
            checked_source(previous, protocol, episode_root)
            if not any(row['run_id'] == run_id for row in sources): sources.append(previous)
            return dict(run_id=run_id, mode='existing_full_validated', review_path=str(review_path),
                review_sha256=replay_pin, metrics=review['evaluation']['metrics'])
    elif review_path.exists():
        raise ValueError('retained review requires explicit descriptor; never infer latest or rerun')
    if reuse_path.exists() or reuse_path.with_suffix('.source.py').exists():
        raise ValueError('retained reuse requires explicit complete descriptor; never overwrite')
    source = choose_source(target, sources, protocol, episode_root)
    mode = 'reuse_existing_replay' if previous and source else ('replay_and_reuse' if source else 'full')
    if previous and source is None:
        raise ValueError('existing replay has no numeric result or equal source; stop without repeating replay')
    emit(dict(event='review_mode_selected_before_replay', run_id=run_id, mode=mode,
        source_run_id=None if source is None else source['run_id']))
    if not previous:
        review = review_experiment(target['root'], expected_manifest_sha256=pin, output=review_path,
            replay_only=source is not None)
        expected = 'experiment_replay_verified' if source else 'experiment_reviewed'
        if review.get('status') != expected: raise ValueError('independent offline review failed: '+str(review_path))
        replay_pin = proof.file_sha256(review_path)
    if source:
        receipt = proof.reuse(source_episode=episode_root/source['run_id'],
            source_manifest_sha256=source['manifest_sha256'], source_review=source['review_path'],
            source_review_sha256=source['review_sha256'], target_episode=target['root'],
            target_manifest_sha256=pin, target_review=review_path, target_review_sha256=replay_pin, output=reuse_path)
        metrics = receipt['evaluation']['metrics']
    else:
        value = dict(run_id=run_id, manifest_sha256=pin, review_path=review_path, review_sha256=replay_pin)
        checked_source(value, protocol, episode_root); sources.append(value)
        metrics = review['evaluation']['metrics']
    row = dict(run_id=run_id, mode=mode, review_path=str(review_path), review_sha256=replay_pin,
        metrics={key:metrics[key] for key in ('C_nav','Q','J_nav')})
    if source: row.update(reuse_path=str(reuse_path), reuse_sha256=proof.file_sha256(reuse_path))
    return row


def execute_reviews(run_ids, load, invoke, check, emit):
    finished = []
    for index, run_id in enumerate(run_ids):
        completed = False
        try:
            check(); target = load(run_id)
            emit(dict(event='sealed_target_bound', run_id=run_id, manifest_sha256=target['manifest_sha256'],
                completion_receipt=target.get('queue_completion_receipt')))
            row = invoke(target)
            finished.append(row); completed = True
            emit(dict(event='offline_review_completed', **row)); check()
        except Exception as error:
            emit(dict(event='offline_review_stopped', run_id=run_id, type=type(error).__name__, message=str(error)))
            return dict(status='finite_queue_review_stopped', finished=finished,
                failed_run_id=None if completed else run_id, stopped_at_boundary_after_completed_review=completed,
                stop_reason=dict(type=type(error).__name__,message=str(error)),
                unstarted_run_ids=run_ids[index+1:], automatic_retry=False)
    return dict(status='finite_queue_review_complete', finished=finished, unstarted_run_ids=[], automatic_retry=False)


def run(declaration_path, declaration_sha256, audit_dir):
    declaration_path = plain_path(declaration_path); audit = plain_path(audit_dir)
    declaration = bound_json(declaration_path, declaration_sha256)
    required = {'schema','protocol_path','protocol_sha256','queues','numeric_sources','existing_receipts','max_wait_s','poll_s'}
    if set(declaration) != required or declaration['schema'] != SCHEMA: raise ValueError('explicit finite review declaration required')
    if (type(declaration['max_wait_s']) not in (int,float) or not 0 <= declaration['max_wait_s'] <= 172800
            or type(declaration['poll_s']) not in (int,float) or not 1 <= declaration['poll_s'] <= 30):
        raise ValueError('wait bound 0..172800 seconds and poll 1..30 seconds required')
    protocol_path = plain_path(declaration['protocol_path']); protocol_pin = declaration['protocol_sha256']
    protocol = validate_protocol(bound_json(protocol_path, protocol_pin))
    if protocol['status'] != 'frozen': raise ValueError('frozen protocol required')
    episode_root = plain_path(protocol['output_relative_path'])
    phase_root = plain_path(protocol['ledger_relative_path']).parent
    rows = declaration['queues']; items = []; run_ids = []
    if not isinstance(rows,list) or not 1 <= len(rows) <= 2: raise ValueError('one or two explicit finite queues required')
    for row in rows:
        if set(row) not in ({'path','sha256'},{'path','sha256','live_process'}): raise ValueError('explicit queue fields required')
        path = plain_path(row['path']); queue = bound_json(path, row['sha256'])
        if (queue.get('schema') != world_queue.SCHEMA or queue.get('protocol_sha256') != protocol_pin
                or plain_path(queue['protocol_path']) != protocol_path): raise ValueError('queue/protocol binding differs')
        ordered = world_queue.validate_schedule(protocol, queue['batches'])
        if set(ordered).intersection(run_ids): raise ValueError('duplicate slot across declared World queues')
        run_ids.extend(ordered)
        item = dict(row, path=path, audit=plain_path(queue['audit_dir']), run_ids=ordered, batches=queue['batches'])
        if 'live_process' in item:
            live = item['live_process']
            if not isinstance(live,dict) or set(live) != {'pid','start_ticks','cmdline_sha256','boot_id'}:
                raise ValueError('complete external PID identity required')
        items.append(item)
    run_ids = interleaved_slots(items)
    sources = [descriptor(row, protocol) for row in declaration['numeric_sources']]
    existing_values = [descriptor(row, protocol, existing=True) for row in declaration['existing_receipts']]
    existing = {row['run_id']:row for row in existing_values}
    if (len(existing) != len(existing_values) or not set(existing) <= set(run_ids)
            or len({row['run_id'] for row in sources}) != len(sources)):
        raise ValueError('distinct explicit source/target inventories required')
    protected = [episode_root, phase_root/'reviews', phase_root/'evaluation_reuse',
        plain_path(protocol['asset_root']), plain_path(protocol['navigation_root']),
        plain_path(protocol['reference_index_path']).parent] + [item['audit'] for item in items]
    if (audit.exists() or not audit.is_relative_to(ROOT/'audit_results')
            or any(audit.is_relative_to(p) or p.is_relative_to(audit) for p in protected)):
        raise ValueError('unique new audit outside all immutable inputs/outputs and World queue audits required')
    bindings = {declaration_path:declaration_sha256, protocol_path:protocol_pin}
    bindings.update({item['path']:item['sha256'] for item in items})
    bindings.update({ROOT/name:proof.file_sha256(ROOT/name) for name in SOURCE_FILES})
    execution = protocol.get('execution_source_sha256')
    if not isinstance(execution,dict) or not execution: raise ValueError('declared execution source map required')
    bindings.update({ROOT/name:pin for name,pin in execution.items()})
    before = proof.runtime_counts_v41()
    def check():
        if any(proof.file_sha256(path) != pin for path,pin in bindings.items()):
            raise ValueError('declaration/protocol/orchestration/executed source drift')
        if proof.runtime_counts_v41() != before: raise ValueError('unexpected World/sensor activity in offline process')
    check()
    for value in sources: checked_source(value, protocol, episode_root)
    audit.mkdir(parents=True)
    for index,(path,pin) in enumerate(bindings.items()):
        if path == declaration_path or path == protocol_path or path in [ROOT/name for name in SOURCE_FILES]:
            copy = audit/'source'/str(index)/path.name; copy.parent.mkdir(parents=True)
            copy.write_bytes(path.read_bytes())
            if proof.file_sha256(copy) != pin: raise ValueError('audit source copy differs')
    world_queue._terminal(audit/'started.json', dict(schema='semantic.finite_queue_review.started.v1',
        declaration_path=str(declaration_path), declaration_sha256=declaration_sha256,
        protocol_sha256=protocol_pin, run_ids=run_ids, queues=declaration['queues'],
        review_order_rule=ORDER_RULE, readiness_rule='complete bound slot_finished line, never manifest existence',
        bindings={str(path):pin for path,pin in bindings.items()}, new_worlds=0, automatic_retry=False))
    with (audit/'events.jsonl').open('x') as stream:
        def emit(value):
            line = json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
            stream.write(line+'\n'); stream.flush(); os.fsync(stream.fileno()); print(line,flush=True)
        began = time.monotonic()
        try:
            terminals = {}
            deadline = time.monotonic()+declaration['max_wait_s']
            def check_queues():
                check()
                for index,item in enumerate(items):
                    terminal = queue_terminal(item,protocol_pin)
                    if terminal is not None:
                        if index not in terminals: emit(dict(event='world_queue_completed',**terminal))
                        elif terminal != terminals[index]: raise ValueError('completed World queue terminal changed')
                        terminals[index] = terminal
                        continue
                    try:
                        identity = process_identity(item.get('live_process',{}).get('pid'),item['path'])
                        if identity != item.get('live_process'): raise ValueError('live queue identity differs')
                    except (OSError,ValueError):
                        terminal = queue_terminal(item,protocol_pin)
                        if terminal is None: raise
                        terminals[index] = terminal
            def sealed(run_id):
                root = episode_root/run_id
                owner = next(index for index,item in enumerate(items) if run_id in item['run_ids'])
                batch = next(batch for batch in items[owner]['batches'] if run_id in batch['run_ids'])
                completion = completed_slot_event(batch,run_id,protocol_pin)
                if completion is None:
                    if owner in terminals: raise ValueError('successful World queue is missing its declared sealed output')
                    return None
                pin = completion['manifest_sha256']
                if run_id in existing and pin != existing[run_id]['manifest_sha256']:
                    raise ValueError('explicit existing manifest pin differs from completed slot receipt')
                target = proof._checked_episode(root,pin)
                if (target['protocol'] != protocol or target['result']['status'] not in COMPLETE
                        or target['started']['run_id'] != run_id): raise ValueError('only complete declared sealed episodes may be reviewed')
                target['queue_completion_receipt'] = completion
                return target
            def load(run_id):
                return wait_for_seal(run_id,sealed=sealed,check_queues=check_queues,
                    deadline=deadline,poll_s=declaration['poll_s'],emit=emit)
            result = execute_reviews(run_ids,load,lambda target:review_one(target,sources=sources,existing=existing,
                protocol=protocol,episode_root=episode_root,phase_root=phase_root,emit=emit),check_queues,emit)
            if result['status'] == 'finite_queue_review_complete':
                wait_for_seal('all_explicit_queue_terminals',
                    sealed=lambda _:list(terminals.values()) if len(terminals) == len(items) else None,
                    check_queues=check_queues,deadline=deadline,poll_s=declaration['poll_s'],emit=emit)
            check()
            result.update(schema='semantic.finite_queue_review.result.v1', declaration_sha256=declaration_sha256,
                protocol_sha256=protocol_pin, world_queue_terminals=list(terminals.values()), elapsed_s=time.monotonic()-began,
                runtime_before=before,runtime_after=proof.runtime_counts_v41(),new_worlds=0,new_sensor_packets=0,
                no_quality_dependent_stopping=True,all_review_targets_explicit=True,review_order_rule=ORDER_RULE,
                readiness_rule='complete bound slot_finished line, never manifest existence')
            world_queue._terminal(audit/'result.json',result)
            return result
        except Exception as error:
            world_queue._terminal(audit/'error.json',dict(schema='semantic.finite_queue_review.error.v1',
                status='finite_queue_review_failed',type=type(error).__name__,message=str(error),
                declaration_sha256=declaration_sha256,automatic_retry=False,elapsed_s=time.monotonic()-began))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration',type=Path,required=True)
    parser.add_argument('--declaration-sha256',required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.declaration,args.declaration_sha256,args.audit_dir)
    return 0 if result['status'] == 'finite_queue_review_complete' else 2


if __name__ == '__main__': sys.exit(main())
