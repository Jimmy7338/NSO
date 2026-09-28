#!/usr/bin/env python3
"""Sequential offline reviews for an explicit pinned list of sealed episodes.

Declaration schema semantic.declared_review_batch.v1:
  protocol_path, protocol_sha256, entries: [run_id, manifest_sha256, mode].
Modes are full, replay_and_reuse, reuse_only. Both reuse modes require source:
  {run_id, manifest_sha256, review_path, review_sha256}.
reuse_only also requires target_review: {path, sha256} for an existing replay.
All declaration paths are repository-relative. Every reuse source review must
already exist when this batch starts. No episode discovery or automatic retry.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reuse_semantic_scene_endpoint_evaluation as proof
from nso.semantic_scene_experiment import review_experiment, validate_protocol

SCHEMA = 'semantic.declared_review_batch.v1'
MODES = ('full', 'replay_and_reuse', 'reuse_only')


def repo_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('explicit repository-relative path required')
    path = ROOT/path
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('plain contained repository path required')
    return path.resolve()


def bound_json(path, pin):
    proof._pin(pin)
    if path.is_symlink() or proof.file_sha256(path) != pin:
        raise ValueError('declared input SHA256 mismatch: '+str(path))
    return proof.read_json(path)


def require_new(path):
    if path.exists() or path.is_symlink():
        raise ValueError('existing output must not be rerun or overwritten: '+str(path))


def prepare(declaration):
    """Read-only preflight of the entire list before any review is invoked."""
    if (set(declaration) != {'schema', 'protocol_path', 'protocol_sha256', 'entries'}
            or declaration['schema'] != SCHEMA):
        raise ValueError('explicit declared-review schema and fields required')
    protocol_path = repo_path(declaration['protocol_path'])
    protocol = validate_protocol(bound_json(protocol_path, declaration['protocol_sha256']))
    if protocol['status'] != 'frozen':
        raise ValueError('frozen protocol required')
    entries = declaration['entries']
    if not isinstance(entries, list) or not 1 <= len(entries) <= len(protocol['slots']):
        raise ValueError('nonempty finite explicit review list required')
    ids = [row['run_id'] for row in entries]
    if len(set(ids)) != len(ids) or not set(ids) <= set(protocol['slots']):
        raise ValueError('distinct declared run IDs required')
    episode_root = repo_path(protocol['output_relative_path'])
    phase_root = repo_path(protocol['ledger_relative_path']).parent
    prepared = []

    def sealed(run_id, pin):
        if run_id not in protocol['slots']:
            raise ValueError('source or target run is outside this declared phase')
        episode = proof._checked_episode(episode_root/run_id, pin)
        if episode['started']['run_id'] != run_id or episode['protocol'] != protocol:
            raise ValueError('sealed episode does not bind the declared run and protocol')
        return episode

    for entry in entries:
        mode = entry['mode']
        required = {'run_id', 'manifest_sha256', 'mode'}
        if mode != 'full': required.add('source')
        if mode == 'reuse_only': required.add('target_review')
        if mode not in MODES or set(entry) != required:
            raise ValueError('mode-specific explicit declaration fields required')
        target = sealed(entry['run_id'], entry['manifest_sha256'])
        review_path = repo_path(str((phase_root/'reviews'/f"{entry['run_id']}.json").relative_to(ROOT)))
        reuse_path = repo_path(str((phase_root/'evaluation_reuse'/f"{entry['run_id']}.json").relative_to(ROOT)))
        require_new(reuse_path); require_new(reuse_path.with_suffix('.source.py'))
        item = dict(entry=entry, episode_root=target['root'], review_path=review_path, reuse_path=reuse_path)
        if mode != 'reuse_only':
            require_new(review_path)
        else:
            declared_review = entry['target_review']
            if set(declared_review) != {'path', 'sha256'}:
                raise ValueError('existing target replay needs its path and external pin')
            item['review_path'] = repo_path(declared_review['path'])
            if item['review_path'].is_relative_to(episode_root):
                raise ValueError('existing replay must be outside immutable episodes')
            proof._checked_review(item['review_path'], declared_review['sha256'], target, complete_numerical=False)
        if mode != 'full':
            source = entry['source']
            if set(source) != {'run_id', 'manifest_sha256', 'review_path', 'review_sha256'}:
                raise ValueError('reuse source needs explicit episode and existing original review pins')
            if source['run_id'] == entry['run_id']:
                raise ValueError('reuse source and target must be distinct')
            original = sealed(source['run_id'], source['manifest_sha256'])
            source_review = repo_path(source['review_path'])
            if source_review.is_relative_to(episode_root):
                raise ValueError('original review must be outside immutable episodes')
            proof._checked_review(source_review, source['review_sha256'], original, complete_numerical=True)
            if ({k:v for k,v in original['slot'].items() if k != 'method'}
                    != {k:v for k,v in target['slot'].items() if k != 'method'}):
                raise ValueError('reuse requires the same nonmethod experimental condition')
            item.update(source_root=original['root'], source_review=source_review)
        prepared.append(item)
    return protocol_path, protocol, prepared


def invoke_review(item):
    entry = item['entry']; mode = entry['mode']; path = item['review_path']
    require_new(item['reuse_path']); require_new(item['reuse_path'].with_suffix('.source.py'))
    if mode == 'reuse_only':
        review_pin = entry['target_review']['sha256']
        bound_json(path, review_pin)
    else:
        require_new(path)
        review = review_experiment(item['episode_root'], expected_manifest_sha256=entry['manifest_sha256'],
            output=path, replay_only=mode == 'replay_and_reuse')
        expected = 'experiment_reviewed' if mode == 'full' else 'experiment_replay_verified'
        if review.get('status') != expected:
            raise ValueError('retained offline review failed: '+str(path)+' '+str(review.get('error')))
        review_pin = proof.file_sha256(path)
    if mode != 'full':
        source = entry['source']
        review = proof.reuse(source_episode=item['source_root'], source_manifest_sha256=source['manifest_sha256'],
            source_review=item['source_review'], source_review_sha256=source['review_sha256'],
            target_episode=item['episode_root'], target_manifest_sha256=entry['manifest_sha256'],
            target_review=path, target_review_sha256=review_pin, output=item['reuse_path'])
        if review.get('status') != 'experiment_endpoint_evaluation_reused':
            raise ValueError('retained exact-input reuse failed')
    result = dict(run_id=entry['run_id'], mode=mode, status=review['status'],
        review_path=str(path), review_sha256=review_pin,
        evaluation_execution='recomputed' if mode == 'full' else 'reused',
        metrics={key: review['evaluation']['metrics'][key] for key in ('C_nav', 'Q', 'J_nav')})
    if mode != 'full':
        result.update(evaluation_reuse_path=str(item['reuse_path']),
                      evaluation_reuse_sha256=proof.file_sha256(item['reuse_path']))
    return result


def execute_sequence(items, invoke, check_binding, emit):
    """Small injected control loop for analytic tests; no implicit discovery."""
    finished = []
    for index, item in enumerate(items):
        invoked = False
        try:
            check_binding()
            emit(dict(event='review_started', run_id=item['entry']['run_id'], mode=item['entry']['mode']))
            invoked = True
            row = invoke(item)
            check_binding()
        except Exception as exc:
            row = dict(run_id=item['entry']['run_id'], mode=item['entry']['mode'], status='offline_review_failed',
                type=type(exc).__name__, message=str(exc), artifacts_may_exist=invoked, automatic_retry=False)
            finished.append(row); emit(dict(event='review_finished', **row))
            return dict(status='declared_review_batch_stopped', finished=finished,
                unstarted_run_ids=[v['entry']['run_id'] for v in items[index+int(invoked):]], automatic_retry=False)
        finished.append(row); emit(dict(event='review_finished', **row))
    return dict(status='declared_review_batch_complete', finished=finished, unstarted_run_ids=[], automatic_retry=False)


def run_batch(declaration_path, audit_dir):
    declaration_path = Path(declaration_path).resolve()
    declaration_pin = proof.file_sha256(declaration_path)
    declaration = bound_json(declaration_path, declaration_pin)
    protocol_path, protocol, items = prepare(declaration)
    audit_dir = Path(audit_dir).absolute()
    if not audit_dir.is_relative_to(ROOT/'audit_results'):
        raise ValueError('new batch audit must be within repository audit_results')
    audit_dir = repo_path(str(audit_dir.relative_to(ROOT)))
    protected = [repo_path(protocol['output_relative_path']),
                 repo_path(protocol['asset_root']), repo_path(protocol['navigation_root']),
                 repo_path(protocol['reference_index_path']).parent,
                 repo_path(protocol['ledger_relative_path']).parent/'reviews',
                 repo_path(protocol['ledger_relative_path']).parent/'evaluation_reuse']
    if any(audit_dir.is_relative_to(path) or path.is_relative_to(audit_dir) for path in protected):
        raise ValueError('new audit must be separate from episodes, reviews and immutable inputs')
    require_new(audit_dir)
    sources = {Path(__file__).resolve(): proof.file_sha256(__file__),
               Path(proof.__file__).resolve(): proof.file_sha256(proof.__file__)}
    bindings = dict(sources)
    bindings.update({declaration_path: declaration_pin, protocol_path: declaration['protocol_sha256']})
    before = proof.runtime_counts_v41()
    def check_binding():
        if any(proof.file_sha256(path) != pin for path, pin in bindings.items()):
            raise ValueError('batch declaration, protocol or wrapper source changed')
        if proof.runtime_counts_v41() != before:
            raise ValueError('unexpected World or sensor activity during offline batch')
    check_binding(); audit_dir.mkdir(parents=True, exist_ok=False)
    for source, name in ((declaration_path, 'declaration.json'), (protocol_path, 'protocol.json'),
                         (Path(__file__).resolve(), 'wrapper.source.py'),
                         (Path(proof.__file__).resolve(), 'reuse.source.py')):
        with (audit_dir/name).open('xb') as stream: stream.write(source.read_bytes())
        if proof.file_sha256(audit_dir/name) != bindings[source]:
            raise ValueError('input or source changed during batch archive')
    with (audit_dir/'events.jsonl').open('x') as stream:
        def emit(value):
            line = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write(line+'\n'); stream.flush(); os.fsync(stream.fileno()); print(line, flush=True)
        emit(dict(event='offline_batch_started', phase_id=protocol['phase_id'],
            declaration_sha256=declaration_pin, run_ids=[i['entry']['run_id'] for i in items]))
        result = execute_sequence(items, invoke_review, check_binding, emit)
        after = proof.runtime_counts_v41()
        result.update(schema='semantic.declared_review_batch_result.v1', phase_id=protocol['phase_id'],
            declaration_sha256=declaration_pin, protocol_sha256=declaration['protocol_sha256'],
            wrapper_source_sha256={str(path.relative_to(ROOT)): pin for path,pin in sources.items()},
            finished_unix_s=time.time(), runtime_before=before, runtime_after=after,
            new_worlds=after.get('worlds_created', 0)-before.get('worlds_created', 0),
            new_sensor_packets=after.get('rgbd_frames', 0)-before.get('rgbd_frames', 0),
            new_primary_experiments=0,
            replay_uses_saved_packets_only=True, no_implicit_extra_tasks=True)
        with (audit_dir/'result.json').open('xb') as terminal:
            terminal.write(proof.canonical_bytes(result)); terminal.flush(); os.fsync(terminal.fileno())
        emit(dict(event='offline_batch_finished', status=result['status'],
            processed_entries=len(result['finished']), unstarted_run_ids=result['unstarted_run_ids']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--audit-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_batch(args.declaration, args.audit_dir)
    except Exception as exc:
        print(json.dumps(dict(status='offline_batch_entry_failed', type=type(exc).__name__, message=str(exc)),
                         sort_keys=True), flush=True)
        return 2
    return 0 if result['status'] == 'declared_review_batch_complete' else 2


if __name__ == '__main__':
    sys.exit(main())
