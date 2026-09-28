#!/usr/bin/env python3
"""Plan or explicitly apply reversible raw-file compaction outside frozen code.

This adapter never edits an original scientific manifest or JSON receipt. It
only removes allowlisted raw files after their byte-exact archive has passed
verification and the exact plan SHA has been explicitly approved. Planning and
materialization do not delete anything. No World, policy or mapper is imported.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.archive_article_episode_20260928 import (META, BoundedXZInput, approved_episode,
    canonical, decode, original_manifest_check, regular, relative, sha_file, tree_files,
    validate_packaging_manifest, verify)
import tarfile


def removable(name):
    return bool(re.fullmatch(r'packets/[0-9]{3}_(rgbd|scan)\.npz',name)
        or re.fullmatch(r'steps/[0-9]{3}\.json\.gz',name)
        or name in ('prediction/mesh.npz','prediction/occupancy.npz'))


def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    sync_directories(path.parent)


def sync_directories(directory):
    """Make newly created operation-record ancestors durable before removal."""
    directory=Path(directory).absolute()
    for path in (directory,*directory.parents):
        descriptor=os.open(path,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)


def sha_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def outside(source,target):
    source=Path(source).absolute();target=Path(target).absolute()
    if target.resolve().is_relative_to(source.resolve()):
        raise ValueError('operation records/archive must be outside original episode')
    if target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError('output symlink ancestors rejected')
    return target


def package_manifest(path):
    with BoundedXZInput(regular(path)) as source:
        with tarfile.open(fileobj=source,mode='r|') as tar:
            first=tar.next()
            if first is None or first.name!=META or not first.isfile() or first.size>2*1024**2:
                raise ValueError('bounded package metadata required')
            data=decode(tar.extractfile(first).read(2*1024**2+1));validate_packaging_manifest(data)
            return data


def record_files(root,names):
    return {name:dict(bytes=(root/name).stat().st_size,sha256=sha_file(root/name)) for name in names}


def verify_record_files(root,manifest):
    for name,row in manifest.items():
        relative(name);path=regular(root/name)
        if path.stat().st_size!=row['bytes'] or sha_file(path)!=row['sha256']:
            raise ValueError('external operation record changed: '+name)


def inspect_plan(plan_dir):
    plan_dir=Path(plan_dir).absolute()
    outer=decode(regular(plan_dir/'manifest.json').read_bytes())
    if outer.get('schema')!='article.raw_compaction_plan_manifest.v1':raise ValueError('plan manifest schema required')
    verify_record_files(plan_dir,outer['files'])
    path=plan_dir/'plan.json';plan=decode(regular(path).read_bytes())
    if plan.get('schema')!='article.raw_compaction_plan.v1':raise ValueError('explicit compaction plan required')
    for key in ('run_id','phase_name'):
        if not isinstance(plan[key],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',plan[key]):
            raise ValueError('one bounded component required for '+key)
    if sha_file(path)!=outer['plan_sha256']:raise ValueError('plan SHA differs from operation manifest')
    episode=Path(plan['episode']);outside(episode,plan_dir);outside(episode,plan['archive'])
    if (not episode.is_absolute() or episode.name!=plan['run_id'] or episode.parent.name!='episodes'
            or episode.parent.parent.name!=plan['phase_name']):
        raise ValueError('episode, run ID and phase must refer to the same original location')
    rows=plan['remove']+plan['retain'];names=[r['path'] for r in rows]
    if len(names)!=len(set(names)):raise ValueError('duplicate compaction path')
    for row in rows:
        relative(row['path'])
        if type(row['bytes']) is not int or row['bytes']<0:raise ValueError('bounded byte record required')
    if any(not removable(r['path']) for r in plan['remove']):
        raise ValueError('compaction deletion path is outside the raw-data allowlist')
    if any(removable(r['path']) for r in plan['retain']):
        raise ValueError('plan must declare the complete fixed raw-data partition')
    if 'artifact_manifest.json' not in {r['path'] for r in plan['retain']}:
        raise ValueError('original artifact manifest must be retained unchanged')
    return plan,outer['plan_sha256']


def plan_compaction(episode,review_root,archive,output):
    began=time.monotonic();episode=Path(episode).absolute();archive=outside(episode,archive)
    output=outside(episode,output)
    if output.exists():raise FileExistsError('new plan directory required')
    provenance=approved_episode(episode,review_root)
    archive_pin=sha_file(regular(archive));validated=verify(archive);packed=package_manifest(archive)
    if (validated['artifact_manifest_sha256']!=provenance['artifact_manifest_sha256']
            or packed['independent_review_sha256']!=provenance['review_sha256']
            or sha_file(archive)!=archive_pin):
        raise ValueError('archive, original episode and independent review must bind the same bytes')
    remove=[];retain=[];packed_rows={r['original_path']:r for r in packed['files']}
    for path in tree_files(episode):
        name=str(path.relative_to(episode));row=dict(path=name,bytes=path.stat().st_size,
            allocated_bytes=path.stat().st_blocks*512,sha256=sha_file(path))
        archived=packed_rows.get(name)
        if archived is None or row['bytes']!=archived['original_bytes'] or row['sha256']!=archived['original_sha256']:
            raise ValueError('original file not byte-exactly covered by archive: '+name)
        (remove if removable(name) else retain).append(row)
    if len(remove)+len(retain)!=len(packed_rows):raise ValueError('incomplete package partition')
    if not remove:raise ValueError('no allowlisted raw files to compact')
    ledger=regular(episode.parent.parent/'start_ledger.json').read_bytes()
    plan=dict(schema='article.raw_compaction_plan.v1',status='planned_only_no_raw_files_deleted',
        episode=str(episode),run_id=episode.name,phase_name=episode.parent.parent.name,
        review_root=str(Path(review_root).absolute()),review_sha256=provenance['review_sha256'],
        archive=str(archive),archive_sha256=archive_pin,archive_bytes=archive.stat().st_size,
        original_manifest_sha256=provenance['artifact_manifest_sha256'],
        ledger_entry=provenance['ledger_entry'],phase_ledger_snapshot_sha256=sha_bytes(ledger),
        remove=remove,retain=retain,original_files=len(remove)+len(retain),
        original_bytes=sum(r['bytes'] for r in remove+retain),
        removable_bytes=sum(r['bytes'] for r in remove),
        expected_reclaimed_allocated_bytes=sum(r['allocated_bytes'] for r in remove),
        retained_bytes=sum(r['bytes'] for r in retain),
        full_original_recoverable=True,original_manifest_will_not_be_edited=True,
        already_compacted=False,apply_requires_exact_plan_sha256=True,
        archiver_source_sha256=sha_file(Path(__file__).with_name('archive_article_episode_20260928.py')),
        adapter_source_sha256=sha_file(Path(__file__)),validation_elapsed_s=time.monotonic()-began)
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'plan.json',canonical(plan));write_new(output/'phase_ledger_snapshot.json',ledger)
    manifest=dict(schema='article.raw_compaction_plan_manifest.v1',plan_sha256=sha_file(output/'plan.json'),
        files=record_files(output,['plan.json','phase_ledger_snapshot.json']),raw_files_deleted=0)
    write_new(output/'manifest.json',canonical(manifest))
    return dict(status=plan['status'],plan_root=str(output),plan_sha256=manifest['plan_sha256'],
        original_bytes=plan['original_bytes'],removable_bytes=plan['removable_bytes'],
        retained_bytes=plan['retained_bytes'],archive_bytes=plan['archive_bytes'],
        retained_plus_archive_bytes=plan['retained_bytes']+plan['archive_bytes'],
        expected_reclaimed_allocated_bytes=plan['expected_reclaimed_allocated_bytes'],
        raw_files_deleted=0,elapsed_s=time.monotonic()-began)


def validate_partition(episode,rows):
    expected={r['path']:r for r in rows}
    actual={str(p.relative_to(episode)):p for p in tree_files(episode)}
    if set(actual)!=set(expected):raise ValueError('working-copy inventory differs from approved plan partition')
    for name,path in actual.items():
        row=expected[name]
        if path.stat().st_size!=row['bytes'] or sha_file(path)!=row['sha256']:
            raise ValueError('working-copy artifact changed: '+name)


def apply_compaction(plan_dir,receipt_dir,approved_plan_sha256):
    """Destructive only after exact explicit plan approval; never used implicitly."""
    began=time.monotonic();plan,pin=inspect_plan(plan_dir)
    if approved_plan_sha256!=pin:raise ValueError('explicit approval of the exact plan SHA256 required')
    episode=Path(plan['episode']);receipt_dir=outside(episode,receipt_dir);plan_dir=Path(plan_dir)
    if receipt_dir.exists():raise FileExistsError('new external compaction receipt directory required')
    if sha_file(Path(__file__))!=plan['adapter_source_sha256']:
        raise ValueError('adapter changed since plan review; create a new plan')
    if sha_file(Path(__file__).with_name('archive_article_episode_20260928.py'))!=plan['archiver_source_sha256']:
        raise ValueError('verified archiver implementation changed')
    provenance=approved_episode(episode,plan['review_root'])
    if (provenance['artifact_manifest_sha256']!=plan['original_manifest_sha256']
            or provenance['review_sha256']!=plan['review_sha256'] or provenance['ledger_entry']!=plan['ledger_entry']):
        raise ValueError('closed episode/review changed since plan approval')
    archive=regular(plan['archive'])
    if sha_file(archive)!=plan['archive_sha256']:raise ValueError('archive changed since plan review')
    proof=verify(archive)
    if proof['artifact_manifest_sha256']!=plan['original_manifest_sha256']:
        raise ValueError('archive cannot restore the approved original manifest')
    with archive.open('rb') as source:os.fsync(source.fileno())
    sync_directories(archive.parent)
    validate_partition(episode,plan['remove']+plan['retain'])
    receipt_dir.mkdir(parents=True,exist_ok=False)
    write_new(receipt_dir/'approved_plan.json',regular(plan_dir/'plan.json').read_bytes())
    write_new(receipt_dir/'phase_ledger_snapshot.json',regular(plan_dir/'phase_ledger_snapshot.json').read_bytes())
    intent=dict(schema='article.raw_compaction_intent.v1',plan_sha256=pin,
                original_manifest_sha256=plan['original_manifest_sha256'],archive_sha256=plan['archive_sha256'],
                expected_raw_removals=len(plan['remove']),original_metadata_edits=0)
    write_new(receipt_dir/'prepared.json',canonical(intent))
    removed=[]
    try:
        for row in plan['remove']:
            path=regular(episode/row['path'])
            if path.stat().st_size!=row['bytes'] or sha_file(path)!=row['sha256']:
                raise ValueError('raw artifact changed immediately before removal')
            path.unlink();removed.append(row['path'])
        for directory in {str((episode/name).parent) for name in removed}:
            sync_directories(Path(directory))
        validate_partition(episode,plan['retain'])
        if sha_file(episode/'artifact_manifest.json')!=plan['original_manifest_sha256']:
            raise ValueError('original scientific manifest was changed')
    except BaseException as exc:
        write_new(receipt_dir/'incomplete.json',canonical(dict(status='compaction_interrupted',
            type=type(exc).__name__,message=str(exc),removed_paths=removed,automatic_resume=False,
            archive_preserved=True,recovery='materialize the original archive into a new review workspace')))
        raise
    receipt=dict(schema='article.raw_compaction_receipt.v1',status='compacted_verified_archive_retained',
        run_id=plan['run_id'],episode=str(episode),plan_sha256=pin,archive=str(archive),
        archive_sha256=plan['archive_sha256'],original_manifest_sha256=plan['original_manifest_sha256'],
        raw_files_removed=len(removed),removed_paths=removed,original_metadata_files_edited=0,
        original_bytes=plan['original_bytes'],retained_bytes=plan['retained_bytes'],
        archive_bytes=archive.stat().st_size,elapsed_s=time.monotonic()-began,
        full_original_recoverable=True,review_requires_fresh_materialization=True,
        scientific_results_changed=False,failed_attempts_compacted=False)
    write_new(receipt_dir/'receipt.json',canonical(receipt))
    write_new(receipt_dir/'manifest.json',canonical(dict(schema='article.raw_compaction_operation_manifest.v1',
        files=record_files(receipt_dir,['approved_plan.json','phase_ledger_snapshot.json','prepared.json','receipt.json']))))
    return receipt


def materialize(plan_dir,workspace):
    began=time.monotonic();plan,pin=inspect_plan(plan_dir);plan_dir=Path(plan_dir)
    episode=Path(plan['episode']);workspace=outside(episode,workspace)
    if workspace.exists():raise FileExistsError('materialization requires a wholly new workspace')
    archive=regular(plan['archive'])
    if sha_file(archive)!=plan['archive_sha256']:raise ValueError('archive pin changed')
    ledger=regular(plan_dir/'phase_ledger_snapshot.json').read_bytes()
    if sha_bytes(ledger)!=plan['phase_ledger_snapshot_sha256']:raise ValueError('phase ledger snapshot changed')
    restored=workspace/plan['phase_name']/'episodes'/plan['run_id']
    receipt=verify(archive,restored)
    if receipt['artifact_manifest_sha256']!=plan['original_manifest_sha256']:
        raise ValueError('restored original manifest differs from plan')
    write_new(restored.parent.parent/'start_ledger.json',ledger)
    report=dict(schema='article.materialized_review_workspace.v1',status='full_original_restored_for_readonly_review',
        source_plan_sha256=pin,archive_sha256=plan['archive_sha256'],episode=str(restored),
        original_manifest_sha256=plan['original_manifest_sha256'],verified_original_files=receipt['verified_original_files'],
        strict_original_manifest_passed=True,raw_files_deleted=0,new_worlds=0,new_tsdf_integrations=0,
        elapsed_s=time.monotonic()-began,
        review_command=['.venv/bin/python','-B','scripts/review_article_episode_20260928.py',
                        '--episode',str(restored),'--output',str(workspace/'independent_review')])
    write_new(workspace/'materialization.json',canonical(report));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    planning=sub.add_parser('plan');planning.add_argument('--episode',type=Path,required=True)
    planning.add_argument('--review',type=Path,required=True);planning.add_argument('--archive',type=Path,required=True)
    planning.add_argument('--output',type=Path,required=True)
    applying=sub.add_parser('apply');applying.add_argument('--plan',type=Path,required=True)
    applying.add_argument('--receipt',type=Path,required=True);applying.add_argument('--approve-plan-sha256',required=True)
    restoring=sub.add_parser('materialize');restoring.add_argument('--plan',type=Path,required=True)
    restoring.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    result=(plan_compaction(args.episode,args.review,args.archive,args.output) if args.command=='plan' else
        apply_compaction(args.plan,args.receipt,args.approve_plan_sha256) if args.command=='apply' else
        materialize(args.plan,args.workspace))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
