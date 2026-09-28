#!/usr/bin/env python3
"""Frozen saved-history mechanism preflight. No physical or quality experiment."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import signal
import sys
from dataclasses import asdict, replace
from time import perf_counter
from types import SimpleNamespace
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.decision_replay_v13 import load_packet
from nso.observed_state_v26 import geometry_state_v26, VisibleSemanticMemoryV26
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.observed_debt_v28_r1 import build_context, rank_context, VERSION
from nso.execution_guard_v8 import ObservedExecutionGuard
from scripts.preflight_observed_decisions_v26 import (
    CONFIG_FIELDS, jsonable, digest, source_snapshot, forbid_world_calls, COUNTERS)
from scripts.run_observed_autonomous_v26 import verify_inventory

SOURCE = ROOT/'audit_results/observed_autonomous_v27_20260916'
OUT = ROOT/'audit_results/v28_descriptor_preflight_r1_20260917'
PROTOCOL = ROOT/'docs/research/V28_DESCRIPTOR_PREFLIGHT_R1_PROTOCOL_20260917.md'
CHECKPOINTS = tuple(range(0, 401, 50))
CAP, RESERVE = 2*1024**2, 64*1024**2
COUNT = dict(saved_packets=0, mapper_updates=0, geometry_plans=0, rank_calls=0,
    new_worlds=0, new_actions=0, new_sensor_packets=0, mesh_extractions=0, Q_evaluations=0)


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as stream:
        return json.load(stream)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(name, value):
    blob = (json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    require(used+len(blob)+65536 < CAP, '2 MiB cap')
    require(shutil.disk_usage(OUT).free-len(blob) > RESERVE, '64 MiB reserve')
    (OUT/name).write_bytes(blob)


def paths_identity(rows):
    keys = ('pose', 'states', 'actions', 'outbound_cost', 'return_cost', 'cost')
    return [{k: row[k] for k in keys} for row in sorted(rows, key=lambda r: tuple(r['pose']))]


def compact(answer):
    def row(r):
        if r is None:
            return None
        return {k:r[k] for k in ('pose','group','outbound_cost','return_cost','cost','score',
            'geometry_gain','semantic_gain','semantic_evidence') if k in r}
    return dict(candidates=[row(r) for r in answer['candidates']], selected=row(answer['selected']),
        audit=answer['v28_audit'], route_identity_sha256=digest(paths_identity(answer['candidates'])))


def classes(row):
    return {e['class_id'] for e in row['semantic_evidence'] if e['hypothesis_gain'] > 0}


def case_run(index, inventory, manifest):
    folder = SOURCE/f'case_{index:02d}'
    # Explicit whitelist; result file is opened but Q/assignment/routes are never forwarded.
    record = read(folder/'result.json')
    config = SimpleNamespace(**{k:record['public_config'][k] for k in CONFIG_FIELDS})
    shape = tuple(record['shape'])
    del record
    budget = manifest['config']['budget']
    require(budget == 400, 'public budget changed')
    mapper = ObservedRuntimeMapperV10(shape, config)
    mapper.mesh = lambda *a, **kw: (_ for _ in ()).throw(ValueError('mesh forbidden'))
    memory = VisibleSemanticMemoryV26()
    snapshots = []
    for action in range(401):
        name = f'case_{index:02d}/packets/{action:04d}.npz'
        require(sha(SOURCE/name) == inventory[name], 'packet bytes changed '+name)
        packet = load_packet(SOURCE/name)
        COUNT['saved_packets'] += 1
        require(packet.action_id == action, 'packet action order')
        if action == 0:
            anchor = (*packet.position, packet.heading)
        mapper.update(packet.frame, packet.scan)
        COUNT['mapper_updates'] += 1
        cues = memory.update(packet)
        audit = read(folder/'audit'/f'{action:04d}.json.gz')
        require(jsonable([asdict(c) for c in cues]) == audit['cues'], 'cue replay mismatch')
        if action % 25 == 0:
            print(json.dumps(dict(case=index, action=action, stage='saved_history')), flush=True)
        if action not in CHECKPOINTS:
            continue
        state = geometry_state_v26(mapper, packet, anchor, budget-action)
        require(state.geometry_sha256 == audit['geometry_sha256'], 'geometry replay mismatch')
        full = geometry_state_v26(mapper, packet, anchor, budget-action, max_patches=2**31-1)
        context = build_context(state, full, cues)
        COUNT['geometry_plans'] += 1
        answers = {mode:rank_context(context, mode) for mode in ('G','O','S','X')}
        answers['S_no_reservation'] = rank_context(context, 'S', reserve_instances=False)
        swapped = replace(context, cues=tuple(replace(c, class_id=5-c.class_id) for c in cues))
        swapped_objectness = rank_context(swapped, 'O')
        require(swapped_objectness == answers['O'], 'O leaked class')
        low = build_context(state, full, tuple(replace(c, confidence=.59) for c in cues), context.geometry_plan)
        answers['L'] = rank_context(low, 'S')
        COUNT['rank_calls'] += 7
        for mode in ('G','L'):
            fallback = dict(answers[mode]); fallback.pop('v28_audit')
            require(fallback == context.geometry_plan, 'fallback changed G')
        safe = ObservedExecutionGuard(state.resolution_m, state.robot_radius_m).safe_grid(state.belief)
        for answer in answers.values():
            require(len(answer['candidates']) <= 12, 'candidate capacity')
            for r in answer['candidates']:
                require(r['cost'] == len(r['actions']) <= state.remaining_budget, 'unpaid route')
                require(r['states'][-1] == list(anchor), 'no paid return')
                require(all(safe[tuple(p[:2])] for p in r['states']), 'unsafe route')
        eligible = answers['S']['v28_audit']['eligible_cue_ids']
        short = answers['S']['candidates']
        eligible_complex = {c.cue_id for c in cues if c.class_id == 3}.intersection(eligible)
        gate_eligible = {c.class_id for c in cues} == {2,3} and bool(eligible_complex)
        choices = {mode:None if answer['selected'] is None else answer['selected']['pose']
            for mode, answer in answers.items()}
        rows = {mode:[r['pose'] for r in answer['candidates']] for mode, answer in answers.items()}
        snapshots.append(dict(action_id=action, sampled_patches=len(state.patches), full_patches=len(full.patches),
            geometry_sha256=state.geometry_sha256, full_geometry_sha256=full.geometry_sha256,
            cues=[asdict(c) for c in cues], descriptors=context.descriptors,
            class_independent_raw_pool_sha256=digest(paths_identity(context.pool)),
            raw_pool_size=len(context.pool), semantic_route_queries=context.semantic_route_queries,
            raw_pool_observed_support=[dict(pose=r['pose'], group=r['group'],
                outbound_cost=r['outbound_cost'], cost=r['cost'], geometry_score=r['geometry_score'],
                cue_geometry=context.cue_geometry[tuple(r['pose'])]) for r in context.pool],
            counterfactual_checks=dict(objectness_sha256=digest(answers['O']),
                swapped_class_objectness_sha256=digest(swapped_objectness),
                actual_low_confidence=.59, low_confidence_route_queries=low.semantic_route_queries,
                G_equals_original=True, L_equals_original=True),
            answers={k:compact(v) for k,v in answers.items()},
            gate_eligible=gate_eligible, positive_eligible_cue_ids=eligible,
            complex_shortlist_rows=sum(3 in classes(r) for r in short),
            positive_semantic_shortlist_rows=sum(bool(r['semantic_evidence']) for r in short),
            shortlist_rows=len(short),
            all_eligible_cues_retained=set(eligible).issubset(answers['S']['v28_audit']['retained_cue_ids']),
            selection_S_vs_O=choices['S'] != choices['O'], selection_S_vs_X=choices['S'] != choices['X'],
            ranking_S_vs_O=rows['S'] != rows['O'], ranking_S_vs_X=rows['S'] != rows['X'],
            shortlist_S_vs_O={tuple(x) for x in rows['S']} != {tuple(x) for x in rows['O']},
            reservation_changes_shortlist={tuple(x) for x in rows['S']} != {tuple(x) for x in rows['S_no_reservation']}))
        write(f'case_{index:02d}_partial.json', dict(status='in_progress', snapshots=snapshots, counters=COUNT))
    eligible = [x for x in snapshots if x['gate_eligible']]
    denominator = sum(x['shortlist_rows'] for x in eligible)
    semantic_denominator = sum(x['positive_semantic_shortlist_rows'] for x in eligible)
    numerator = sum(x['complex_shortlist_rows'] for x in eligible)
    share = numerator/denominator if denominator else None
    summary = dict(case_index=index, snapshots=len(snapshots), eligible_snapshots=len(eligible),
        complex_shortlist_rows=numerator, all_shortlist_rows=denominator,
        positive_semantic_shortlist_rows=semantic_denominator, complex_share_all_shortlist=share,
        complex_share_semantic_rows=numerator/semantic_denominator if semantic_denominator else None,
        complex_20pct_gate_passed=share is not None and share >= .20,
        all_eligible_cues_retained=all(x['all_eligible_cues_retained'] for x in snapshots),
        ten_consecutive_plan_absence_gate='not_tested_sparse_snapshots',
        **{k:sum(x[k] for x in snapshots) for k in ('selection_S_vs_O','selection_S_vs_X',
           'ranking_S_vs_O','ranking_S_vs_X','shortlist_S_vs_O','reservation_changes_shortlist')})
    write(f'case_{index:02d}.json', dict(summary=summary, snapshots=snapshots))
    (OUT/f'case_{index:02d}_partial.json').unlink()
    return summary


def run():
    require(not OUT.exists(), 'output exists; do not overwrite evidence')
    require(shutil.disk_usage(OUT.parent).free > RESERVE+CAP, 'disk reserve')
    OUT.mkdir()
    start = perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('900 second bound')))
    signal.alarm(900)
    try:
        verify_inventory(SOURCE)
        inventory = read(SOURCE/'artifact_hashes.json')
        manifest = read(SOURCE/'manifest.json')
        for name, expected in manifest['source_sha256'].items():
            require(sha(ROOT/name) == expected, 'old frozen source changed '+name)
        guarded = forbid_world_calls()
        sources = source_snapshot()
        for p in (Path(__file__), PROTOCOL, ROOT/'docs/research/V28_DESCRIPTOR_PREFLIGHT_PROTOCOL_20260917.md',
                  ROOT/'tests/virtual3d/test_observed_debt_v28_r1.py'):
            sources[str(p.relative_to(ROOT))] = sha(p)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(sources):
                archive.writestr(name, (ROOT/name).read_bytes())
        blob = buffer.getvalue()
        require(len(blob) < CAP//2, 'source archive cap')
        (OUT/'sources.zip').write_bytes(blob)
        write('manifest.json', dict(version=VERSION, checkpoints=CHECKPOINTS, cases=[1,3],
            source_sha256=sources, source_archive_sha256=sha(OUT/'sources.zip'),
            original_inventory_sha256=sha(SOURCE/'artifact_hashes.json'),
            source_root=str(SOURCE.relative_to(ROOT)), protocol=str(PROTOCOL.relative_to(ROOT)),
            original_frozen_sources_verified=len(manifest['source_sha256']), guarded_world_classes=guarded,
            public_config_whitelist=CONFIG_FIELDS, evaluation_fields_forwarded=False,
            scope='18 saved-history snapshots; no mesh, Q, physical actions or autonomous outcomes'))
        summaries = [case_run(index, inventory, manifest) for index in (1,3)]
        for name, expected in sources.items():
            require(sha(ROOT/name) == expected, 'run source changed '+name)
        verify_inventory(SOURCE)
        write('result.json', dict(status='complete', summaries=summaries, counters=COUNT,
            world_tripwire_counters=COUNTERS, elapsed_seconds=perf_counter()-start,
            behavioral_checks_passed=True, both_complex_share_gates_passed=all(
                s['complex_20pct_gate_passed'] for s in summaries),
            all_eligible_cues_retained=all(s['all_eligible_cues_retained'] for s in summaries),
            original_inventory_unchanged=True, source_unchanged=True, main_tasks_used=12,
            autonomous_gain_proven=False, semantic_gain_proven=False, Q_calibrated=False,
            main_task_permission=False))
    except BaseException as error:
        write('failure.json', dict(type=type(error).__name__, error=str(error), counters=COUNT,
            elapsed_seconds=perf_counter()-start))
        raise
    finally:
        signal.alarm(0)
        write('artifact_hashes.json', {str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps(read(OUT/'result.json')), flush=True)


if __name__ == '__main__':
    run()
