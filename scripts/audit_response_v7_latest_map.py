#!/usr/bin/env python3
"""Audit latest-observation feasibility on every sealed V7 T branch.

Reuses only the frozen mapper in an isolated source snapshot. No world object,
GT geometry, target metrics, new route execution, or future sensors are made.
After first denied action, later rows describe the original recorded trajectory,
not what a guarded counterfactual trajectory would have observed.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter, deque
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def independent_safe(belief, resolution, robot_radius):
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    assert np.isin(belief, (-1, 0, 1)).all()
    # Unknown and occupied cell squares, including outside-array cells, blocked.
    distance = distance_transform_edt(np.pad(belief == 0, 1, constant_values=False))[1:-1, 1:-1]
    return distance > robot_radius / resolution + np.sqrt(2) / 2


def advance(position, heading, action):
    if action == 'left': return tuple(position), (heading - 1) % 4
    if action == 'right': return tuple(position), (heading + 1) % 4
    assert action == 'forward'
    dr, dc = DIRECTIONS[heading]
    return (position[0] + dr, position[1] + dc), heading


def inside(safe, cell): return 0 <= cell[0] < safe.shape[0] and 0 <= cell[1] < safe.shape[1]


def independent_assess(safe, position, heading, action, remaining):
    target, _ = advance(position, heading, action)
    if remaining < 1: return False, 'no_action_budget', target
    if not inside(safe, position) or not safe[position]: return False, 'current_footprint_not_known_safe', target
    if not inside(safe, target) or not safe[target]: return False, 'next_footprint_not_known_safe', target
    return True, 'latest_observed_grid_allows_action', target


def independent_return(safe, position, heading, anchor, remaining):
    if not inside(safe, position) or not safe[position]: return False, 'current_footprint_not_known_safe', 0
    if not inside(safe, anchor[:2]) or not safe[anchor[:2]]: return False, 'anchor_footprint_not_known_safe', 0
    start = (*position, heading); queue = deque([(start, 0)]); visited = {start}
    while queue:
        state, cost = queue.popleft()
        if state == anchor:
            return (True, 'known_safe_return_within_budget', cost) if cost <= remaining else (False, 'known_return_exceeds_remaining_budget', cost)
        for action in ('left', 'right', 'forward'):
            next_position, next_heading = advance(state[:2], state[2], action)
            next_state = (*next_position, next_heading)
            if inside(safe, next_position) and safe[next_position] and next_state not in visited:
                visited.add(next_state); queue.append((next_state, cost + 1))
    return False, 'anchor_disconnected_in_latest_map', 0


def worker(args):
    import numpy as np
    sys.path.insert(0, str(args.snapshot))
    from env.virtual3d_response_v7 import ResponseConfigV7
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from nso.execution_guard_v8 import ObservedExecutionGuard
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    hashes, reports, action_count = {}, [], 0

    def use(path, manifest, run):
        path = path.resolve(); key = str(path.relative_to(run)); actual = sha(path)
        assert manifest[key] == actual, f'changed raw input: {path}'
        hashes[str(path)] = actual
        return path

    def collision_update(mapper, row):
        if row['collision']:
            target, _ = advance(row['position'], row['heading'], 'forward')
            if inside(mapper.belief, target): mapper.belief[target] = 1

    started = time.monotonic()
    for cid in (f'T{i}' for i in range(8)):
        run = args.runs / cid; manifest = read(run / 'artifact_hashes.json')
        hashes[str(run / 'artifact_hashes.json')] = sha(run / 'artifact_hashes.json')
        meta = read(use(run / 'metadata.json', manifest, run)); replay = read(run / 'verification.json')
        hashes[str(run / 'verification.json')] = sha(run / 'verification.json')
        assert meta['status'] == 'complete' and meta['context']['context_id'] == cid and meta['context']['role'] == 'train'
        assert replay['passed_full'] and replay['status'] == 'passed_full' and not replay['partial']
        assert replay['branches_checked'] == replay['branches_total'] == 12
        assert replay['run'] == str(run) and replay['artifact_manifest_sha256'] == sha(run / 'artifact_hashes.json')
        for family in ('storage_shelves', 'ventilation_baffles'):
            folder = run / family; fixture = read(use(folder / 'fixture.json', manifest, run))
            config = ResponseConfigV7(**fixture['config'])
            with np.load(use(folder / 'prefix_map.npz', manifest, run)) as data: prefix_belief = data['belief'].copy()
            shape = prefix_belief.shape
            prefix = folder / 'prefix'; records = read(use(prefix / 'records.json', manifest, run))
            assert len(records) == 21
            frames, scans = [], []
            for j in range(21):
                frames.append(RGBDFrame.load(use(prefix / 'frames' / f'{j:04d}.npz', manifest, run)))
                scans.append(PlanarScan.load(use(prefix / 'scans' / f'{j:04d}.npz', manifest, run)))
            routes = read(use(folder / 'candidates.json', manifest, run))
            assert [r['candidate_id'] for r in routes] == list(range(6))
            guard = ObservedExecutionGuard(config.resolution_m, config.robot_radius_m)
            for route in routes:
                branch = folder / f"candidate_{route['candidate_id']:03d}"
                actions = read(use(branch / 'actions.json', manifest, run))
                mapper = SemanticHistoryMapperV3(shape, config, config.truncation_m)
                for frame, scan, row in zip(frames, scans, records):
                    mapper.update(frame, scan); collision_update(mapper, row)
                np.testing.assert_array_equal(mapper.belief, prefix_belief)
                safe0 = independent_safe(mapper.belief, config.resolution_m, config.robot_radius_m)
                np.testing.assert_array_equal(safe0, guard.safe_grid(mapper.belief))
                assert all(safe0[tuple(state[:2])] for state in route['states'])
                position, heading = tuple(records[-1]['position']), records[-1]['heading']
                anchor = (*position, heading)
                assert tuple(route['states'][0]) == anchor and tuple(route['states'][-1]) == anchor
                assert route['cost'] == len(route['actions']) <= 48
                trace, first_denial, first_suffix_change = [], None, None
                for index, row in enumerate(actions, start=1):
                    action_count += 1
                    assert row['action_index'] == index and row['action'] == route['actions'][index - 1]
                    assert (*position, heading) == tuple(route['states'][index - 1])
                    safe = independent_safe(mapper.belief, config.resolution_m, config.robot_radius_m)
                    np.testing.assert_array_equal(safe, guard.safe_grid(mapper.belief))
                    remaining = 48 - (index - 1)
                    allowed, reason, target = independent_assess(safe, position, heading, row['action'], remaining)
                    production = guard.assess(mapper.belief, position, heading, row['action'], remaining)
                    assert (production.allowed, production.reason, production.target) == (allowed, reason, target)
                    suffix_unsafe = sum(not safe[tuple(state[:2])] for state in route['states'][index:])
                    if suffix_unsafe and first_suffix_change is None: first_suffix_change = index
                    item = {'action_index': index, 'action': row['action'], 'stage': row['stage'],
                            'before': [*position, heading], 'target': list(target),
                            'current_known_safe': bool(safe[position]),
                            'target_known_safe': bool(inside(safe, target) and safe[target]),
                            'allowed': bool(allowed), 'reason': reason, 'remaining_actions': remaining,
                            'remaining_frozen_states_unsafe': int(suffix_unsafe),
                            'collision_recorded': bool(row['collision']),
                            'after_first_guard_divergence': first_denial is not None}
                    if not allowed and first_denial is None:
                        expected_return = independent_return(safe, position, heading, anchor, remaining)
                        proposed = guard.return_plan(mapper.belief, position, heading, anchor, remaining)
                        assert (proposed.available, proposed.reason, proposed.paid_cost) == expected_return
                        if proposed.available:
                            at_position, at_heading = position, heading
                            for action in proposed.actions:
                                okay, _, _ = independent_assess(safe, at_position, at_heading, action, remaining)
                                assert okay
                                at_position, at_heading = advance(at_position, at_heading, action)
                            assert (*at_position, at_heading) == anchor and len(proposed.actions) <= remaining
                        first_denial = {**item, 'return_plan_without_execution': asdict(proposed)}
                    trace.append(item)
                    next_position, next_heading = advance(position, heading, row['action'])
                    if row['collision']:
                        assert row['action'] == 'forward' and index == len(actions)
                        next_position = position
                    assert tuple(row['position']) == next_position and row['heading'] == next_heading
                    frame = RGBDFrame.load(use(branch / 'frames' / f'{index:04d}.npz', manifest, run))
                    scan = PlanarScan.load(use(branch / 'scans' / f'{index:04d}.npz', manifest, run))
                    mapper.update(frame, scan); collision_update(mapper, row)
                    position, heading = next_position, next_heading
                successful = len(actions) == route['cost'] and not any(r['collision'] for r in actions) and (*position, heading) == anchor
                assert successful or actions[-1]['collision'], 'unaccounted incomplete branch'
                reports.append({'context_id': cid, 'family': family, 'candidate_id': route['candidate_id'],
                                'role': route['group'], 'original_successful_round_trip': successful,
                                'planned_actions': route['cost'], 'recorded_actions': len(actions),
                                'recorded_collision': any(r['collision'] for r in actions),
                                'guard_would_change_original_route': first_denial is not None,
                                'first_denial': first_denial,
                                'first_remaining_frozen_route_unsafe_action': first_suffix_change,
                                'denied_actions_on_original_trace': sum(not r['allowed'] for r in trace),
                                'trace': trace})
                del mapper
            print('audited latest-map', cid, family, '6 branches', flush=True)
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('env', 'nso', 'utils') and getattr(module, '__file__', None):
            assert Path(module.__file__).resolve().is_relative_to(args.snapshot), f'live import: {name}'
    for path, digest in hashes.items(): assert sha(path) == digest, f'changed raw input during audit: {path}'
    changed = [r for r in reports if r['guard_would_change_original_route']]
    successful = [r for r in reports if r['original_successful_round_trip']]
    changed_successful = [r for r in changed if r['original_successful_round_trip']]
    with gzip.open(args.output / 'action_trace.json.gz', 'wt', encoding='utf8') as stream:
        json.dump(reports, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    small = [{k: v for k, v in row.items() if k != 'trace'} for row in reports]
    summary = {'status': 'passed_all_96_recorded_routes_latest_map_review', 'scope': 'posthoc map-only feasibility, not new navigation',
               'run_root': str(args.runs), 'contexts': 8, 'histories': 16, 'branches': len(reports),
               'original_successful_round_trips': len(successful),
               'original_collision_branches': sum(r['recorded_collision'] for r in reports),
               'guard_changed_routes': len(changed), 'guard_changed_original_successes': len(changed_successful),
               'guard_changed_original_collisions': sum(r['recorded_collision'] for r in changed),
               'first_denial_reasons': dict(Counter(r['first_denial']['reason'] for r in changed)),
               'first_denial_return_reasons': dict(Counter(r['first_denial']['return_plan_without_execution']['reason'] for r in changed)),
               'recorded_actions_reviewed': action_count, 'prefix_frames_reused_per_branch': 21,
               'independent_footprint_and_assessment_equal_to_v8': True,
               'independent_BFS_return_cost_equal_to_v8_at_first_denial': True,
               'world_or_evaluator_instantiated': False, 'GT_read': False, 'outcome_tables_read': False,
               'frozen_mapper_reused_not_independently_implemented': True,
               'guard_return_plans_executed': False, 'new_physical_branches': 0,
               'post_divergence_observations_are_original_unmodified_trace_only': True,
               'runtime_seconds': time.monotonic() - started, 'rows': small}
    assert len(reports) == 96
    write(args.output / 'summary.json', summary); write(args.output / 'raw_input_hashes.json', hashes)


def run(args):
    args.runs = args.runs.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}; own_hash = sha(__file__); guard = ROOT / 'nso/execution_guard_v8.py'
    guard_hash = sha(guard)
    try:
        with tempfile.TemporaryDirectory(prefix='nso_v7_latest_map_') as directory:
            snapshot = Path(directory) / 'frozen'; snapshot.mkdir()
            script = Path(directory) / 'audit_response_v7_latest_map.py'; shutil.copy2(__file__, script)
            shared_sources = None
            for cid in (f'T{i}' for i in range(8)):
                folder = args.runs / cid; meta = read(folder / 'metadata.json')
                archive_path = folder / 'sources.zip'
                assert meta['status'] == 'complete'
                if shared_sources is None: shared_sources = meta['source_sha256']
                assert meta['source_sha256'] == shared_sources
                inputs[str(archive_path)] = sha(archive_path)
                with zipfile.ZipFile(archive_path) as archive:
                    for name, digest in shared_sources.items():
                        data = archive.read(name)
                        assert hashlib.sha256(data).hexdigest() == digest
                        if cid == 'T0':
                            target = (snapshot / name).resolve(); assert target.is_relative_to(snapshot)
                            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            target_guard = snapshot / 'nso/execution_guard_v8.py'
            assert not target_guard.exists(), 'the diagnostic guard is not part of the original V7 acquisition'
            shutil.copy2(guard, target_guard)
            subprocess.run([sys.executable, str(script), '--worker', '--snapshot', str(snapshot),
                            '--runs', str(args.runs), '--output', str(args.output)], cwd=snapshot,
                           env={**os.environ, 'PYTHONPATH': str(snapshot)}, check=True)
            for path, digest in inputs.items(): assert sha(path) == digest
            assert sha(guard) == guard_hash and sha(__file__) == own_hash
        summary = read(args.output / 'summary.json')
        summary.update(script_sha256=own_hash, guard_source_sha256=guard_hash,
                       frozen_source_sha256=shared_sources, source_archive_hashes_rechecked=True)
        write(args.output / 'summary.json', summary)
        write(args.output / 'source_input_hashes.json', {**inputs, str(guard): guard_hash})
        changed = [r for r in summary['rows'] if r['guard_would_change_original_route']]
        lines = ['# V7 全部 96 条原路线最新地图可行性审计', '',
                 f"共检查 {summary['recorded_actions_reviewed']} 个已记录分支动作。原 {summary['original_successful_round_trips']} 条成功往返、{summary['original_collision_branches']} 条碰撞分支全部覆盖。",
                 f"最新地图守卫会改变 {summary['guard_changed_routes']} 条原路线，其中 {summary['guard_changed_original_successes']} 条原本成功、{summary['guard_changed_original_collisions']} 条原本碰撞。", '',
                 '用冻结 sources.zip 中 mapper 重融每个原 prefix 和动作后的 RGB-D/雷达；独立实现同口径 footprint 与动作前检查，并逐动作核对新 V8 guard。首次拒绝处额外独立 BFS 核对朝向恢复和剩余 48 动作预算内的返程最短成本，未执行任何返程。没有构造世界、读取 GT/outcome 表或生成新传感器。', '',
                 '| context/family | 候选 | 原成功 | 首次拒绝动作 | 原因 | 最新地图返程 |',
                 '| --- | ---: | --- | ---: | --- | --- |']
        for r in changed:
            first = r['first_denial']; ret = first['return_plan_without_execution']
            lines.append(f"| {r['context_id']}/{r['family']} | {r['candidate_id']} | {r['original_successful_round_trip']} | {first['action_index']} | {first['reason']} | {ret['reason']} ({ret['paid_cost']}) |")
        lines += ['', '首次拒绝前的观测是新守卫与旧执行器共享的实际观测；首次拒绝后的继续记录只描述旧轨迹，不能代替守卫介入后的反事实传感器、重建收益或新碰撞率。通过静态检查的 return plan 也不保证在未来更新后的地图仍可执行。', '',
                  '因此不应只修补 4 条失败并保留其他收益：守卫是否影响成功分支由上表完整计数决定。需要使用独立新版本重放/执行整套固定候选，保留旧 V7 冻结数据与拟合结果；本审计没有改动它们。复用 mapper 不等于独立 SLAM，地图不确定性与保守栅格边界仍是适用范围限制。']
        (args.output / 'LATEST_MAP_REVIEW.md').write_text('\n'.join(lines) + '\n')
        print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'frozen_source_sha256')}, ensure_ascii=False), flush=True)
    except Exception:
        write(args.output / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
    finally:
        write(args.output / 'artifact_hashes.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
