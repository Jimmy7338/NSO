#!/usr/bin/env python3
"""One fixed offline camera qualification; no World, control, fusion or score.

Private geometry chooses exactly two nearest statically eligible public coarse
nodes per facility BEFORE any rendering. These fixtures are never planner input.
Original color components and the unchanged V41 marker-plane fitter are used.
"""
import argparse
from collections import Counter
from itertools import permutations
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.development_sensor_v41 import (
    camera_transform_xyyaw, render_rgbd_arrays, runtime_counts_v41,
)
from nso.instance_belief_v40 import PaidRGBDObservationV40, _components
from nso.observed_residual_v41 import fit_marker_plane_v41, observed_points_v41
from nso.primitive_navigation_v41 import PrimitiveStateV41
from nso.public_navigation_v43 import segment_intersects_rectangle_v43
from nso.semantic_scene_assets import (
    CONDITIONS, PARENTS, asset_id, file_sha256, load_semantic_scene_asset,
)
from nso.semantic_scene_navigation import load_semantic_navigation_bundle

ASSET_PIN = '1566282bc88ef557daa7498c2816e4968c08ae7088bcc6e3d1413b6a96dadfd1'
NAV_PIN = '3b6d8de9316ed8952c81dfa43130fc52cef9c5104b775388d6b61be02e2a0813'
SOURCE_FILES = (
    'scripts/check_semantic_scene_sensing.py', 'env/development_sensor_v41.py',
    'nso/instance_belief_v40.py', 'nso/observed_residual_v41.py',
    'nso/primitive_navigation_v41.py', 'nso/public_navigation_v43.py',
    'nso/semantic_scene_assets.py', 'nso/semantic_scene_navigation.py',
)


def write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def nearest_heading(delta):
    bearing = math.atan2(delta[1], delta[0])
    return min(range(12), key=lambda h: (round(abs(math.atan2(
        math.sin(bearing-h*math.pi/6), math.cos(bearing-h*math.pi/6))), 12), h))


def select_two(asset, navigation, marker):
    """Rank only declared geometry, never pixels, fitted results or task scores."""
    graph = navigation['graph']
    center = np.asarray(marker['center_world_m'])[:2]
    normal = np.asarray(marker['normal_world'])[:2]
    rectangles = [[b[0], b[1], b[2], b[3]]
                  for b in asset['metadata']['background_boxes'] if b[5] > 0]
    for item in asset['metadata']['private_instances']:
        if item['instance_id'] != marker['instance_id']:
            lower, upper = item['navigation_envelope_aabb_m']
            rectangles.append([lower[0], upper[0], lower[1], upper[1]])
    eligible = []
    for node in sorted(graph.original_nodes):
        position = np.asarray(graph.positions[node])
        delta = position-center
        distance = float(np.linalg.norm(delta))
        projection = float(delta @ normal)
        if not .45 <= distance <= 2.5 or projection < .3 or projection/distance < .4:
            continue
        if any(segment_intersects_rectangle_v43(position, center, r) for r in rectangles):
            continue
        heading = nearest_heading(-delta)
        eligible.append(dict(node=node, heading=heading, xy_m=position.tolist(),
            range_xy_to_marker_m=distance, front_projection_m=projection,
            front_cosine=projection/distance))
    eligible.sort(key=lambda v: (v['range_xy_to_marker_m'], v['node']))
    return eligible[:2], len(eligible)


def route_costs(navigation, views):
    """An offline graph action count, not executed observations or a rollout."""
    graph, home = navigation['graph'], navigation['home_state']
    states = [PrimitiveStateV41(v['node'], v['heading']) for v in views]
    cache = {}
    def cost(left, right):
        key = (left, right)
        if key not in cache:
            route = graph.route(left, right)
            if route is None:
                raise ValueError('selected public fixture unexpectedly unreachable')
            cache[key] = route.cost
        return cache[key]
    individual = [cost(home, s)+1+cost(s, home) for s in states]
    tours = []
    for order in permutations(range(len(states))):
        chain = [home]+[states[i] for i in order]+[home]
        movement = sum(cost(a, b) for a, b in zip(chain, chain[1:]))
        tours.append(dict(view_order=list(order), movement_turn_actions=movement,
            explicit_observe_actions=len(states), total_actions=movement+len(states)))
    best = min(tours, key=lambda t: (t['total_actions'], t['view_order'])) if tours else None
    return dict(individual_home_view_observe_home_actions=individual,
                shortest_all_selected_views_return_tour=best,
                returns_home_position_and_heading=True,
                movement_arrival_already_provides_rgbd=True,
                explicit_observe_is_extra_macro_step_not_first_diagnosis=True,
                route_executed=False)


def selection_plan(asset_root, navigation_root, asset_pin, navigation_pin):
    selections, loaded = [], {}
    for parent in PARENTS:
        for condition in CONDITIONS:
            identity = asset_id(parent, condition)
            asset = load_semantic_scene_asset(asset_root/identity,
                expected_manifest_sha256=asset_pin)
            navigation = load_semantic_navigation_bundle(navigation_root/identity,
                expected_manifest_sha256=navigation_pin)
            if navigation['asset_manifest_sha256'] != asset_pin:
                raise ValueError('public navigation belongs to a different asset bundle')
            sensor = navigation['public_spec']['sensor']
            if (sensor['width'] != 96 or sensor['height'] != 72
                    or sensor['depth_noise_relative_std'] != .01
                    or sensor['depth_min_m'] != .1 or sensor['depth_max_m'] != 4.
                    or navigation['public_spec']['motion']['camera_height_m'] != .9):
                raise ValueError('fixed original camera/noise contract differs')
            loaded[identity] = (asset, navigation)
            instances = []
            for marker in sorted(asset['markers'], key=lambda m: m['instance_id']):
                views, eligible_count = select_two(asset, navigation, marker)
                for view_index, view in enumerate(views):
                    # Paired conditions receive the same noise field for the same
                    # parent/instance/view. This is not a trajectory paid step.
                    view.update(view_index=view_index,
                        noise_step=2*marker['instance_id']+view_index)
                truth = next(i for i in asset['metadata']['private_instances']
                             if i['instance_id'] == marker['instance_id'])
                instances.append(dict(instance_id=marker['instance_id'],
                    true_class=truth['category'], visible_class=marker['category'],
                    eligible_coarse_nodes=eligible_count, selected_views=views,
                    route_costs=route_costs(navigation, views)))
            class_tours = []
            for label in sorted({i['true_class'] for i in instances}):
                group = [i for i in instances if i['true_class'] == label]
                views = [v for i in group for v in i['selected_views']]
                class_tours.append(dict(true_class=label,
                    instance_ids=[i['instance_id'] for i in group],
                    flattened_view_keys=[[i['instance_id'], v['view_index']]
                        for i in group for v in i['selected_views']],
                    route_costs=route_costs(navigation, views)))
            selections.append(dict(asset_id=identity, parent_id=parent,
                condition=condition, instances=instances, class_tours=class_tours))
    plan = dict(schema='semantic_scene_static_sensing.selection.v1',
        status='fixed_before_first_render', asset_manifest_sha256=asset_pin,
        navigation_manifest_sha256=navigation_pin, assets=selections,
        expected_maximum_frames=192,
        planned_frames=sum(len(i['selected_views']) for a in selections for i in a['instances']),
        source_sha256={name: file_sha256(ROOT/name) for name in SOURCE_FILES},
        selection_rule=dict(nodes='original public coarse nodes only',
            xy_range_to_marker_m=[.45, 2.5], minimum_front_projection_m=.3,
            minimum_front_cosine=.4,
            visibility='closed XY segment avoids background and other all-structure envelopes',
            excluded_from_xy_occlusion='own conservative envelope and floor',
            ordering=['Euclidean XY distance to marker', 'lexical node ID'],
            number_per_facility=2, heading='closest 30-degree heading; minimum heading on tie',
            no_pixel_or_fit_result_selection=True, no_replacement_for_failure=True),
        component_target_matching=dict(rule='unique exact-color component with measured median world point within 0.35 m of private target label center',
            maximum_distance_m=.35, performed_before_plane_fitting=True,
            minimum_component_pixels_before_matching=1,
            multiple_matches='ambiguous; refuse without choosing a better-fitting component',
            planning_input=False),
        noise=dict(relative_sigma=.01, seed=230923,
            step='2 * instance_id + view_index; paired conditions share noise field'),
        observation_container_paid_step='fixture noise index only; zero real paid actions',
        worlds_created=0, direct_rgbd_renders=0, tsdf_integrations=0,
        task_scores_read=False, policy_trajectories_created=0,
        truth_access='offline viewpoint and target-component qualification only')
    if plan['planned_frames'] > 192:
        raise ValueError('fixed offline frame cap exceeded')
    return plan, loaded


def inspect_original_marker(observation, marker):
    valid = (observation.depth_m >= .1) & (observation.depth_m <= 4.)
    mask = np.all(observation.rgb == np.asarray(marker['rgb'], np.uint8), axis=-1) & valid
    components = []
    matches = []
    for component in _components(mask):
        pixels = np.sort(component[:, 0]*mask.shape[1]+component[:, 1])
        points = observed_points_v41(observation, pixels)
        measured_center = np.median(points, axis=0)
        target_distance = float(np.linalg.norm(measured_center-marker['center_world_m']))
        row = dict(marker_pixels=len(pixels), target_center_distance_m=target_distance,
            measured_median_world_m=measured_center.tolist())
        components.append(row)
        if target_distance <= .35:
            matches.append(pixels)
    result = dict(exact_color_valid_depth_components=components,
        matched_target_components=len(matches), accepted=False,
        reason='target_component_missing' if not matches else 'target_component_ambiguous',
        marker_pixels=0 if not matches else sum(len(p) for p in matches),
        original_plane_gate='not_reached', original_completeness_gate='not_reached',
        original_fit=None)
    if len(matches) != 1:
        return result
    pixels = matches[0]
    fit = fit_marker_plane_v41(observation, pixels)
    rows, columns = np.divmod(pixels, mask.shape[1])
    plane_gate = ('failed' if fit['reason'] == 'marker_not_reliably_planar' else
                  'passed' if 'plane_rms_m' in fit else 'not_reached')
    completeness = 'not_reached'
    if 'observed_marker_extent_m' in fit:
        completeness = ('failed' if fit['reason'] == 'marker_extent_incomplete_or_wrong_mount'
                        else 'passed')
    result.update(accepted=fit['accepted'], reason=fit['reason'], marker_pixels=len(pixels),
        pixel_bounds_xy=[int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())],
        touches_image_border=bool(np.any((rows == 0) | (rows == mask.shape[0]-1)
            | (columns == 0) | (columns == mask.shape[1]-1))),
        original_plane_gate=plane_gate, original_completeness_gate=completeness,
        original_fit=fit)
    if len(pixels) >= 3:
        points = observed_points_v41(observation, pixels)
        singular = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
        result['svd_singular_values'] = singular.tolist()
        result['svd_small_to_middle_ratio'] = (float(singular[2]/singular[1])
                                               if singular[1] > 0 else None)
    return result


def execute(plan, loaded, output):
    before = runtime_counts_v41()
    frames, instances = [], []
    with (output/'frames.jsonl').open('x') as stream:
        for asset_entry in plan['assets']:
            identity = asset_entry['asset_id']
            asset, navigation = loaded[identity]
            spec = navigation['public_spec']
            sensor = spec['sensor']
            for instance in asset_entry['instances']:
                marker = next(m for m in asset['markers'] if m['instance_id'] == instance['instance_id'])
                results = []
                for view in instance['selected_views']:
                    frame_id = f"static_qualification/{identity}/{instance['instance_id']}/{view['view_index']}"
                    transform = camera_transform_xyyaw([*view['xy_m'], view['heading']*math.pi/6],
                        height_m=spec['motion']['camera_height_m'])
                    arrays = asset['arrays']
                    rgb, depth = render_rgbd_arrays(arrays['vertices'], arrays['triangles'],
                        intrinsic=np.asarray(sensor['intrinsic']), world_from_camera=transform,
                        width=sensor['width'], height=sensor['height'],
                        triangle_rgb=arrays['triangle_rgb'], triangle_instance_id=arrays['triangle_instance_id'],
                        marker_patches=asset['markers'], minimum_depth_m=.1, maximum_depth_m=4.,
                        relative_sigma=.01, noise_seed=230923, step=view['noise_step'])
                    observation = PaidRGBDObservationV40(frame_id, view['noise_step'], rgb, depth,
                        np.asarray(sensor['intrinsic']), transform)
                    filename = f"{identity}__i{instance['instance_id']}__v{view['view_index']}.npz"
                    path = output/'frames'/filename
                    with path.open('xb') as packet:
                        np.savez_compressed(packet, frame_id=observation.frame_id,
                            paid_step=observation.paid_step, rgb=rgb, depth_m=depth,
                            intrinsic=observation.intrinsic, world_from_camera=transform)
                    diagnosis = inspect_original_marker(observation, marker)
                    row = dict(asset_id=identity, instance_id=instance['instance_id'],
                        view=view, frame_path='frames/'+filename, frame_sha256=file_sha256(path),
                        observation_sha256=observation.sha256(), diagnosis=diagnosis)
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                    stream.flush()
                    frames.append(row); results.append(diagnosis)
                instances.append(dict(asset_id=identity, parent_id=asset_entry['parent_id'],
                    condition=asset_entry['condition'], instance_id=instance['instance_id'],
                    selected_views=len(results), qualified_views=sum(r['accepted'] for r in results),
                    reasons=[r['reason'] for r in results], route_costs=instance['route_costs']))
    counts = Counter(i['qualified_views'] for i in instances)
    by_parent = []
    for parent in PARENTS:
        group = [i for i in instances if i['parent_id'] == parent]
        hist = Counter(i['qualified_views'] for i in group)
        by_parent.append(dict(parent_id=parent, facilities=len(group),
            qualified_views=sum(i['qualified_views'] for i in group),
            facilities_by_qualified_views={str(k): hist[k] for k in range(3)}))
    after = runtime_counts_v41()
    if before != after or len(frames) != plan['planned_frames']:
        raise AssertionError('offline counter or frame-count contract violated')
    if any(file_sha256(ROOT/name) != pin for name, pin in plan['source_sha256'].items()):
        raise AssertionError('qualification source changed during the one fixed run')
    return dict(schema='semantic_scene_static_sensing.report.v1',
        selection_sha256=file_sha256(output/'selection.json'),
        frames_jsonl_sha256=file_sha256(output/'frames.jsonl'),
        assets=24, facilities=len(instances), direct_rgbd_renders=len(frames),
        worlds_created=0, paid_actions=0, tsdf_integrations=0, task_quality_evaluations=0,
        original_runtime_counters_before=before, original_runtime_counters_after=after,
        qualified_frames=sum(f['diagnosis']['accepted'] for f in frames),
        facilities_by_qualified_views={str(k): counts[k] for k in range(3)},
        frame_reasons=dict(Counter(f['diagnosis']['reason'] for f in frames)),
        plane_gate_counts=dict(Counter(f['diagnosis']['original_plane_gate'] for f in frames)),
        completeness_gate_counts=dict(Counter(f['diagnosis']['original_completeness_gate'] for f in frames)),
        per_parent=by_parent, per_facility=instances,
        boundaries=[
            'Fixed private-geometry camera fixtures, not autonomous paid trajectories or method results.',
            'The PaidRGBDObservation container step is only a noise/fixture index in this offline check.',
            'Original exact-color components and original plane/completeness gates; no thresholds changed.',
            'A fit accepted here establishes only this limited label-interface necessary condition.',
            'No learned recognition, residual correctness, semantic gain, reconstruction quality, or planner reachability claim.',
            'No retries, replacement viewpoints, layout changes, fusion, task scoring or World construction.',
            'Graph costs include all movement/turns and explicit extra observe actions, then exact home heading return.',
            'Movement-arrival frames already observe; the explicit observe is not a first diagnostic view.',
            'All private target choices and truth associations remain outside runtime planner inputs.'])


def markdown_report(report):
    lines = ['# 新场景固定视点感知资格检查', '',
        '这是一轮开发资产的离线必要条件检查，不是自主付费轨迹、建图评分或语义方法效果实验。', '',
        f"- 24 资产、96 设施；固定离线 RGB-D 渲染 {report['direct_rgbd_renders']} 帧；World 0、实际付费动作 0、融合 0、任务评分 0。",
        f"- 原平面拟合合格 {report['qualified_frames']} 帧。设施具备 0／1／2 个合格视点的数量："+
            '／'.join(str(report['facilities_by_qualified_views'][str(i)]) for i in range(3))+'。',
        '- 所有视点先写入 selection.json；只选两个最近静态合格公共粗节点，未根据像素或拟合结果补点。',
        '- 原 96×72 内参、1% 相对深度噪声、seed=230923；同父同设施同视点在四条件中复用噪声序号。',
        '- 原精确颜色连通组件及原平面门限保持不变。原类型中的 paid_step 在本检查只是噪声序号，不代表已执行动作。', '',
        '| 父布局 | 设施数（含四条件） | 合格帧 | 0 视点 | 1 视点 | 2 视点 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in report['per_parent']:
        counts = row['facilities_by_qualified_views']
        lines.append(f"| {row['parent_id']} | {row['facilities']} | {row['qualified_views']} | {counts['0']} | {counts['1']} | {counts['2']} |")
    lines += ['', '原拟合／关联原因：', '']
    lines += [f'- `{reason}`：{count}' for reason, count in sorted(report['frame_reasons'].items())]
    lines += ['', 'selection.json 保存每个视点完整往返动作数、每设施两个视点及每同类批次四个视点的最短遍历返回动作数；都包含 0.25 m 移动、30° 转角、额外 observe 和精确起点朝向返回。实际移动到达帧本身已经含观察，额外 observe 仅是预算和宏终止步骤。静态路线没有执行，不能保证策略会选择它，也未要求通过近邻 32 候选集逐步触达。', '',
        'report.json 保存每设施 0／1／2 视点结果和完整原因；frames.jsonl 保存逐帧像素数、边界触碰、SVD、原平面／完整性门及输入 SHA；frames/ 保存这一次渲染的 RGB-D 原包。早退前未执行的完整性门记为 not_reached，不能当作通过。', '',
        '这些设施仍共享受控标签和几何原型。合格只证明指定静态视点能通过当前受控标签接口，不证明识别泛化、结构反馈准确、语义收益或完整架构领先。未合格资产原样保留，没有改变布局或渲染更多视点。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, default=ROOT/'audit_results/semantic_scene_assets_20260923')
    parser.add_argument('--navigation-root', type=Path, default=ROOT/'audit_results/semantic_scene_navigation_20260923')
    parser.add_argument('--asset-manifest-sha256', default=ASSET_PIN)
    parser.add_argument('--navigation-manifest-sha256', default=NAV_PIN)
    parser.add_argument('--output', type=Path, default=ROOT/'audit_results/semantic_scene_sensing_20260923')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('one fixed qualification cannot overwrite previous evidence')
    plan, loaded = selection_plan(args.asset_root.resolve(), args.navigation_root.resolve(),
        args.asset_manifest_sha256, args.navigation_manifest_sha256)
    output.mkdir(parents=True)
    (output/'frames').mkdir()
    write_json(output/'selection.json', plan)
    print(json.dumps(dict(selection_saved_before_render=True, planned_frames=plan['planned_frames'],
        selection_sha256=file_sha256(output/'selection.json'))), flush=True)
    report = execute(plan, loaded, output)
    write_json(output/'report.json', report)
    with (output/'report.md').open('x') as stream:
        stream.write(markdown_report(report))
    print(json.dumps(dict(output=str(output), direct_rgbd_renders=report['direct_rgbd_renders'],
        worlds_created=0, qualified_frames=report['qualified_frames'],
        facilities_by_qualified_views=report['facilities_by_qualified_views'],
        frame_reasons=report['frame_reasons']), sort_keys=True))


if __name__ == '__main__':
    main()
