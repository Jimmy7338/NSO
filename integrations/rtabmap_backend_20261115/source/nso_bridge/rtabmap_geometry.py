"""Declared RTAB keyframe correction of original measured sensor poses.

Pixels, ranges, timestamps and original TF receipts are never replaced. A
non-keyframe pose uses the optimized preceding keyframe and measured relative
odometry. This is a physical input adapter, not independent pose ground truth.
"""
import json
import numpy as np

PROFILE = 'rtabmap_optimized_replay_v1'
WORLD_FRAME = 'nso_pair_map'  # Explicit identity alias of nso_loop_map.


def rigid(value):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1])
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1., atol=1e-5)):
        raise ValueError('finite rigid measured/optimized transform required')
    return a


def correction_at(snapshot, stamp_ns):
    if type(stamp_ns) is not int or stamp_ns <= 0:
        raise ValueError('original positive sensor stamp required')
    nodes = sorted((n for n in snapshot['nodes']
                    if type(n.get('stamp_ns')) is int and n['stamp_ns'] > 0
                    and n.get('odom_from_base') is not None and n.get('map_from_base') is not None),
                   key=lambda n: (n['stamp_ns'], n['id']))
    if not nodes:
        raise ValueError('optimized graph has no time-bound raw keyframe poses')
    predecessors = [n for n in nodes if n['stamp_ns'] <= stamp_ns]
    node = predecessors[-1] if predecessors else nodes[0]
    correction = rigid(node['map_from_base']) @ np.linalg.inv(rigid(node['odom_from_base']))
    return correction, dict(
        method='optimized_keyframe_plus_original_measured_relative_odometry',
        node_id=node['id'], node_stamp_ns=node['stamp_ns'], sensor_stamp_ns=stamp_ns,
        sensor_minus_node_s=(stamp_ns-node['stamp_ns'])/1e9,
        temporal_association=('preceding_keyframe' if predecessors else 'first_keyframe_backwards_replay'),
        graph_revision_sha256=snapshot['graph_revision_sha256'],
        graph_node_metadata_available=len(nodes), graph_nodes_total=len(snapshot['nodes']),
        source_graph_stamp_ns=snapshot['graph_source_stamp_ns'],
        map_from_odom=correction.tolist(), independent_pose_accuracy_verified=False)


def corrected_geometry(source, snapshot):
    receipt = json.loads(str(np.asarray(source['receipt_json']).item()))
    historical = receipt['historical_tf']
    depth_stamp = int(receipt['headers']['depth']['stamp_ns'])
    scan_stamp = int(receipt['headers']['scan']['stamp_ns'])
    rgbd_correction, rgbd_receipt = correction_at(snapshot, depth_stamp)
    scan_correction, scan_receipt = correction_at(snapshot, scan_stamp)
    # Always start from the immutable original TF, including on repeated replay.
    camera = rgbd_correction @ rigid(historical['world_from_camera']['matrix'])
    base = rgbd_correction @ rigid(historical['odom_from_base']['matrix'])
    laser = scan_correction @ rigid(historical['odom_from_laser']['matrix'])
    updates = dict(world_from_camera=camera, world_from_laser=laser,
        base_odom_pose=np.asarray([base[0, 3], base[1, 3], np.arctan2(base[1, 0], base[0, 0])]))
    return updates, dict(profile=PROFILE, geometry_world_frame=WORLD_FRAME,
        rgbd=rgbd_receipt, scan=scan_receipt, original_source_receipt_preserved=True,
        original_pixel_range_and_stamp_values_preserved=True,
        legacy_odom_field_names='RPC compatibility names; values are in explicitly declared RTAB world')
