#!/usr/bin/env python3
"""Export an ended mission offline. No ROS node, master, hardware or playback.

Usage: source /opt/ros/noetic/setup.bash
       /usr/bin/python3 export_episode.py --report /absolute/ended/report

Only the chosen report is read. Outputs are result_summary.txt,
trajectory/map.png, trajectory/poses.csv, paired_replay.mp4 and export metadata.
--self-test checks rotated OccupancyGrid conversion and source-time TF lookup.
"""
import argparse
import bisect
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def ns(stamp):
    return int(stamp.to_nsec())


def yaw_from_quaternion(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def rigid_matrix(translation, quaternion):
    x, y, z, w = [float(getattr(quaternion, key)) for key in ('x', 'y', 'z', 'w')]
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('invalid recorded TF quaternion')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    out = np.eye(4)
    out[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    out[:3, 3] = [translation.x, translation.y, translation.z]
    if not np.isfinite(out).all():
        raise ValueError('nonfinite recorded TF')
    return out


def world_to_grid(meta, x, y):
    """ROS lower-left cell coordinates; rotated origins are supported."""
    dx, dy = x-meta['origin_x'], y-meta['origin_y']
    c, s = math.cos(meta['origin_yaw']), math.sin(meta['origin_yaw'])
    return ((c*dx+s*dy)/meta['resolution'], (-s*dx+c*dy)/meta['resolution'])


def world_to_image(meta, x, y):
    col, row = world_to_grid(meta, x, y)
    return int(math.floor(col)), meta['height']-1-int(math.floor(row))


class PastSeries:
    def __init__(self, rows=()):
        # At duplicate source timestamps retain the last recorded value.
        unique = {row['stamp_ns']: row for row in rows}
        self.rows = [unique[t] for t in sorted(unique)]
        self.times = [row['stamp_ns'] for row in self.rows]

    def past(self, timestamp):
        index = bisect.bisect_right(self.times, int(timestamp))-1
        return self.rows[index] if index >= 0 else None


def composed_map_poses(map_odom, odom_base):
    result = []
    history = PastSeries(map_odom)
    for row in PastSeries(odom_base).rows:
        correction = history.past(row['stamp_ns'])
        if correction is None:
            continue
        matrix = correction['matrix'] @ row['matrix']
        result.append(dict(stamp_ns=row['stamp_ns'], x=float(matrix[0, 3]), y=float(matrix[1, 3]),
            yaw=math.atan2(matrix[1, 0], matrix[0, 0]), map_tf_stamp_ns=correction['stamp_ns'],
            odom_tf_stamp_ns=row['stamp_ns'], map_tf_age_s=(row['stamp_ns']-correction['stamp_ns'])/1e9,
            timestamp_basis='source-header TF; nearest-past map<-odom composed with odom<-base_link'))
    return result


def load_bag(path, map_frame='map', odom_frame='odom', base_frame='base_link'):
    import rosbag
    maps, paths, goals, terminals, map_odom, odom_base = [], [], [], [], [], []
    warnings = []
    with rosbag.Bag(str(path), 'r') as bag:
        start, end = int(bag.get_start_time()*1e9), int(bag.get_end_time()*1e9)
        # Topic suffixes support both the mission namespace and standard ROS.
        selected = [topic for topic in bag.get_type_and_topic_info().topics
            if topic in ('/tf', '/tf_static') or topic.endswith(('/map', '/goal', '/result',
                         '/NavfnROS/plan', '/global_plan', '/local_plan'))]
        for topic, message, received in bag.read_messages(topics=selected):
            received_ns = ns(received)
            if topic in ('/tf', '/tf_static'):
                for transform in message.transforms:
                    parent, child = transform.header.frame_id.lstrip('/'), transform.child_frame_id.lstrip('/')
                    pair = (parent, child)
                    direct = pair in ((map_frame, odom_frame), (odom_frame, base_frame))
                    reverse = pair in ((odom_frame, map_frame), (base_frame, odom_frame))
                    if not direct and not reverse:
                        continue
                    matrix = rigid_matrix(transform.transform.translation, transform.transform.rotation)
                    if reverse:
                        matrix = np.linalg.inv(matrix)
                        pair = pair[::-1]
                    stamp = ns(transform.header.stamp)
                    if topic == '/tf_static':
                        stamp = 0
                    elif not stamp:
                        warnings.append('TF with zero source header uses bag receipt time')
                        stamp = received_ns
                    row = dict(stamp_ns=stamp, received_ns=received_ns, matrix=matrix)
                    (map_odom if pair == (map_frame, odom_frame) else odom_base).append(row)
                continue
            kind = getattr(message, '_type', '')
            if kind == 'nav_msgs/OccupancyGrid':
                frame = message.header.frame_id.lstrip('/')
                if frame != map_frame:
                    warnings.append('Skipped occupancy frame '+frame+'; expected '+map_frame)
                    continue
                info = message.info
                maps.append(dict(stamp_ns=ns(message.header.stamp) or received_ns,
                    received_ns=received_ns, width=info.width, height=info.height,
                    resolution=info.resolution, origin_x=info.origin.position.x, origin_y=info.origin.position.y,
                    origin_yaw=yaw_from_quaternion(info.origin.orientation),
                    data=np.asarray(message.data, dtype=np.int16).reshape(info.height, info.width)))
            elif kind == 'nav_msgs/Path':
                paths.append(dict(stamp_ns=ns(message.header.stamp) or received_ns,
                    received_ns=received_ns, frame=message.header.frame_id.lstrip('/'),
                    kind='local' if topic.endswith('/local_plan') else 'global',
                    topic=topic, xy=[(p.pose.position.x, p.pose.position.y) for p in message.poses]))
            elif kind == 'move_base_msgs/MoveBaseActionGoal':
                target = message.goal.target_pose
                goals.append(dict(stamp_ns=ns(message.header.stamp) or received_ns,
                    received_ns=received_ns, frame=target.header.frame_id.lstrip('/'),
                    id=message.goal_id.id, x=target.pose.position.x, y=target.pose.position.y,
                    yaw=yaw_from_quaternion(target.pose.orientation)))
            elif kind == 'move_base_msgs/MoveBaseActionResult':
                terminals.append(dict(stamp_ns=ns(message.header.stamp) or received_ns,
                    id=message.status.goal_id.id, status=message.status.status, text=message.status.text))
    correction_history = PastSeries(map_odom)
    for path_row in paths:
        if path_row['frame'] == odom_frame:
            transform = correction_history.past(path_row['stamp_ns'])
            if transform is not None:
                points = np.asarray([(x, y, 0., 1.) for x, y in path_row['xy']]).reshape(-1, 4)
                path_row['xy'] = (points @ transform['matrix'].T)[:, :2].tolist()
                path_row['frame'] = map_frame
                path_row['converted_with_map_tf_stamp_ns'] = transform['stamp_ns']
        if path_row['frame'] != map_frame:
            warnings.append('Skipped path frame '+path_row['frame'])
    poses = composed_map_poses(map_odom, odom_base)
    if not poses:
        warnings.append('No source-time map<-odom<-base_link TF composition; no odom-integrated trajectory substituted')
    return dict(start_ns=start, end_ns=end, maps=PastSeries(maps),
        paths=PastSeries([p for p in paths if p['frame'] == map_frame]),
        goals=PastSeries([g for g in goals if g['frame'] == map_frame]), terminals=PastSeries(terminals),
        poses=PastSeries(poses), warnings=sorted(set(warnings)),
        map_odom_count=len(map_odom), odom_base_count=len(odom_base))


def camera_sources(report):
    rows, warnings = [], []
    index = report/'video/timestamps.jsonl'
    if index.is_file():
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            timestamp = int(row['source_stamp_ns'])
            path = report/'video/frames'/('%d.jpg' % timestamp)
            if path.is_file():
                rows.append(dict(stamp_ns=timestamp, path=str(path), format='jpeg'))
            else:
                warnings.append('Missing camera JPEG for source timestamp '+str(timestamp))
    if rows:
        return PastSeries(rows), 'source-timestamped sampled JPEG video; nearest-past frame held between acquisitions', warnings
    for directory in ('frames', 'paid'):
        for path in sorted((report/directory).glob('*.npz')):
            try:
                with np.load(str(path), allow_pickle=False) as frame:
                    if not any(key in frame for key in ('rgb', 'color_rgb')):
                        continue
                    stamp = int(round(float(frame['timestamp_s'])*1e9))
                    opaque_id = str(frame['frame_id'].item()) if 'frame_id' in frame else ''
                    # Mission frame IDs preserve integer nanoseconds better
                    # than floating Unix seconds. Use only a matching value.
                    if opaque_id.isdigit() and abs(int(opaque_id)-stamp) <= 1024:
                        stamp = int(opaque_id)
                    rows.append(dict(stamp_ns=stamp, path=str(path), format='npz'))
            except (OSError, ValueError, KeyError) as error:
                warnings.append('Could not read saved RGB '+path.name+': '+str(error))
    return PastSeries(rows), 'SPARSE saved RGB-D NPZ acquisitions; held images are not continuous video', warnings


def fallback_meta(poses, goals):
    xy = [(r['x'], r['y']) for r in poses.rows+goals.rows] or [(0., 0.)]
    low, high = np.min(xy, axis=0)-1., np.max(xy, axis=0)+1.
    resolution = max(.02, float(max(high-low))/600.)
    width, height = [max(2, int(math.ceil(v/resolution))) for v in high-low]
    return dict(width=width, height=height, resolution=resolution, origin_x=float(low[0]),
                origin_y=float(low[1]), origin_yaw=0., data=np.full((height, width), -1), stamp_ns=None)


def goal_outcomes(bag, result, timestamp=None):
    """Retain every goal, including failed terminals; verify only report evidence."""
    cutoff = timestamp if timestamp is not None else float('inf')
    verified = {row.get('goal', {}).get('id') for row in result.get('successful_arrivals', [])}
    terminals = {row['id']: row for row in bag['terminals'].rows if row['stamp_ns'] <= cutoff}
    output, seen = [], set()
    for goal in bag['goals'].rows:
        if goal['stamp_ns'] > cutoff or goal['id'] in seen:
            continue
        seen.add(goal['id'])
        terminal = terminals.get(goal['id'], {})
        code = terminal.get('status')
        outcome = ('verified' if code == 3 and goal['id'] in verified else
                   'succeeded_unverified' if code == 3 else
                   'failed' if code in (2, 4, 5, 8, 9) else 'active_or_no_recorded_terminal')
        output.append(dict(goal, number=len(output)+1, outcome=outcome,
            verified_arrival=outcome == 'verified', terminal_status=code,
            terminal_text=terminal.get('text'), terminal_source_ns=terminal.get('stamp_ns')))
    return output


def text(image, value, xy, scale=.48, color=(235, 235, 235), thickness=1):
    import cv2
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_map(meta, poses, goal=None, path=None, size=(640, 600), title='', all_goals=()):
    import cv2
    width, height = size
    canvas = np.full((height, width, 3), 32, np.uint8)
    values = meta['data']
    gray = np.where(values < 0, 140, np.clip(250-2.2*values, 20, 250)).astype(np.uint8)
    source = cv2.cvtColor(np.flipud(gray), cv2.COLOR_GRAY2BGR)
    scale = min((width-30)/source.shape[1], (height-90)/source.shape[0])
    scaled_width, scaled_height = max(1, int(source.shape[1]*scale)), max(1, int(source.shape[0]*scale))
    left, top = (width-scaled_width)//2, 40+(height-90-scaled_height)//2
    source = cv2.resize(source, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
    canvas[top:top+scaled_height, left:left+scaled_width] = source

    def pixel(x, y):
        col, row = world_to_grid(meta, x, y)
        return int(round(left+col*scale)), int(round(top+(meta['height']-row)*scale))

    def polyline(xy, color, line_width):
        if len(xy) >= 2:
            points = np.asarray([pixel(x, y) for x, y in xy], dtype=np.int32)
            cv2.polylines(canvas, [points], False, color, line_width, cv2.LINE_AA)
    if path:
        polyline(path['xy'], (185, 90, 190), 2)
    polyline([(p['x'], p['y']) for p in poses], (235, 135, 35), 2)
    if poses:
        first, current = poses[0], poses[-1]
        cv2.circle(canvas, pixel(first['x'], first['y']), 5, (255, 220, 80), -1)
        center = pixel(current['x'], current['y'])
        cv2.circle(canvas, center, 7, (55, 200, 70), -1)
        image_yaw = current['yaw']-meta['origin_yaw']
        endpoint = (int(round(center[0]+20*math.cos(image_yaw))),
                    int(round(center[1]-20*math.sin(image_yaw))))
        cv2.arrowedLine(canvas, center, endpoint, (30, 140, 40), 2, cv2.LINE_AA, tipLength=.35)
    for target in all_goals or ([goal] if goal else []):
        point = pixel(target['x'], target['y'])
        color = ((50, 190, 50) if target.get('outcome') == 'verified' else
                 (60, 60, 240) if target.get('outcome') == 'failed' else (20, 150, 255))
        cv2.drawMarker(canvas, point, color, cv2.MARKER_TILTED_CROSS, 16, 2)
        text(canvas, '#'+str(target.get('number', '?')), (point[0]+7, point[1]-7), .46, color, 2)
    text(canvas, title, (12, 20), .5)
    text(canvas, 'Blue: TF path  Green arrow: robot  Purple: planner path', (12, height-28), .38)
    text(canvas, 'Targets: green=verified red=failed orange=active; gray=unknown', (12, height-10), .36)
    return canvas


def load_camera(row):
    import cv2
    if row['format'] == 'jpeg':
        image = cv2.imread(row['path'])
        if image is None:
            raise ValueError('unreadable saved camera JPEG '+row['path'])
        return image
    with np.load(row['path'], allow_pickle=False) as source:
        rgb = source['rgb'] if 'rgb' in source else source['color_rgb']
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_replay(report, bag, cameras, camera_disclosure, result, fps):
    import cv2
    start = min(bag['start_ns'], cameras.times[0]) if cameras.rows else bag['start_ns']
    end = max(bag['end_ns'], cameras.times[-1]) if cameras.rows else bag['end_ns']
    duration = max(1./fps, (end-start)/1e9)
    count = max(1, int(math.ceil(duration*fps))+1)
    output = report/'paired_replay.mp4'
    command = ['/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', '1280x720', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
        '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    status = str(result.get('status', 'unknown'))
    adapter = read_json(report/'adapter/result.json', result.get('adapter_result', {})) or {}
    distance = adapter.get('can_distance_m')
    final_goals = goal_outcomes(bag, result)
    failed_count = sum(row['outcome'] == 'failed' for row in final_goals)
    fallback = fallback_meta(bag['poses'], bag['goals'])
    cached, camera_image, frame_metadata = None, None, []
    with (report/'replay_ffmpeg.log').open('w') as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index in range(count):
                timestamp = min(end, start+int(index/fps*1e9))
                camera = cameras.past(timestamp)
                if camera and camera['path'] != cached:
                    camera_image = load_camera(camera)
                    cached = camera['path']
                pose = bag['poses'].past(timestamp)
                history_end = bisect.bisect_right(bag['poses'].times, timestamp)
                history = bag['poses'].rows[:history_end]
                map_row = bag['maps'].past(timestamp)
                goal, path = bag['goals'].past(timestamp), bag['paths'].past(timestamp)
                current_goals = goal_outcomes(bag, result, timestamp)
                canvas = np.full((720, 1280, 3), 24, np.uint8)
                text(canvas, str(result.get('session_id', report.name))+'  status='+status, (18, 28), .65,
                     (70, 90, 255) if status == 'failed' else (235, 235, 235), 2)
                text(canvas, 'SOURCE-TIME REPLAY | elapsed %.2fs | source_ns=%d' % ((timestamp-start)/1e9, timestamp),
                     (18, 55), .5)
                if camera_image is not None:
                    image = cv2.resize(camera_image, (640, 360), interpolation=cv2.INTER_AREA)
                    canvas[82:442, :640] = image
                else:
                    text(canvas, 'No earlier saved camera acquisition', (35, 250), .6)
                camera_age = None if camera is None else (timestamp-camera['stamp_ns'])/1e9
                label = ('SPARSE NPZ - HOLD LAST ACQUISITION' if camera and camera['format'] == 'npz' else
                         'SAMPLED CAMERA - NEAREST PAST')
                text(canvas, label, (16, 470), .51, (70, 160, 255))
                text(canvas, 'Camera source_ns='+str(camera['stamp_ns'] if camera else None), (16, 497))
                text(canvas, 'Camera hold age: '+('unavailable' if camera_age is None else '%.3fs' % camera_age), (16, 521))
                text(canvas, 'TF map pose: '+('unavailable' if pose is None else 'x=%.3f y=%.3f yaw=%.3f' %
                    (pose['x'], pose['y'], pose['yaw'])), (16, 549))
                age = None if pose is None else (timestamp-pose['stamp_ns'])/1e9
                text(canvas, 'TF pose hold / map correction age: '+('unavailable' if pose is None else
                     '%.3fs / %.3fs' % (age, pose['map_tf_age_s'])), (16, 573))
                text(canvas, 'CAN distance (final adapter): '+('unavailable' if distance is None else '%.4f m' % distance),
                     (16, 602), .49)
                text(canvas, 'Final verified arrivals: %d | failed goals: %d' %
                     (len(result.get('successful_arrivals', [])), failed_count), (16, 627), .49)
                reason_lines = textwrap.wrap('Ended: '+str(result.get('reason') or result.get('status')), width=81)
                for line_number, line in enumerate(reason_lines[:3]):
                    text(canvas, line, (16, 654+20*line_number), .37, (180, 180, 240))
                map_title = ('Map source_ns='+str(map_row['stamp_ns']) if map_row else 'No recorded map yet')
                canvas[82:682, 640:1280] = render_map(map_row or fallback, history, goal, path,
                                                     title=map_title, all_goals=current_goals)
                text(canvas, 'SLAM estimate; no motion-capture ground truth. Final CAN/arrival facts are not time-series.',
                     (16, 715), .39)
                encoder.stdin.write(canvas.tobytes())
                frame_metadata.append(dict(video_frame=index, timeline_source_ns=timestamp,
                    camera_source_ns=camera['stamp_ns'] if camera else None, camera_hold_age_s=camera_age,
                    map_source_ns=map_row['stamp_ns'] if map_row else None,
                    pose_source_ns=pose['stamp_ns'] if pose else None, pose_hold_age_s=age,
                    map_tf_source_ns=pose['map_tf_stamp_ns'] if pose else None,
                    goal_id=goal['id'] if goal else None,
                    goal_outcomes=[dict(number=g['number'], id=g['id'], outcome=g['outcome']) for g in current_goals]))
        finally:
            encoder.stdin.close()
        if encoder.wait(timeout=120) != 0:
            raise RuntimeError('ffmpeg encoding failed; see replay_ffmpeg.log')
    with (report/'paired_replay_timestamps.jsonl').open('w') as stream:
        for row in frame_metadata:
            stream.write(json.dumps(row)+'\n')
    return dict(path=str(output), width=1280, height=720, fps=fps, frames=count,
        timeline_start_ns=start, timeline_end_ns=end, source_duration_s=duration,
        camera_disclosure=camera_disclosure, camera_acquisitions=len(cameras.rows),
        alignment='nearest-past source time; no future camera/TF/map chosen; held frames explicitly labeled')


def export_report(report, fps=5., map_frame='map', odom_frame='odom', base_frame='base_link'):
    import cv2
    result_path = report/'result.json'
    if not result_path.is_file():
        raise ValueError('result.json required; exporter only handles ended reports')
    result = read_json(result_path)
    if result.get('status') in ('running', 'active', 'preparing', None):
        raise ValueError('report has no terminal mission status')
    bag_path = report/'navigation.bag'
    if not bag_path.is_file():
        raise ValueError('navigation.bag required for source TF/map reconstruction')
    bag = load_bag(bag_path, map_frame, odom_frame, base_frame)
    cameras, disclosure, camera_warnings = camera_sources(report)
    trajectory = report/'trajectory'
    trajectory.mkdir(exist_ok=True)
    with (trajectory/'poses.csv').open('w', newline='') as stream:
        fields = ['stamp_ns', 'x', 'y', 'yaw', 'map_tf_stamp_ns', 'odom_tf_stamp_ns', 'map_tf_age_s', 'timestamp_basis']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bag['poses'].rows)
    final_map = bag['maps'].rows[-1] if bag['maps'].rows else fallback_meta(bag['poses'], bag['goals'])
    final_goal = bag['goals'].rows[-1] if bag['goals'].rows else None
    final_path = bag['paths'].rows[-1] if bag['paths'].rows else None
    outcomes = goal_outcomes(bag, result)
    image = render_map(final_map, bag['poses'].rows, final_goal, final_path, (1000, 1000),
        title='FINAL MAP + recorded TF trajectory | status='+str(result.get('status')), all_goals=outcomes)
    if not cv2.imwrite(str(trajectory/'map.png'), image):
        raise RuntimeError('failed to save trajectory/map.png')
    replay = encode_replay(report, bag, cameras, disclosure, result, fps)
    adapter = read_json(report/'adapter/result.json', result.get('adapter_result', {})) or {}
    warnings = bag['warnings']+camera_warnings
    metadata = dict(schema='nso.ended_episode_export.v1', report=str(report), mission_status=result['status'],
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        result_reason=result.get('reason') or result.get('status'), mission_elapsed_run_s=result.get('elapsed_run_s'),
        source_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        actual_CAN_distance_m=adapter.get('can_distance_m'), CAN_distance_source='adapter/result.json:can_distance_m',
        feedback_accounting_complete=adapter.get('feedback_accounting_complete'),
        independently_observed_physical_stop=adapter.get('physical_stop_independently_observed'),
        verified_arrivals=result.get('successful_arrivals', []),
        goal_outcomes=outcomes, failed_goal_count=sum(row['outcome'] == 'failed' for row in outcomes),
        feedback_zero_before_close=result.get('feedback_zero_before_close'),
        feedback_zero_after_close=result.get('feedback_zero_after_close'),
        semantic_benefit_claimed=False,
        recorded_goals=len(bag['goals'].rows), recorded_map_messages=len(bag['maps'].rows),
        map_source_stamps_ns=bag['maps'].times, map_bag_receipt_stamps_ns=[r['received_ns'] for r in bag['maps'].rows],
        trajectory_source='bag /tf map<-odom @ nearest-past source time composed with odom<-base_link; no CAN/odom integration',
        map_pose_is_ground_truth=False, source_time_alignment_proves_online_availability=False,
        map_pose_samples=len(bag['poses'].rows), tf_map_odom_samples=bag['map_odom_count'],
        tf_odom_base_samples=bag['odom_base_count'], replay=replay, warnings=warnings)
    (report/'export_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = [str(result.get('session_id', report.name)), 'Status: '+str(result['status']),
        'Ended because: '+str(result.get('reason') or result.get('status')),
        'Actual mission run time: '+str(result.get('elapsed_run_s'))+' s; no ten-minute completion implied',
        'Verified arrivals: '+str(len(result.get('successful_arrivals', []))),
        'Failed terminal goals retained: '+str(metadata['failed_goal_count']),
        'Recorded map-frame goals: '+str(len(bag['goals'].rows)),
        'Actual CAN distance: '+str(adapter.get('can_distance_m'))+' m (adapter/result.json; not visual odometry integration)',
        'CAN feedback accounting complete: '+str(adapter.get('feedback_accounting_complete')),
        'Physical stop independently observed: '+str(adapter.get('physical_stop_independently_observed')),
        'Trajectory: '+metadata['trajectory_source'], 'Map-frame pose samples: '+str(len(bag['poses'].rows)),
        'Trajectory is a visual/SLAM estimate, not motion-capture ground truth or independent proof of travel.',
        'TF resampling: nearest-past source header; pose and map-correction ages are shown in the replay.',
        'Occupancy map source timestamps ns: '+str(bag['maps'].times),
        'Occupancy map bag receipt timestamps ns: '+str(metadata['map_bag_receipt_stamps_ns']),
        'Camera: '+disclosure, 'Saved camera acquisitions: '+str(len(cameras.rows)),
        'Replay: source timeline %.3f s at %.2f FPS; gaps hold the last acquisition and display its age.' %
            (replay['source_duration_s'], fps),
        'Video and map snapshots use source time; old latched map timestamps are preserved.',
        'Final CAN distance/arrival/status labels are final report facts, not measurements at each replay frame.',
        'No motion, failed goals, no verified arrival, and unavailable TF are retained as recorded.',
        'No semantic benefit is established or claimed by this replay.',
        'Outputs: trajectory/map.png, trajectory/poses.csv, paired_replay.mp4, paired_replay_timestamps.jsonl, export_metadata.json']
    for goal in outcomes:
        lines.append('Goal #%d: %s; target=(%.3f, %.3f); source_ns=%d; terminal_status=%s; %s' %
            (goal['number'], goal['outcome'], goal['x'], goal['y'], goal['stamp_ns'],
             goal['terminal_status'], goal['terminal_text']))
    lines += ['Warning: '+warning for warning in warnings]
    (report/'result_summary.txt').write_text('\n'.join(lines)+'\n')
    return metadata


def self_test():
    import unittest

    class PureConversions(unittest.TestCase):
        def test_rotated_map_origin_and_row_flip(self):
            meta = dict(origin_x=10., origin_y=20., origin_yaw=math.pi/2, resolution=.5, height=10)
            col, row = world_to_grid(meta, 8.5, 21.)
            self.assertAlmostEqual(col, 2.)
            self.assertAlmostEqual(row, 3.)
            # Cell-center location avoids floor ambiguity on exact boundaries.
            self.assertEqual(world_to_image(meta, 8.25, 21.25), (2, 6))

        def test_nearest_past_never_uses_future_transform(self):
            self.assertIsNone(PastSeries([dict(stamp_ns=20)]).past(19))
            series = PastSeries([dict(stamp_ns=10, value=1), dict(stamp_ns=20, value=2)])
            self.assertEqual(series.past(15)['value'], 1)

        def test_actual_composition_includes_map_correction(self):
            correction, odometry = np.eye(4), np.eye(4)
            correction[:2, :2] = [[0., -1.], [1., 0.]]
            correction[:2, 3] = [10., 5.]
            odometry[0, 3] = 2.
            rows = composed_map_poses([dict(stamp_ns=5, matrix=correction)],
                                     [dict(stamp_ns=10, matrix=odometry)])
            self.assertEqual((rows[0]['x'], rows[0]['y']), (10., 7.))
            self.assertAlmostEqual(rows[0]['yaw'], math.pi/2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PureConversions))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--fps', type=float, default=5.)
    parser.add_argument('--map-frame', default='map')
    parser.add_argument('--odom-frame', default='odom')
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.report is None:
        parser.error('--report is required')
    if not math.isfinite(args.fps) or not .5 <= args.fps <= 30:
        parser.error('--fps must be 0.5..30')
    try:
        import rosbag
    except ImportError:
        if os.environ.get('NSO_EXPORT_SYSTEM_PYTHON') == '1':
            raise RuntimeError('ROS rosbag unavailable after sourcing /opt/ros/noetic/setup.bash')
        environment = dict(os.environ, NSO_EXPORT_SYSTEM_PYTHON='1')
        return subprocess.call(['/bin/bash', '-c',
            'source /opt/ros/noetic/setup.bash; exec /usr/bin/python3 -s "$@"',
            'nso-offline-export', str(Path(__file__).resolve())]+sys.argv[1:], env=environment)
    metadata = export_report(args.report.resolve(), args.fps, args.map_frame, args.odom_frame, args.base_frame)
    print(json.dumps(dict(report=metadata['report'], status=metadata['mission_status'],
        map_pose_samples=metadata['map_pose_samples'], camera_acquisitions=metadata['replay']['camera_acquisitions'],
        replay=metadata['replay']['path']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
