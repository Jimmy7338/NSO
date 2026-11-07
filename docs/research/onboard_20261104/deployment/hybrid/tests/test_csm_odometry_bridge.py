"""Offline contract tests: synthetic messages only, no ROS master or hardware."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from xml.etree import ElementTree


def vec(**kw):
    return types.SimpleNamespace(**kw)


def message():
    return vec(header=vec(stamp=None, frame_id=''), child_frame_id='',
               pose=vec(pose=None, covariance=[0.] * 36), twist=vec(twist=vec(linear=vec(x=0., y=0.), angular=vec(z=0.)), covariance=[0.] * 36))


def transform():
    return vec(header=vec(stamp=None, frame_id=''), child_frame_id='',
               transform=vec(translation=vec(x=0., y=0., z=0.), rotation=None))


class Sink:
    def __init__(self, *args, **kwargs): self.items = []
    def publish(self, item): self.items.append(item)
    def sendTransform(self, item): self.items.append(item)


rospy = types.ModuleType('rospy')
rospy.Publisher = Sink
rospy.Subscriber = lambda *a, **kw: None
rospy.logwarn_throttle = lambda *a: None
rospy.logerr = lambda *a: None
rospy.signal_shutdown = lambda *a: None
rospy.set_param = lambda *a: None
tf = types.ModuleType('tf2_ros')
tf.TransformBroadcaster = Sink
geom = types.ModuleType('geometry_msgs.msg')
geom.PoseStamped = object
geom.TransformStamped = transform
nav = types.ModuleType('nav_msgs.msg')
nav.Odometry = message
std = types.ModuleType('std_msgs.msg')
std.String = lambda data='': vec(data=data)
modules = {'rospy': rospy, 'tf2_ros': tf, 'geometry_msgs': types.ModuleType('geometry_msgs'),
           'geometry_msgs.msg': geom, 'nav_msgs': types.ModuleType('nav_msgs'), 'nav_msgs.msg': nav,
           'std_msgs': types.ModuleType('std_msgs'), 'std_msgs.msg': std}
path = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('csm_odometry_bridge', path / 'runtime/csm_odometry_bridge.py')
bridge = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, modules):
    spec.loader.exec_module(bridge)


class Stamp:
    def __init__(self, ns): self.ns = ns
    def to_nsec(self): return self.ns
    def __eq__(self, other): return isinstance(other, Stamp) and self.ns == other.ns


def fake_feedback(vx, w):
    return vec(vx=vx, w=w, provenance=lambda: dict(kernel_stamp_ns=1_090_000_000,
        receipt_monotonic_ns=2_000_000_000, scan_stamp_ns=1_100_000_000,
        selected_wall_ns=1_120_000_000, selected_monotonic_ns=2_020_000_000,
        scan_minus_can_ns=10_000_000, current_minus_can_ns=30_000_000))


def pose(ns, x=0., y=0., yaw=0., z=0.):
    return vec(_connection_header={'callerid': '/nso_csm/matcher'},
               header=vec(stamp=Stamp(ns), frame_id='odom'),
               pose=vec(position=vec(x=x, y=y, z=z),
                        orientation=vec(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2))))


class BridgeTest(unittest.TestCase):
    def assertTwist(self, actual, expected):
        for a, e in zip(actual, expected): self.assertAlmostEqual(a, e, places=9)

    def test_se2_constant_body_velocity_and_world_rotation(self):
        for vx, vy, w, dt in [(0.3, 0, 0, 2), (-0.2, 0, 0, 1),
                              (0, 0, 0.7, 0.5), (0.4, -0.2, 0.7, 1.3),
                              (0.1, 0.3, -0.8, 0.8)]:
            for heading in (0., 1.2, 3.0):
                angle = w * dt
                if w:
                    dx = (math.sin(angle)*vx - (1-math.cos(angle))*vy)/w
                    dy = ((1-math.cos(angle))*vx + math.sin(angle)*vy)/w
                else:
                    dx, dy = vx*dt, vy*dt
                end = (4 + math.cos(heading)*dx-math.sin(heading)*dy,
                       -2 + math.sin(heading)*dx+math.cos(heading)*dy,
                       math.atan2(math.sin(heading+angle), math.cos(heading+angle)))
                with self.subTest(vx=vx, vy=vy, w=w, heading=heading):
                    self.assertTwist(bridge.derived_twist((4, -2, heading), end, dt), (vx, vy, w))

    def test_first_invalid_and_nonincreasing_frames_do_not_publish_or_replace_previous(self):
        b = bridge.CsmOdometryBridge()
        b.callback(pose(1_000_000_000))
        self.assertEqual((len(b.pub.items), len(b.tf.items)), (0, 0))
        invalid = pose(2_000_000_000, x=float('nan'))
        for msg in (pose(1_000_000_000, x=9), pose(500_000_000, x=9), invalid, pose(0, x=9)):
            b.callback(msg)
        self.assertEqual((len(b.pub.items), len(b.tf.items)), (0, 0))
        b.callback(pose(3_000_000_000, x=2))
        self.assertTwist((b.pub.items[0].twist.twist.linear.x, b.pub.items[0].twist.twist.linear.y,
                          b.pub.items[0].twist.twist.angular.z), (1, 0, 0))

    def test_output_preserves_input_pose_stamp_and_does_not_mutate(self):
        b = bridge.CsmOdometryBridge()
        b.callback(pose(1_000_000_000, yaw=2.9))
        msg = pose(2_000_000_000, x=.7, y=-.4, yaw=-3.0, z=.2)
        original = copy.deepcopy(msg)
        b.callback(msg)
        self.assertEqual(msg, original)
        out, tfout = b.pub.items[0], b.tf.items[0]
        self.assertEqual(out.pose.pose, original.pose)
        self.assertEqual(out.header.stamp, original.header.stamp)
        self.assertEqual(tfout.header.stamp, original.header.stamp)
        self.assertEqual(tfout.transform.rotation, original.pose.orientation)
        self.assertEqual((tfout.transform.translation.x, tfout.transform.translation.y, tfout.transform.translation.z), (.7, -.4, .2))
        self.assertEqual((out.header.frame_id, out.child_frame_id, tfout.header.frame_id, tfout.child_frame_id),
                         ('odom', 'base_link', 'odom', 'base_link'))
        self.assertAlmostEqual(out.twist.twist.angular.z, 2*math.pi-5.9)

    def test_unexpected_caller_frame_and_zero_quaternion_rejected(self):
        b = bridge.CsmOdometryBridge()
        b.callback(pose(1_000_000_000))
        wrong_caller = pose(2_000_000_000, x=10)
        wrong_caller._connection_header['callerid'] = '/other'
        wrong_frame = pose(2_000_000_000, x=10)
        wrong_frame.header.frame_id = 'laser'
        wrong_quaternion = pose(2_000_000_000, x=10)
        wrong_quaternion.pose.orientation.w = 0
        for msg in (wrong_caller, wrong_frame, wrong_quaternion): b.callback(msg)
        b.callback(pose(3_000_000_000, x=2))
        self.assertEqual(len(b.pub.items), 1)
        self.assertAlmostEqual(b.pub.items[0].twist.twist.linear.x, 1)

    def test_control_projection_preserves_raw_and_copies_all_other_fields(self):
        b = bridge.CsmOdometryBridge()
        b.callback(pose(1_000_000_000, yaw=.4))
        msg = pose(2_000_000_000, x=.3, y=.6, yaw=.7, z=.2)
        original = copy.deepcopy(msg)
        b.callback(msg)
        raw, control, tfout = b.pub.items[0], b.control_pub.items[0], b.tf.items[0]
        self.assertEqual(msg, original)
        self.assertNotAlmostEqual(raw.twist.twist.linear.y, 0.)
        self.assertEqual(control.twist.twist.linear.y, 0.)
        self.assertEqual(raw.header, control.header)
        self.assertEqual(raw.child_frame_id, control.child_frame_id)
        self.assertEqual(raw.pose, control.pose)
        self.assertEqual(raw.twist.covariance, control.twist.covariance)
        self.assertEqual(raw.twist.twist.linear.x, control.twist.twist.linear.x)
        self.assertEqual(raw.twist.twist.angular.z, control.twist.twist.angular.z)
        self.assertEqual(tfout.header.stamp, original.header.stamp)
        self.assertEqual(tfout.transform.rotation, original.pose.orientation)
        self.assertEqual((tfout.transform.translation.x, tfout.transform.translation.y, tfout.transform.translation.z), (.3, .6, .2))
        self.assertIsNot(raw, control)
        self.assertIsNot(raw.pose, control.pose)
        self.assertIsNot(raw.pose.pose, control.pose.pose)
        self.assertIsNot(raw.twist, control.twist)
        self.assertIsNot(raw.twist.covariance, control.twist.covariance)
        control.pose.pose.position.x = 99
        control.twist.covariance[0] = 99
        self.assertEqual(raw.pose.pose.position.x, .3)
        self.assertEqual(raw.twist.covariance[0], 0.)
        self.assertEqual(msg, original)

    def test_quaternion_norm_validation_preserves_accepted_raw_quaternion(self):
        b = bridge.CsmOdometryBridge()
        b.callback(pose(1_000_000_000))
        invalid = pose(2_000_000_000)
        invalid.pose.orientation.w = 1.01
        b.callback(invalid)
        self.assertEqual(len(b.pub.items), 0)
        accepted = pose(3_000_000_000, x=2)
        accepted.pose.orientation.w = 1.0001
        b.callback(accepted)
        self.assertEqual(b.pub.items[0].pose.pose.orientation.w, 1.0001)
        self.assertEqual(b.control_pub.items[0].pose.pose.orientation.w, 1.0001)
        self.assertEqual(b.tf.items[0].transform.rotation.w, 1.0001)

    def test_invalid_delta_time_rejected(self):
        for dt in (0., -1., float('nan'), float('inf'), -float('inf')):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                bridge.derived_twist((0, 0, 0), (1, 0, 0), dt)

    def test_launch_selects_base_and_matcher_does_not_publish_tf(self):
        root = ElementTree.parse(path / 'config/csm_primary.launch').getroot()
        matcher = root.find(".//node[@name='matcher']")
        params = {p.attrib['name']: p.attrib['value'] for p in matcher.findall('param')}
        self.assertEqual(params['base_frame'], 'base_link')
        self.assertEqual(params['fixed_frame'], 'odom')
        self.assertEqual(params['publish_tf'], 'false')
        self.assertEqual(params['publish_pose_stamped'], 'true')
        self.assertEqual(len(root.findall(".//node[@name='nso_csm_odometry_bridge']")), 0)

    def test_can_opt_in_changes_only_control_vx_w_preserving_raw_and_tf(self):
        source = vec(for_scan=lambda ns: fake_feedback(vx=-.004, w=.292),
                     in_startup_grace=lambda: False, health_reason=lambda: None, source_receipt=lambda: {'can_interface':'mock0'})
        original, adapted = bridge.CsmOdometryBridge(), bridge.CsmOdometryBridge('can_feedback', source)
        for ns, x, y, yaw in [(1_000_000_000, 0., 0., .4), (1_100_000_000, .04, .01, .43)]:
            msg = pose(ns, x=x, y=y, yaw=yaw, z=.2)
            saved = copy.deepcopy(msg)
            original.callback(msg); adapted.callback(msg)
            self.assertEqual(msg, saved)
        self.assertEqual(original.pub.items, adapted.pub.items)
        self.assertEqual(original.tf.items, adapted.tf.items)
        expected = copy.deepcopy(original.control_pub.items[0])
        expected.twist.twist.linear.x = -.004
        expected.twist.twist.angular.z = .292
        self.assertEqual(expected, adapted.control_pub.items[0])
        self.assertEqual(adapted.control_pub.items[0].twist.twist.linear.y, 0.)
        self.assertNotEqual(adapted.pub.items[0].twist.twist.linear.x, -.004)
        import json
        provenance=json.loads(adapted.provenance_pub.items[0].data)
        self.assertEqual(provenance['csm_source_stamp_ns'],1_100_000_000)
        self.assertEqual(provenance['can_pair']['kernel_stamp_ns'],1_090_000_000)
        self.assertEqual(provenance['can_pair']['receipt_monotonic_ns'],2_000_000_000)
        self.assertEqual(provenance['control_twist'],{'vx':-.004,'vy':0.,'w':.292})
        self.assertEqual(provenance['raw_csm_twist']['vx'],adapted.pub.items[0].twist.twist.linear.x)
        self.assertEqual(provenance['raw_csm_twist']['vy'],adapted.pub.items[0].twist.twist.linear.y)
        self.assertEqual(provenance['raw_csm_twist']['w'],adapted.pub.items[0].twist.twist.angular.z)
        self.assertIsNone(original.provenance_pub)

    def test_can_missing_fails_all_publications_and_latches_without_fallback(self):
        source = vec(for_scan=lambda ns: None, in_startup_grace=lambda: False,
                     health_reason=lambda: None, source_receipt=lambda: {'can_interface':'mock0'})
        b = bridge.CsmOdometryBridge('can_feedback', source)
        b.callback(pose(1_000_000_000))
        with patch.object(rospy, 'signal_shutdown') as stop:
            b.callback(pose(1_100_000_000, x=.04))
            source.for_scan = lambda ns: fake_feedback(vx=.01, w=0.)
            b.callback(pose(1_200_000_000, x=.05))
        self.assertEqual((b.pub.items, b.control_pub.items, b.tf.items), ([], [], []))
        self.assertIsNotNone(b.failed)
        stop.assert_called_once()

    def test_can_startup_grace_withholds_all_output_until_first_valid_sample(self):
        source = vec(for_scan=lambda ns: None, in_startup_grace=lambda: True,
                     health_reason=lambda: None, source_receipt=lambda: {'can_interface':'mock0'})
        b = bridge.CsmOdometryBridge('can_feedback', source)
        b.callback(pose(1_000_000_000))
        b.callback(pose(1_050_000_000))
        self.assertIsNone(b.failed)
        self.assertEqual(b.pub.items, [])
        source.for_scan = lambda ns: fake_feedback(vx=.006, w=-.01)
        b.callback(pose(1_100_000_000))
        self.assertEqual(len(b.pub.items), 1)
        # A subsequent loss cannot re-enter startup grace.
        source.for_scan = lambda ns: None
        b.callback(pose(1_150_000_000))
        self.assertIsNotNone(b.failed)
        self.assertEqual(len(b.pub.items), 1)

    def test_can_watchdog_stops_without_any_csm_callback_and_reader_fault_stops(self):
        for reason in ('CAN startup source timeout', 'CAN velocity source expired', 'CAN receive fault: OSError'):
            source = vec(health_reason=lambda: reason)
            b = bridge.CsmOdometryBridge('can_feedback', source)
            with patch.object(rospy, 'signal_shutdown') as stop:
                b.check_can_health()
            self.assertEqual(b.failed, reason)
            stop.assert_called_once_with(reason)
            self.assertEqual(b.pub.items, [])

    def test_can_selection_exception_fails_without_publishing(self):
        def bad(ns): raise ValueError('CAN clock offset changed')
        source = vec(for_scan=bad)
        b = bridge.CsmOdometryBridge('can_feedback', source)
        b.callback(pose(1_000_000_000))
        b.callback(pose(1_100_000_000))
        self.assertEqual(b.failed, 'CAN clock offset changed')
        self.assertEqual((b.pub.items, b.control_pub.items, b.tf.items), ([], [], []))

    def test_can_requires_explicit_valid_mode_and_injected_source(self):
        with self.assertRaises(ValueError): bridge.CsmOdometryBridge('other')
        with self.assertRaises(ValueError): bridge.CsmOdometryBridge('can_feedback')
        # Supplying a source alone must not enable it in the default mode.
        b = bridge.CsmOdometryBridge(can_feedback=vec())
        b.callback(pose(1_000_000_000)); b.callback(pose(1_100_000_000, x=.01))
        self.assertAlmostEqual(b.control_pub.items[0].twist.twist.linear.x, .1)

    def test_main_default_never_constructs_can_reader(self):
        module = types.ModuleType('can_control_feedback')
        module.CanVelocityFeedback = lambda *args: self.fail('default opened CAN source')
        with patch.dict(sys.modules, {'can_control_feedback': module}), \
                patch.object(rospy, 'myargv', return_value=['bridge.py'], create=True), \
                patch.object(rospy, 'init_node', create=True), \
                patch.object(rospy, 'spin', create=True) as spin, \
                patch.object(rospy, 'set_param') as profile:
            self.assertEqual(bridge.main(), 0)
        spin.assert_called_once()
        profile.assert_called_once_with('~control_velocity_source','csm_se2')

    def test_main_opt_in_watchdog_failure_exits_nonzero_and_closes_source(self):
        from unittest.mock import Mock
        source = Mock()
        source.health_reason.return_value = 'CAN velocity source expired'
        module = types.ModuleType('can_control_feedback')
        module.CanVelocityFeedback = Mock(return_value=source)
        with patch.dict(sys.modules, {'can_control_feedback': module}), \
                patch.object(rospy, 'myargv', return_value=['bridge.py', '--control-velocity-source', 'can_feedback'], create=True), \
                patch.object(rospy, 'init_node', create=True), \
                patch.object(rospy, 'is_shutdown', side_effect=[False, True], create=True), \
                patch.object(rospy, 'signal_shutdown') as stop, \
                patch.object(bridge.time, 'sleep'), \
                patch.object(rospy, 'set_param') as profile:
            self.assertEqual(bridge.main(), 2)
        module.CanVelocityFeedback.assert_called_once_with('can0')
        source.start.assert_called_once(); source.close.assert_called_once()
        stop.assert_called_once_with('CAN velocity source expired')
        profile.assert_called_once_with('~control_velocity_source','can_feedback')


if __name__ == '__main__': unittest.main()
