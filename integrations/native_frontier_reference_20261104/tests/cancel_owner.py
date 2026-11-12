#!/usr/bin/env python3
"""Fixture-only stand-in for owner cancel_all_goals; never publishes velocity."""
import actionlib, rospy
from move_base_msgs.msg import MoveBaseAction
rospy.init_node('fixture_supervisor')
client=actionlib.SimpleActionClient('/fixture/move_base',MoveBaseAction)
if not client.wait_for_server(rospy.Duration(5)): raise RuntimeError('fixture action server unavailable')
client.cancel_all_goals()
rospy.sleep(1.)
