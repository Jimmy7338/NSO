#!/usr/bin/env python3
"""Replace this process with the selected native binary, preserving ROS argv."""
import os
import sys
if len(sys.argv) < 2 or not os.path.isabs(sys.argv[1]):
    raise SystemExit('native binary must be an absolute path')
os.execv(sys.argv[1], [sys.argv[1], *sys.argv[2:]])
