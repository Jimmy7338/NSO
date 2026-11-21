"""Run isolated ROS native diagnostics with execute=false; SDK is not opened."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import rospy
from std_msgs.msg import Int32

D = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--validation', type=Path, required=True)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(exist_ok=False)
binary = args.binary.resolve()
expected = json.loads(args.validation.read_text())['candidate_binary_sha256']
assert hashlib.sha256(binary.read_bytes()).hexdigest() == expected
assert os.environ['ROS_MASTER_URI'] == 'http://127.0.0.1:11382'
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
rospy.init_node('nso_native_capture_validation', anonymous=True, disable_signals=True, disable_rosout=True)
results = []
for name, output, induce in [('late', str(OUT / 'late_timing.json'), True),
                             ('bad_output', '/proc/nso_native_no_such_directory/timing.json', False)]:
    prefix = '/nso_native_capture_' + name
    argv = [str(binary), '_execute:=false', '_session_id:=native_capture_' + name,
            '_expected_boot_id:=' + boot, '_can_interface:=can0',
            '_guard_node:=' + prefix + '_guard', '_timing_output:=' + output,
            '__name:=nso_native_capture_' + name,
            '/nso_radius/guarded_cmd:=' + prefix + '/guarded_cmd',
            '/nso_radius/stop:=' + prefix + '/stop',
            '/nso_radius/driver_fault:=' + prefix + '/driver_fault']
    row = dict(name=name, argv=argv, execute=False, SDK_opened=False,
               robot_motion_commands=False, topics_isolated=True)
    with (OUT / (name + '.log')).open('xb') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        row['pid'] = child.pid
        try:
            ready = rospy.wait_for_message(prefix + '/driver_fault', Int32, timeout=10)
            assert ready.data == 0, ('unexpected startup fault', ready.data)
            row['first_driver_fault'] = ready.data
            time.sleep(.7)
            if induce:
                row['SIGSTOP_monotonic_ns'] = time.monotonic_ns()
                os.kill(child.pid, signal.SIGSTOP)
                time.sleep(.1)
                os.kill(child.pid, signal.SIGCONT)
                row['SIGCONT_monotonic_ns'] = time.monotonic_ns()
            time.sleep(.5)
            os.kill(child.pid, signal.SIGINT)
            row['exit_code'] = child.wait(timeout=10)
        finally:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGCONT)
                os.kill(child.pid, signal.SIGINT)
                child.wait(timeout=10)
    assert row['exit_code'] == 0
    if induce:
        data = json.loads(Path(output).read_text())
        first = data['first_fault']
        assert first['fault_after_sample'] == 5, first
        gap = first['gate_now_ns'] - first['gate_prev_tick_ns']
        assert gap > 40000000, gap
        row.update(first_fault=first, gate_gap_ms=gap / 1e6,
                   retained_rows=len(data['rows']), output_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest())
    else:
        row['failed_output_open_did_not_change_clean_exit'] = True
    results.append(row)
receipt = dict(scope='Actual ROS capture-mode integration; no SDK, CAN socket, or actuation in this native mode',
               candidate_binary_sha256=expected, results=results,
               limitations=['Deliberately injected scheduling pause is not the cause of B1_C inferred from data.',
                            'No SDK timing, hardware zero transmission, or motion validation performed.'])
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
