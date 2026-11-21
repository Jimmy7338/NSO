"""One passive live capture through production collectors; never starts an actuator."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

B = Path('/home/ouc/nso_robot')
sys.path.insert(0, str(B/'workspaces/nso_hybrid_20261029/runtime'))
import rospy
import tf2_ros
from can_control_feedback import CanVelocityFeedback
from reference_observation import ReferenceObservationCollector
from joint_observation import JointObservationCollector

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(exist_ok=False)
    cfg = json.loads((B/'analyses/sensor_repair_20261108/nov06_timing_trial/fresh/configuration.json').read_text())
    rospy.init_node('nso_npz_passive_validation', disable_signals=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener = tf2_ros.TransformListener(buf)
    can = CanVelocityFeedback(); can.start()
    ref = ReferenceObservationCollector(buf)
    joint = JointObservationCollector(buf, cfg['calibration']['initial_grid'],
        topic='/nso_pair/map', publisher='/nso_pair/gmapping', map_frame='nso_pair_map',
        apply_approved_self_mask=False)
    def snapshot():
        with can.lock:
            latest = asdict(can.samples[-1]) if can.samples else None
        return dict(actual_CAN_velocity=latest, health_reason=can.health_reason(),
                    source='receive-only kernel-timestamped CAN; no actuator')
    timings = {}
    try:
        deadline = time.monotonic()+3
        while snapshot()['actual_CAN_velocity'] is None and time.monotonic()<deadline:
            time.sleep(.02)
        before = snapshot()
        raw = Path(ref.capture_live_joint(out/'observed', lambda:None, snapshot, timings=timings))
        packet = Path(joint.capture(out/'joint_inputs', raw, source_role='passive_archive_codec_validation_not_experiment', timings=timings))
        after = snapshot()
        checks = []
        for path in [raw, packet]:
            with np.load(path, allow_pickle=False) as z:
                arrays = {k:z[k] for k in z.files}
            expected = {k:dict(dtype=v.dtype.str, shape=list(v.shape),
                sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
            with tempfile.TemporaryDirectory(prefix='nso_npz_validate_') as tmp:
                slow = Path(tmp)/'numpy_default.npz'
                np.savez_compressed(slow, **arrays)
                with np.load(slow, allow_pickle=False) as z:
                    got = {k:dict(dtype=z[k].dtype.str, shape=list(z[k].shape),
                        sha256=hashlib.sha256(z[k].tobytes()).hexdigest()) for k in z.files}
                assert got == expected
            receipt = json.loads(str(arrays['receipt_json']))
            checks.append(dict(path=str(path), bytes=path.stat().st_size,
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                arrays=expected, default_npz_roundtrip_exact=True,
                original_received_ROS_messages_preserved=all(k+'_received_ros_serialized' in arrays
                    for k in ['rgb','depth','camera_info','scan']),
                source_headers=receipt['headers']))
        (out/'receipt.json').write_text(json.dumps(dict(
            scope='Actual production collectors, one newly received passive observation; existing navigation map retained. Not a method trial or online motion speed comparison.',
            numpy_version=np.__version__, no_actuator_started=True, no_motion_commands=True,
            used_existing_map_epoch=True, timings=timings, CAN_before=before, CAN_after=after,
            packets=checks), indent=2)+'\n')
        print(json.dumps(dict(no_motion_commands=True, timings=timings,
            packets_checked=len(checks), default_npz_roundtrip_exact=True)))
    finally:
        ref.close(); joint.close(); can.close()
        rospy.signal_shutdown('passive capture validation finished')

if __name__ == '__main__':
    main()
