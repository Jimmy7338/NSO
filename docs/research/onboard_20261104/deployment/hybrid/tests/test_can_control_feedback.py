"""Synthetic SocketCAN packets and clocks only. No ROS or CAN device access."""
import importlib.util
from pathlib import Path
import socket
import struct
import sys
import unittest

path = Path(__file__).resolve().parents[1] / 'runtime/can_control_feedback.py'
spec = importlib.util.spec_from_file_location('can_control_feedback', path)
can = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = can
spec.loader.exec_module(can)


def packet(ns, vx=-4, w=292, cid=0x221, dlc=8):
    raw = struct.pack('=IB3x8s', cid, dlc, struct.pack('>hh', vx, w) + b'\0'*4)
    anc = [(socket.SOL_SOCKET, can.SO_TIMESTAMPNS, struct.pack('@ll', ns//10**9, ns%10**9))]
    return raw, anc, 0


class Clock:
    def __init__(self): self.wall = 10_000_000_000; self.mono = 2_000_000_000
    def advance(self, ns): self.wall += ns; self.mono += ns


class FakeSocket:
    def __init__(self, *args): self.options=[]; self.closed=False
    def setsockopt(self,*args): self.options.append(args)
    def bind(self, address): self.address=address
    def settimeout(self,value): self.timeout=value
    def close(self): self.closed=True
    # Intentionally no send/sendto/sendmsg method.


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.reader = can.CanVelocityFeedback(socket_factory=FakeSocket,
            wall_ns=lambda:self.clock.wall, monotonic_ns=lambda:self.clock.mono)
    def tearDown(self): self.reader.close()

    def test_kernel_timestamp_and_signed_big_endian_decoding(self):
        sample = self.reader.ingest(*packet(self.clock.wall-23_456_789))
        self.assertEqual(sample.kernel_stamp_ns, 9_976_543_211)
        self.assertEqual(sample.receipt_monotonic_ns, self.clock.mono)
        self.assertEqual((sample.vx, sample.w), (-.004, .292))
        sample = self.reader.ingest(*packet(self.clock.wall, vx=-32768, w=32767))
        self.assertEqual((sample.vx,sample.w),(-32.768,32.767))

    def test_drops_truncated_error_extended_remote_wrong_id_length_and_missing_stamp(self):
        raw,anc,_=packet(self.clock.wall)
        cases=[(raw[:-1],anc,0),(raw,anc,socket.MSG_TRUNC),(raw,anc,socket.MSG_CTRUNC),
               (raw,[],0),(raw,anc+anc,0)]
        for cid in (0x211,0x221|0x20000000,0x221|0x40000000,0x221|0x80000000):
            cases.append(packet(self.clock.wall,cid=cid))
        cases.append(packet(self.clock.wall,dlc=7))
        cases.append((raw,[(socket.SOL_SOCKET,can.SO_TIMESTAMPNS,b'\0')],0))
        cases.append((raw,[(socket.SOL_SOCKET,can.SO_TIMESTAMPNS,struct.pack('@ll',10,10**9))],0))
        for args in cases:
            with self.subTest(args=args): self.assertIsNone(self.reader.ingest(*args))
        self.assertEqual(len(self.reader.samples),0)

    def test_source_age_and_future_kernel_stamp_are_not_renewed(self):
        self.assertIsNone(self.reader.ingest(*packet(self.clock.wall-150_000_001)))
        self.assertIsNone(self.reader.ingest(*packet(self.clock.wall+1)))
        s=self.reader.ingest(*packet(self.clock.wall-150_000_000))
        self.assertIsNotNone(s)
        self.clock.advance(1)
        self.assertIsNone(self.reader.for_scan(self.clock.wall))
        self.assertEqual(self.reader.health_reason(),'CAN velocity source expired')

    def test_selection_latest_not_after_scan_not_latest_arrival(self):
        for age,v in [(30_000_000,3),(10_000_000,1),(20_000_000,2)]:
            self.reader.ingest(*packet(self.clock.wall-age,vx=v))
        s=self.reader.for_scan(self.clock.wall-15_000_000)
        self.assertEqual(s.vx,.002)
        self.assertEqual(s.kernel_stamp_ns,self.clock.wall-20_000_000)
        self.assertEqual(self.reader.for_scan(self.clock.wall).vx,.001)
        self.assertIsNone(self.reader.for_scan(self.clock.wall-40_000_000))

    def test_scan_age_and_current_age_are_independently_required(self):
        stamp=self.clock.wall
        self.reader.ingest(*packet(stamp))
        self.assertIsNotNone(self.reader.for_scan(stamp+150_000_000))
        self.assertIsNone(self.reader.for_scan(stamp+150_000_001))
        self.clock.advance(150_000_001)
        # The old scan still matches, but the same sample is stale now.
        self.assertIsNone(self.reader.for_scan(stamp))

    def test_startup_deadline_health_loss_and_clock_steps(self):
        self.assertTrue(self.reader.in_startup_grace());self.assertIsNone(self.reader.health_reason())
        self.clock.advance(150_000_000)
        self.assertIsNone(self.reader.health_reason())
        self.clock.advance(1)
        self.assertFalse(self.reader.in_startup_grace())
        self.assertEqual(self.reader.health_reason(),'CAN startup source timeout')
        self.reader.ingest(*packet(self.clock.wall))
        self.assertFalse(self.reader.in_startup_grace())
        self.clock.wall+=20_000_001
        with self.assertRaises(ValueError):self.reader.ingest(*packet(self.clock.wall))
        with self.assertRaises(ValueError):self.reader.for_scan(self.clock.wall)
        self.assertEqual(self.reader.health_reason(),'CAN clock offset changed')

    def test_error_overflow_and_receive_exception_surface(self):
        raw,anc,flags=packet(self.clock.wall)
        with self.assertRaises(ValueError):
            self.reader.ingest(raw,anc+[(socket.SOL_SOCKET,can.SO_RXQ_OVFL,struct.pack('=I',1))],flags)
        self.reader.socket.recvmsg=lambda *args: (_ for _ in ()).throw(OSError('mock receive fault'))
        self.reader.receive()
        self.assertIn('mock receive fault',self.reader.health_reason())
        with self.assertRaises(ValueError):self.reader.for_scan(self.clock.wall)

    def test_receive_only_configuration_and_close(self):
        self.assertEqual(self.reader.socket.address,('can0',))
        self.assertIn((socket.SOL_SOCKET,can.SO_TIMESTAMPNS,1),self.reader.socket.options)
        # Regression: adding CAN_ERR_FLAG to this mask suppressed all live
        # normal data frames on Linux; errors use a separate socket filter.
        self.assertIn((socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,
                       struct.pack('=II',0x221,0xc00007ff)),self.reader.socket.options)
        self.assertFalse(hasattr(self.reader,'send'))
        self.assertFalse(hasattr(self.reader.socket,'send'))
        self.reader.close();self.assertTrue(self.reader.socket.closed)

    def test_cached_post_scan_samples_do_not_cancel_fixed_startup_grace(self):
        self.clock.advance(40_000_000)
        self.reader.ingest(*packet(self.clock.wall-20_000_000))
        self.assertIsNone(self.reader.for_scan(self.clock.wall-60_000_000))
        self.assertTrue(self.reader.in_startup_grace())
        self.clock.advance(110_000_001)
        self.assertFalse(self.reader.in_startup_grace())

    def test_pairing_receipt_preserves_kernel_stamp_raw_frame_and_both_clocks(self):
        raw,anc,flags=packet(self.clock.wall-30_000_000)
        original=self.reader.ingest(raw,anc,flags)
        self.clock.advance(10_000_000)
        selected=self.reader.for_scan(self.clock.wall-20_000_000)
        row=selected.provenance()
        self.assertEqual(row['kernel_stamp_ns'],original.kernel_stamp_ns)
        self.assertEqual(row['receipt_monotonic_ns'],original.receipt_monotonic_ns)
        self.assertEqual(row['raw_frame_hex'],raw.hex())
        self.assertEqual(row['scan_minus_can_ns'],20_000_000)
        self.assertEqual(row['current_minus_can_ns'],40_000_000)
        self.assertEqual(row['receipt_age_ns'],10_000_000)
        self.assertEqual(self.reader.source_receipt()['accepted_frames'],1)
        self.assertEqual(self.reader.source_receipt()['dropped_frames'],0)
        self.assertEqual(self.reader.samples[0],original)


if __name__=='__main__':unittest.main()
