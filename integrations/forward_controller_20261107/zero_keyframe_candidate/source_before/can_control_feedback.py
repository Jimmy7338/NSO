"""Receive-only SocketCAN chassis velocity; no command or transmit API.

Kernel realtime stamps are retained as integer nanoseconds. No interpolation,
filtering, zero substitution, or renewal of old timestamps is performed.
"""
from collections import deque
from dataclasses import dataclass, asdict
import socket
import struct
import threading
import time

VELOCITY_TTL_NS = 150_000_000  # Existing chassis velocity source TTL, 0.15 s.
SO_TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)
SO_RXQ_OVFL = getattr(socket, 'SO_RXQ_OVFL', 40)
# CAN_ERR_FLAG in can_mask selects the kernel error-frame filter list. It
# must NOT be added to this data filter; errors are separately subscribed and
# rejected by decode_velocity. Match the existing adapter's standard/RTR mask.
DATA_FILTER_MASK = 0xc00007ff


@dataclass(frozen=True)
class VelocitySample:
    kernel_stamp_ns: int
    receipt_monotonic_ns: int
    vx: float
    w: float
    raw_frame_hex: str = ''


@dataclass(frozen=True)
class VelocitySelection(VelocitySample):
    scan_stamp_ns: int = 0
    selected_wall_ns: int = 0
    selected_monotonic_ns: int = 0

    def provenance(self):
        row = asdict(self)
        row.update(scan_minus_can_ns=self.scan_stamp_ns-self.kernel_stamp_ns,
                   current_minus_can_ns=self.selected_wall_ns-self.kernel_stamp_ns,
                   receipt_age_ns=self.selected_monotonic_ns-self.receipt_monotonic_ns)
        return row


def decode_velocity(raw, ancillary, flags, now_ns, receipt_monotonic_ns):
    """Return an original 0x221 sample or None for unusable frames."""
    if len(raw) != 16 or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        return None
    stamps = []
    for level, kind, value in ancillary:
        if level != socket.SOL_SOCKET:
            continue
        if kind == SO_RXQ_OVFL:
            if len(value) < 4 or struct.unpack('=I', value[:4])[0]:
                raise ValueError('CAN receive overflow')
        if kind == SO_TIMESTAMPNS:
            size = struct.calcsize('@ll')
            if len(value) < size:
                return None
            sec, ns = struct.unpack('@ll', value[:size])
            if sec < 0 or not 0 <= ns < 1_000_000_000:
                return None
            stamps.append(sec * 1_000_000_000 + ns)
    if len(stamps) != 1:
        return None
    cid, dlc, payload = struct.unpack('=IB3x8s', raw)
    if cid & 0xe0000000 or cid != 0x221 or dlc != 8:
        return None
    stamp = stamps[0]
    if not 0 <= now_ns - stamp <= VELOCITY_TTL_NS:
        return None
    vx, w = struct.unpack('>hh', payload[:4])
    return VelocitySample(stamp, receipt_monotonic_ns, vx / 1000., w / 1000., raw.hex())


class CanVelocityFeedback:
    def __init__(self, interface='can0', socket_factory=socket.socket,
                 wall_ns=time.time_ns, monotonic_ns=time.monotonic_ns):
        self.wall_ns, self.monotonic_ns = wall_ns, monotonic_ns
        self.lock = threading.Lock()
        self.samples = deque(maxlen=256)
        self.error = None
        self.interface = interface
        self.received_frames = self.accepted_frames = self.dropped_frames = 0
        self.done = threading.Event()
        self.started_monotonic_ns = monotonic_ns()
        self.clock_offset_ns = wall_ns() - self.started_monotonic_ns
        self.socket = socket_factory(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                                   struct.pack('=II', 0x221, DATA_FILTER_MASK))
            self.socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER,
                                   struct.pack('=I', 0x1fffffff))
            self.socket.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
            self.socket.bind((interface,))
            self.socket.settimeout(.01)
        except BaseException:
            self.socket.close()
            raise
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def ingest(self, raw, ancillary, flags):
        now, mono = self.wall_ns(), self.monotonic_ns()
        # Match the existing receiver's 20 ms clock-step detection; never
        # reinterpret a realtime jump as fresh source data.
        if abs(now - mono - self.clock_offset_ns) > 20_000_000:
            raise ValueError('CAN clock offset changed')
        sample = decode_velocity(raw, ancillary, flags, now, mono)
        with self.lock:
            self.received_frames += 1
            if sample is not None:
                self.accepted_frames += 1
                self.samples.append(sample)
            else:
                self.dropped_frames += 1
        return sample

    def receive(self):
        try:
            while not self.done.is_set():
                try:
                    raw, anc, flags, _ = self.socket.recvmsg(16, 128)
                except socket.timeout:
                    continue
                self.ingest(raw, anc, flags)
        except Exception as exc:
            if not self.done.is_set():
                with self.lock:
                    self.error = type(exc).__name__ + ': ' + str(exc)

    def health_reason(self):
        now, mono = self.wall_ns(), self.monotonic_ns()
        with self.lock:
            if self.error:
                return 'CAN receive fault: ' + self.error
            if abs(now - mono - self.clock_offset_ns) > 20_000_000:
                return 'CAN clock offset changed'
            if not self.samples:
                if mono - self.started_monotonic_ns <= VELOCITY_TTL_NS:
                    return None
                return 'CAN startup source timeout'
            latest = max(self.samples, key=lambda s: s.kernel_stamp_ns)
            if (not 0 <= now - latest.kernel_stamp_ns <= VELOCITY_TTL_NS or
                    not 0 <= mono - latest.receipt_monotonic_ns <= VELOCITY_TTL_NS):
                return 'CAN velocity source expired'
        return None

    def for_scan(self, scan_stamp_ns):
        """Latest measured sample no later than scan, fresh at scan and now."""
        now, mono = self.wall_ns(), self.monotonic_ns()
        with self.lock:
            if self.error:
                raise ValueError('CAN receive fault: ' + self.error)
            if abs(now - mono - self.clock_offset_ns) > 20_000_000:
                raise ValueError('CAN clock offset changed')
            eligible = [s for s in self.samples
                        if 0 <= scan_stamp_ns - s.kernel_stamp_ns <= VELOCITY_TTL_NS
                        and 0 <= now - s.kernel_stamp_ns <= VELOCITY_TTL_NS
                        and 0 <= mono - s.receipt_monotonic_ns <= VELOCITY_TTL_NS]
            if eligible:
                sample = max(eligible, key=lambda s: s.kernel_stamp_ns)
                return VelocitySelection(**asdict(sample), scan_stamp_ns=scan_stamp_ns,
                                         selected_wall_ns=now, selected_monotonic_ns=mono)
        return None

    def in_startup_grace(self):
        with self.lock:
            return (not self.error and
                    self.monotonic_ns() - self.started_monotonic_ns <= VELOCITY_TTL_NS)

    def source_receipt(self):
        with self.lock:
            return dict(can_interface=self.interface, can_id='0x221',
                        data_filter_mask=hex(DATA_FILTER_MASK), timestamp_socket_option=SO_TIMESTAMPNS,
                        source_timestamp_semantics='original SocketCAN SO_TIMESTAMPNS kernel realtime nanoseconds',
                        source_ttl_ns=VELOCITY_TTL_NS, received_frames=self.received_frames,
                        accepted_frames=self.accepted_frames, dropped_frames=self.dropped_frames,
                        reader_error=self.error, started_monotonic_ns=self.started_monotonic_ns,
                        realtime_monotonic_offset_ns=self.clock_offset_ns)

    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=.5)
        self.socket.close()
