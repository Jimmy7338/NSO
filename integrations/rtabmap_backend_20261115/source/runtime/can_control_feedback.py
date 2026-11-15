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
CLOCK_STEP_NS = 20_000_000  # Original strict >20 ms boundary.
CLOCK_SAMPLE_MAX_WIDTH_NS = 100_000
CLOCK_SAMPLE_ATTEMPTS = 8
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
        self.first_health_failure = None
        self.interface = interface
        self.received_frames = self.accepted_frames = self.dropped_frames = 0
        self.done = threading.Event()
        initial = self._clock_sample(initial=True)
        self.started_monotonic_ns = initial['monotonic_after_ns']
        self.clock_offset_interval_ns = (initial['offset_low_ns'],initial['offset_high_ns'])
        self.clock_offset_ns = sum(self.clock_offset_interval_ns)//2
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

    def _clock_sample(self, initial=False):
        """Bracket realtime with monotonic; retry uncertain measurements only.

        No baseline renewal. A threshold-straddling interval is never accepted.
        Persistent measurement uncertainty fails separately from a proved step.
        """
        for _ in range(CLOCK_SAMPLE_ATTEMPTS):
            before = self.monotonic_ns()
            wall = self.wall_ns()
            after = self.monotonic_ns()
            row = dict(monotonic_before_ns=before,wall_ns=wall,
                       monotonic_after_ns=after,measurement_width_ns=after-before,
                       offset_low_ns=wall-after,offset_high_ns=wall-before)
            self.last_clock_sample = row
            if not 0 <= after-before <= CLOCK_SAMPLE_MAX_WIDTH_NS:
                continue
            if initial:
                return row
            low,high = self.clock_offset_interval_ns
            delta_low,delta_high = wall-after-high,wall-before-low
            row.update(offset_delta_low_ns=delta_low,offset_delta_high_ns=delta_high)
            if delta_low > CLOCK_STEP_NS or delta_high < -CLOCK_STEP_NS:
                raise ValueError('CAN clock offset changed: '+str(row))
            if delta_low >= -CLOCK_STEP_NS and delta_high <= CLOCK_STEP_NS:
                return row
        raise ValueError('CAN clock sample uncertain: '+str(self.last_clock_sample))

    def start(self):
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def ingest(self, raw, ancillary, flags):
        clock = self._clock_sample()
        now,mono = clock['wall_ns'],clock['monotonic_after_ns']
        sample = decode_velocity(raw, ancillary, flags, now, mono)
        # Conservative age upper bound: sampling uncertainty cannot renew TTL.
        if sample is not None and now+clock['measurement_width_ns']-sample.kernel_stamp_ns > VELOCITY_TTL_NS:
            sample = None
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

    def _health_failure(self, reason, clock=None):
        """Caller holds lock; freeze evidence at the first failed health check."""
        if self.first_health_failure is None:
            latest=max(self.samples,key=lambda s:s.kernel_stamp_ns) if self.samples else None
            wall=None if clock is None else clock['wall_ns']
            mono=None if clock is None else clock['monotonic_after_ns']
            self.first_health_failure=dict(reason=reason,clock=None if clock is None else dict(clock),
                latest_sample=None if latest is None else asdict(latest),
                source_age_ns=None if wall is None or latest is None else wall-latest.kernel_stamp_ns,
                receipt_age_ns=None if mono is None or latest is None else mono-latest.receipt_monotonic_ns,
                source_ttl_ns=VELOCITY_TTL_NS,reader_error=self.error,
                received_frames=self.received_frames,accepted_frames=self.accepted_frames,
                dropped_frames=self.dropped_frames)
        return reason

    def health_reason(self):
        # Clock and selected sample must describe one locked receiver snapshot.
        # Otherwise a new sample can appear newer than the earlier clock read.
        with self.lock:
            if self.error:
                return self._health_failure('CAN receive fault: ' + self.error)
            try:
                clock = self._clock_sample()
            except ValueError as exc:
                return self._health_failure(str(exc))
            now,mono = clock['wall_ns'],clock['monotonic_after_ns']
            if not self.samples:
                if mono - self.started_monotonic_ns <= VELOCITY_TTL_NS:
                    return None
                return self._health_failure('CAN startup source timeout',clock)
            latest = max(self.samples, key=lambda s: s.kernel_stamp_ns)
            if (not (0 <= now-latest.kernel_stamp_ns and now+clock['measurement_width_ns']-latest.kernel_stamp_ns <= VELOCITY_TTL_NS) or
                    not 0 <= mono - latest.receipt_monotonic_ns <= VELOCITY_TTL_NS):
                return self._health_failure('CAN velocity source expired',clock)
        return None

    def for_scan(self, scan_stamp_ns):
        """Latest measured sample no later than scan, fresh at scan and now."""
        with self.lock:
            if self.error:
                raise ValueError('CAN receive fault: ' + self.error)
            clock = self._clock_sample()
            now,mono = clock['wall_ns'],clock['monotonic_after_ns']
            eligible = [s for s in self.samples
                        if 0 <= scan_stamp_ns - s.kernel_stamp_ns <= VELOCITY_TTL_NS
                        and 0 <= now-s.kernel_stamp_ns and now+clock['measurement_width_ns']-s.kernel_stamp_ns <= VELOCITY_TTL_NS
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
                        first_health_failure=self.first_health_failure,
                        reader_error=self.error, started_monotonic_ns=self.started_monotonic_ns,
                        realtime_monotonic_offset_ns=self.clock_offset_ns,
                        clock_profile='bracketed_clock_interval_v1',clock_step_threshold_ns=CLOCK_STEP_NS,
                        clock_step_boundary='strict_greater_than',
                        initial_offset_interval_ns=list(self.clock_offset_interval_ns),
                        latest_clock_sample=dict(self.last_clock_sample),
                        clock_sample_max_width_ns=CLOCK_SAMPLE_MAX_WIDTH_NS)

    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=.5)
        self.socket.close()
