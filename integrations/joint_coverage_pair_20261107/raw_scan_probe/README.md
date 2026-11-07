# One bounded passive raw-lidar diagnostic

No new real data have been acquired. `permission_receipt.json` records the earlier
permission block; `synthetic_fixture/` is entirely artificial and is not evidence
about the robot or the isolated S32 return.

Run **once in a local robot terminal as the desktop user**, not as root Python:

```bash
ROS_MASTER_URI=http://127.0.0.1:11382 PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages /usr/bin/python3 /home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/raw_scan_probe/capture.py --sudo --output /home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/raw_scan_probe/capture_01
```

`sudo -v` prompts only in that terminal; Python never reads or stores a password.
Only `timeout`/`tcpdump` are elevated, and tcpdump drops to the desktop user with
`-Z` for writing. The command neither changes permissions nor installs anything.
An existing output directory is refused. A permission/error outcome is retained;
there is no retry, second window, rosbag, driver restart or motion command.

The packet command is fixed: non-promiscuous `eth0`, inbound only, source
`192.168.1.222`, UDP destination port `2368`, full packets, host kernel PCAP
timestamps at nanosecond file precision, at most 2000 packets. A root-owned GNU
timeout allows 7.5 seconds plus 0.3 seconds forced-cleanup grace, below 8 seconds.
This uses passive link capture, never a second UDP bind on port2368. Native N301
1.7/profile parameters and pinned decoder library are checked before capture.

ROS collection is independently bounded to90 messages per source and only armed
once tcpdump reports ready. It saves original `/scan` ranges/intensities and
header/source/receipt times, `/lslidar_point_cloud` point count, exact canonical
XYZI/data hashes and8 nearest points (no full cloud stream), and `/time_topic`
Float64 source values and receipt times. No source time is replaced. Readiness,
packet count/span/filter contents, PCAP completeness, overlapping host/ROS receipt
windows, source authorities, callback errors and overflow are checked. Ctrl-C or
SIGTERM stops only the owned capture supervisor; partial buffers and the receipt
are retained for review. No success is reported for an empty or invalid window.

Successful output includes `lidar.pcap`, `scans.npz`, `ros_window.json`,
`tcpdump.log`, `capture_receipt.json`, and automatically generated
`comparison.json`. If cancelled, the receipt identifies the incomplete state.
Do not repeat a failed/empty window without a separate decision.

## Offline review

```bash
/usr/bin/python3 /home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/raw_scan_probe/analyze.py /home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/raw_scan_probe/capture_01
```

`build_oracle.py` builds only a small offline library from exact current vendor
`decodePacketN301_1_7`, `isPointValid`, and `pointcloudToLaserscan` bodies, with
minimal non-ROS types. Vendor source/header hashes and function-body hashes are
recorded. It retains the actual `.017453f` angle constant, float geometry,
0.004m distance resolution, metadata-slot skip, original valid ranges and
minimum-range rebin; it adds no smoothing or rejection policy. No SDK/node is run.

`analyze.py` decodes the retained Ethernet/IPv4 UDP bytes, groups complete turns
using the original >35000-centidegree wrap, discards boundary partial turns, then
compares the complete native2000-bin ranges/intensities and cloud XYZI hashes.
It reports all possible geometry matches and same-geometry scan/cloud stamp
offsets, rather than pairing solely by header time. A 1µs Float64-representation
association for `time_topic` is explicitly not per-beam timing accuracy. Raw
host PCAP stamps are retained; native PCAP-driver replay's `now()` is never used.
Malformed packets and multiple wraps within a packet are disclosed, not silently
treated as evidence of a publication race.

A raw .504m value and a sweep/header mispair can coexist. Repeated geometry can
make matching ambiguous. Packet loss, boundary turns and receipt delays can
produce unmatched scans. If the isolated return does not recur, neither proposed
cause is excluded. No physical object, self-hit, noise or driver-race attribution
follows merely from a successful tool check.

## Synthetic checks (no ROS/network access)

```bash
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -m unittest discover -s /home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/raw_scan_probe -p test_probe.py -v
```

The fixture has18 artificial packets, two complete retained turns, one .504m raw
sample and one deliberately altered synthetic ROS scan. It validates the PCAP
parser, actual vendor1.7 bodies, analytically known rebin/minimum behavior,
window/source validation, negative mismatch/truncation and mocked cleanup. It
does not validate privileged execution or explain the original physical event.
