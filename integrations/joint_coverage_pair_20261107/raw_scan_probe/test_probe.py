"""Synthetic-only tests; never opens sockets, ROS, tcpdump or sudo."""
import copy,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
import analyze,capture
P=Path(__file__).resolve().parent

def payload(number):
    b=bytearray(1206)
    for block in range(12):
        struct.pack_into('<BBH',b,block*100,255,238,((number*12+block)*540)%36000)
        for j in range(31):
            offset=block*100+j*3+4
            if j==15:b[offset:offset+3]=b'\xfa\xfb\xfc';continue
            index=number*360+block*30+j-(j>15)
            count=126 if index==2604 else 0 if index in [2603,2605] else 1000+index%100
            struct.pack_into('<HB',b,offset,count,7 if index==2604 else 10)
    return bytes(b)

def ethernet_udp(data):
    udp=struct.pack('!HHHH',2368,2368,8+len(data),0)+data
    ip=bytearray(struct.pack('!BBHHHBBH4s4s',0x45,0,20+len(udp),0,0,64,17,0,bytes([192,168,1,222]),bytes([192,168,1,125])))
    total=sum(struct.unpack('!10H',ip));total=(total&65535)+(total>>16);total=(total&65535)+(total>>16);struct.pack_into('!H',ip,10,(~total)&65535)
    return bytes.fromhex('00112233445566778899aabb0800')+ip+udp

def fixture(directory):
    directory.mkdir(exist_ok=True);raw=bytearray(struct.pack('<IHHIIII',0xa1b23c4d,2,4,0,0,65535,1))
    for i in range(18):
        packet=ethernet_udp(payload(i));raw+=struct.pack('<IIII',1700000000,i*20000000,len(packet),len(packet))+packet
    (directory/'lidar.pcap').write_bytes(raw)
    rows,_=analyze.read_pcap(directory/'lidar.pcap');o=analyze.Oracle();groups,_=analyze.sweeps(rows,o);assert len(groups)==2
    ros={'data_kind':'SYNTHETIC_FIXTURE_NOT_MEASURED','scans':[],'clouds':[],'times':[],'callback_errors':[],'overflow_counts':{'scans':0,'clouds':0,'times':0}};arrays={}
    for i,group in enumerate(groups):
        ranges,ints,cloud=o.rebin(group);key='scan_%03d'%i;ik=key+'_intensities';arrays[key]=ranges;arrays[ik]=ints
        timing={'caller':'/lslidar_driver_node','receipt_monotonic_ns':100000000+i*100000000,'receipt_wall_ns':1700000000100000000+i*100000000}
        ros['scans'].append(dict(timing,ranges_key=key,intensities_key=ik,stamp_ns=group[-1][4]+1000000))
        ros['clouds'].append(dict(timing,stamp_ns=group[-1][4]+1000000,canonical_xyzi_sha256=analyze.sha(np.ascontiguousarray(cloud,dtype='<f4').tobytes())))
        ros['times'].append(dict(timing,source_time_s=(group[-1][4]+1000000)/1e9))
    # A deliberate mismatch must not be mistaken for an exact geometry match.
    bad=arrays['scan_000'].copy();bad[396]+=.01;arrays['scan_002']=bad;arrays['scan_002_intensities']=arrays['scan_000_intensities'].copy();ros['scans'].append(dict(ros['scans'][0],ranges_key='scan_002',intensities_key='scan_002_intensities'))
    np.savez_compressed(directory/'scans.npz',**arrays);(directory/'ros_window.json').write_text(json.dumps(ros,indent=2)+'\n')
    (directory/'FIXTURE_NOT_MEASURED.json').write_text(json.dumps({'kind':'SYNTHETIC','packets':18,'description':'Artificial N301 1.7 packets, two retained complete turns; one .504m sample plus intentionally mismatched ROS scan. No real sensor evidence.'},indent=2)+'\n')
    return rows,ros

class Tests(unittest.TestCase):
    def setUp(self):self.o=analyze.Oracle()
    def test_native_decode_independent_values_and_skip(self):
        az,r,i=self.o.decode(payload(0));self.assertEqual(len(az),360)
        self.assertEqual(az[:16].tolist(),list(range(0,16*18,18)))
        self.assertEqual(r[0],np.float32(np.float32(1000)*np.float32(.004)))
        self.assertEqual(r[15],np.float32(np.float32(1015)*np.float32(.004)))
        self.assertEqual(i[15],10.) # Metadata slot15 must not become a point.
    def test_native_rebin_known_bin_min_and_invalid(self):
        r,i,c=self.o.rebin([(0,1.2,20.),(0,.504,7.),(18000,.1,30.)])
        self.assertEqual(np.flatnonzero(np.isfinite(r)).tolist(),[1000]);self.assertEqual(r[1000],np.float32(.504));self.assertEqual(i[1000],7.);self.assertEqual(len(c),2)
    def test_pcap_and_complete_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            rows,ros=fixture(Path(td));result=analyze.analyze(td)
            self.assertEqual(result['data_kind'],'SYNTHETIC_FIXTURE_NOT_MEASURED');self.assertEqual(result['pcap']['precision_ns'],1);self.assertEqual(result['pcap']['packets'],18)
            self.assertEqual(result['matched_scan_indices'],[0,1]);self.assertEqual(result['unmatched_scan_indices'],[2]);self.assertEqual(result['matched_cloud_indices'],[0,1])
            self.assertEqual(sum(s['raw_distance_0_504_count'] for s in result['complete_raw_sweeps']),1)
            self.assertAlmostEqual(result['complete_raw_sweeps'][0]['minimum_range_m'],.504,places=6)
            receipt={'ROS_window_start_wall_ns':1700000000000000000,'ROS_window_end_wall_ns':1700000000500000000,'ROS_window_start_monotonic_ns':0,'ROS_window_end_monotonic_ns':500000000}
            capture.validate_window(rows,ros,receipt)
            bad=copy.deepcopy(receipt);bad['ROS_window_start_wall_ns']+=10000000000
            with self.assertRaises(ValueError):capture.validate_window(rows,ros,bad)
            badrows=copy.deepcopy(ros);badrows['scans'][0]['caller']='/wrong'
            with self.assertRaises(ValueError):capture.validate_window(rows,badrows,receipt)
    def test_reject_truncated_packet_file(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(Path(td));p=Path(td)/'lidar.pcap';p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):analyze.read_pcap(p)
        with self.assertRaises(ValueError):analyze.udp_payload(ethernet_udp(payload(0))[:30])
    def test_capture_argv_is_bounded_and_passive(self):
        cmd=capture.command(Path('/tmp/fixture'),True)
        self.assertEqual(cmd[:3],['sudo','-n','--']);self.assertIn('7.5s',cmd);self.assertIn('--kill-after=0.3s',cmd);self.assertEqual(cmd[cmd.index('-i')+1],'eth0');self.assertIn('-p',cmd);self.assertEqual(cmd[cmd.index('-c')+1],'2000');self.assertEqual(cmd[-1],capture.FILTER);self.assertEqual(cmd[cmd.index('-Q')+1],'in')
    def test_cleanup_only_owned_process(self):
        class Proc:
            def __init__(self):self.signals=[];self.waits=0;self.terminated=False
            def poll(self):return None
            def send_signal(self,s):self.signals.append(s)
            def wait(self,timeout):
                self.waits+=1
                if self.waits==1:raise capture.subprocess.TimeoutExpired('fixture',timeout)
            def terminate(self):self.terminated=True
        p=Proc();capture.stop_process(p);self.assertEqual(p.signals,[capture.signal.SIGINT]);self.assertTrue(p.terminated)

if __name__=='__main__':unittest.main()
