#!/usr/bin/env python3
"""One bounded passive lidar capture. Run interactively on the robot terminal.

Only tcpdump/timeout may use sudo. No driver, motion, ROS parameters or permissions
are changed. Passwords are handled by sudo on the local terminal, never Python.
"""
import argparse,contextlib,getpass,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
FILTER='src host 192.168.1.222 and udp dst port 2368'

def command(output,sudo=False):
    # 7.5 s + 0.3 s forced-cleanup allowance remains below the 8 s bound.
    args=['/usr/bin/timeout','--signal=INT','--kill-after=0.3s','7.5s',
          '/usr/local/bin/tcpdump','-p','-i','eth0','-Q','in','-nn','-s','0',
          '-j','host','--time-stamp-precision=nano','-c','2000','-Z',getpass.getuser(),
          '-w',str(output/'lidar.pcap'),FILTER]
    return (['sudo','-n','--'] if sudo else [])+args

def stop_process(proc):
    if proc is None or proc.poll() is not None:return
    proc.send_signal(signal.SIGINT)
    try:proc.wait(timeout=1.)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:proc.wait(timeout=1.)
        except subprocess.TimeoutExpired:raise RuntimeError('capture supervisor did not stop; retain PID for local inspection')

def validate_window(packets,records,receipt):
    if not packets:raise ValueError('No packets captured')
    if len(packets)>2000:raise ValueError('Packet bound exceeded')
    stamps=[p['kernel_stamp_ns'] for p in packets]
    if any(b<a for a,b in zip(stamps,stamps[1:])):raise ValueError('PCAP host time regressed')
    start,end=stamps[0],stamps[-1]
    if end-start>8000000000:raise ValueError('PCAP exceeds permitted time span')
    if end<receipt['ROS_window_start_wall_ns'] or start>receipt['ROS_window_end_wall_ns']:raise ValueError('PCAP and ROS receipt windows do not overlap')
    if not 0<receipt['ROS_window_end_monotonic_ns']-receipt['ROS_window_start_monotonic_ns']<=8000000000:raise ValueError('ROS window exceeds time bound')
    if records['callback_errors'] or any(records['overflow_counts'].values()):raise ValueError('ROS callback error or bound exceeded')
    for kind in ['scans','clouds','times']:
        if not 0<len(records[kind])<=90:raise ValueError('ROS source absent or count exceeded: '+kind)
        for row in records[kind]:
            if row['caller']!='/lslidar_driver_node':raise ValueError('Unexpected ROS source authority')
            if not receipt['ROS_window_start_monotonic_ns']<=row['receipt_monotonic_ns']<=receipt['ROS_window_end_monotonic_ns']:raise ValueError('ROS message outside receipt window')
    return {'kernel_stamp_ns_bounds':[start,end],'PCAP_span_s':(end-start)/1e9,'ROS_counts':{k:len(records[k]) for k in ['scans','clouds','times']},'ROS_window_s':(receipt['ROS_window_end_monotonic_ns']-receipt['ROS_window_start_monotonic_ns'])/1e9}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--sudo',action='store_true',help='Call sudo -v interactively, then use cached authorization only for timeout/tcpdump')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if os.geteuid()==0:raise SystemExit('Run Python as the desktop user with --sudo; only packet capture should be elevated')
    def interrupted(signum,unused):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    receipt={'schema':'nso.bounded_passive_lidar_capture.v1','status':'preparing','bounds':{'capture_seconds_max':8,'packets_max':2000,'messages_per_ROS_topic_max':90},'interface':'eth0','filter':FILTER,'promiscuous':False,'created_unix_ns':time.time_ns(),'ROS_master':'http://127.0.0.1:11382'}
    proc=None;subscribers=[];lock=threading.Lock();armed=False
    records={'data_kind':'measured_raw_ROS_window','scans':[],'clouds':[],'times':[],'callback_errors':[],'overflow_counts':{'scans':0,'clouds':0,'times':0}};arrays={}
    def save_receipt():(out/'capture_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    def header(msg):
        return {'stamp_ns':msg.header.stamp.to_nsec(),'seq':msg.header.seq,'frame':msg.header.frame_id,'caller':msg._connection_header.get('callerid'),'receipt_wall_ns':time.time_ns(),'receipt_monotonic_ns':time.monotonic_ns()}
    def store(kind,msg):
        nonlocal armed
        with lock:
            if not armed:return
            if len(records[kind])>=90:records['overflow_counts'][kind]+=1;return
            try:
                index=len(records[kind])
                if kind=='scans':
                    row=header(msg);key='scan_%03d'%index;ik=key+'_intensities';arrays[key]=np.asarray(msg.ranges,dtype='<f4').copy();arrays[ik]=np.asarray(msg.intensities,dtype='<f4').copy()
                    row.update(ranges_key=key,intensities_key=ik,angle_min=msg.angle_min,angle_max=msg.angle_max,angle_increment=msg.angle_increment,range_min=msg.range_min,range_max=msg.range_max,time_increment=msg.time_increment,scan_time=msg.scan_time,ranges_sha256=hashlib.sha256(arrays[key].tobytes()).hexdigest())
                elif kind=='clouds':
                    from sensor_msgs.point_cloud2 import read_points
                    points=np.asarray(list(read_points(msg,field_names=('x','y','z','intensity'),skip_nans=False)),dtype='<f4').reshape((-1,4));row=header(msg)
                    distances=np.hypot(points[:,0].astype(float),points[:,1].astype(float));order=np.argsort(distances)[:8]
                    row.update(point_count=len(points),canonical_xyzi_sha256=hashlib.sha256(points.tobytes()).hexdigest(),message_data_sha256=hashlib.sha256(msg.data).hexdigest(),width=msg.width,height=msg.height,point_step=msg.point_step,row_step=msg.row_step,is_bigendian=msg.is_bigendian,closest_points=[{'point_index':int(i),'xyzi':points[i].astype(float).tolist(),'range_m':float(distances[i])} for i in order if np.isfinite(points[i]).all()])
                else:row={'source_time_s':float(msg.data),'caller':msg._connection_header.get('callerid'),'receipt_wall_ns':time.time_ns(),'receipt_monotonic_ns':time.monotonic_ns()}
                records[kind].append(row)
            except Exception as e:records['callback_errors'].append({'kind':kind,'error':str(e)})
    try:
        master=os.environ.setdefault('ROS_MASTER_URI','http://127.0.0.1:11382')
        if master.rstrip('/')!='http://127.0.0.1:11382':raise ValueError('Refusing a different ROS master')
        from build_oracle import build,V
        if not (P/'vendor_oracle.so').exists():build()
        oracle_meta=json.loads((P/'oracle_build.json').read_text())
        if hashlib.sha256((P/'vendor_oracle.so').read_bytes()).hexdigest()!=oracle_meta['library_sha256']:raise ValueError('Offline oracle library checksum differs')
        if hashlib.sha256((V/'src/lslidar_x10_driver.cpp').read_bytes()).hexdigest()!=oracle_meta['vendor_source_sha256']:raise ValueError('Vendor source changed since oracle build')
        for name,digest in oracle_meta['vendor_header_sha256'].items():
            if hashlib.sha256((V/name).read_bytes()).hexdigest()!=digest:raise ValueError('Vendor header changed since oracle build: '+name)
        from analyze import Oracle
        Oracle() # Fail before capture if the local offline library cannot load.
        receipt['oracle_build']=oracle_meta
        import rospy
        from sensor_msgs.msg import LaserScan,PointCloud2
        from std_msgs.msg import Float64
        rospy.init_node('nso_passive_raw_scan_probe',anonymous=True,disable_signals=True)
        params=rospy.get_param('/lslidar_driver_node');receipt['driver_parameters']=params
        expected={'lidar_type':'X10','lidar_model':'N301','device_ip':'192.168.1.222','msop_port':2368,'serial_port':'','use_time_service':False,'min_range':.15,'max_range':25.,'enable_noise_filter':False,'is_pretreatment':False,'use_high_precision':False,'angle_disable_min':0,'angle_disable_max':0,'frame_id':'laser_link','publish_scan':True}
        differences={k:[v,params.get(k)] for k,v in expected.items() if params.get(k)!=v}
        if differences:raise ValueError('Offline decoder profile differs: '+str(differences))
        if args.sudo:
            # Local terminal prompt only. No password is read, stored or logged by Python.
            if not sys.stdin.isatty():raise ValueError('--sudo requires an interactive local terminal')
            subprocess.run(['sudo','-v'],check=True)
        subscribers=[rospy.Subscriber('/scan',LaserScan,lambda m:store('scans',m),queue_size=10),rospy.Subscriber('/lslidar_point_cloud',PointCloud2,lambda m:store('clouds',m),queue_size=10),rospy.Subscriber('/time_topic',Float64,lambda m:store('times',m),queue_size=10)]
        receipt['command']=command(out,args.sudo);env=dict(os.environ);env['LC_ALL']='C'
        with (out/'tcpdump.log').open('wb') as log:
            proc=subprocess.Popen(receipt['command'],stdout=log,stderr=log,env=env)
            receipt['capture_supervisor_pid']=proc.pid;started=time.monotonic()
            while proc.poll() is None and time.monotonic()-started<2.:
                if 'listening on eth0' in (out/'tcpdump.log').read_text(errors='replace'):break
                time.sleep(.02)
            if proc.poll() is not None:raise RuntimeError('tcpdump did not start; see tcpdump.log')
            if 'listening on eth0' not in (out/'tcpdump.log').read_text(errors='replace'):raise RuntimeError('No capture readiness confirmation')
            with lock:
                receipt['ROS_window_start_wall_ns']=time.time_ns();receipt['ROS_window_start_monotonic_ns']=time.monotonic_ns();armed=True
            while proc.poll() is None and time.monotonic()-started<7.85:time.sleep(.01)
            with lock:
                armed=False;receipt['ROS_window_end_wall_ns']=time.time_ns();receipt['ROS_window_end_monotonic_ns']=time.monotonic_ns()
            stop_process(proc);receipt['tcpdump_supervisor_returncode']=proc.returncode
        for s in subscribers:
            with contextlib.suppress(Exception):s.unregister()
        subscribers=[]
        np.savez_compressed(out/'scans.npz',**arrays)
        (out/'ros_window.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        from analyze import read_pcap,udp_payload
        packets,meta=read_pcap(out/'lidar.pcap')
        for packet in packets:udp_payload(packet['bytes'])
        receipt.update(status='captured_validated',PCAP=meta,**validate_window(packets,records,receipt))
        if not (P/'vendor_oracle.so').exists():
            from build_oracle import build
            build()
        from analyze import analyze
        analysis=analyze(out);receipt['offline_comparison']='comparison.json';receipt['complete_sweeps']=len(analysis['complete_raw_sweeps'])
    except BaseException as e:
        receipt.update(status='failed_or_cancelled',error={'type':type(e).__name__,'message':str(e)})
        raise
    finally:
        with lock:
            armed=False
            if 'ROS_window_start_wall_ns' in receipt:
                receipt.setdefault('ROS_window_end_wall_ns',time.time_ns())
                receipt.setdefault('ROS_window_end_monotonic_ns',time.monotonic_ns())
        for s in subscribers:
            with contextlib.suppress(Exception):s.unregister()
        try:stop_process(proc)
        except Exception as e:receipt['cleanup_error']=str(e)
        if any(records[k] for k in ['scans','clouds','times']):
            # A cancelled single window remains reviewable; never silently reacquire.
            receipt.setdefault('ROS_window_end_wall_ns',time.time_ns())
            receipt.setdefault('ROS_window_end_monotonic_ns',time.monotonic_ns())
            try:
                if not (out/'scans.npz').exists():np.savez_compressed(out/'scans.npz',**arrays)
                if not (out/'ros_window.json').exists():(out/'ros_window.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
            except Exception as e:receipt['partial_ROS_save_error']=str(e)
        receipt['finished_unix_ns']=time.time_ns();save_receipt()
    print(json.dumps({'status':receipt['status'],'output':str(out),'ROS_counts':receipt.get('ROS_counts')},indent=2))

if __name__=='__main__':main()
