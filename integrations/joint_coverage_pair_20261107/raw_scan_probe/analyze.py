#!/usr/bin/env python3
"""Offline PCAP -> vendor N301 1.7 -> wrap groups -> native rebin comparison.

Uses captured kernel timestamps unchanged. Does not execute PCAP driver playback.
"""
import argparse,ctypes,hashlib,json,math,struct
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()

def read_pcap(path):
    raw=Path(path).read_bytes()
    choices={b'\x4d\x3c\xb2\xa1':('<',1),b'\xa1\xb2\x3c\x4d':('>',1),b'\xd4\xc3\xb2\xa1':('<',1000),b'\xa1\xb2\xc3\xd4':('>',1000)}
    if len(raw)<24 or raw[:4] not in choices:raise ValueError('not a complete classic PCAP header')
    endian,scale=choices[raw[:4]];major,minor,_,_,snap,link=struct.unpack(endian+'HHIIII',raw[4:24])
    if (major,minor)!=(2,4) or link!=1:raise ValueError('expected Ethernet PCAP 2.4')
    rows=[];pos=24
    while pos<len(raw):
        if len(raw)-pos<16:raise ValueError('truncated PCAP record header')
        sec,frac,n,original=struct.unpack(endian+'IIII',raw[pos:pos+16]);pos+=16
        if frac*scale>=1000000000:raise ValueError('invalid PCAP fractional timestamp')
        if n!=original or n>snap or pos+n>len(raw):raise ValueError('truncated captured packet')
        packet=raw[pos:pos+n];pos+=n
        rows.append({'kernel_stamp_ns':sec*1000000000+frac*scale,'bytes':packet})
    if len(rows)>2000:raise ValueError('packet bound exceeded')
    return rows,{'precision_ns':scale,'link_type':link,'packets':len(rows),'pcap_sha256':sha(raw)}

def udp_payload(packet):
    if len(packet)<14:raise ValueError('short Ethernet')
    proto=struct.unpack('!H',packet[12:14])[0];offset=14
    while proto in (0x8100,0x88a8):
        if len(packet)<offset+4:raise ValueError('short VLAN')
        proto=struct.unpack('!H',packet[offset+2:offset+4])[0];offset+=4
    if proto!=0x0800 or len(packet)<offset+20:raise ValueError('not IPv4')
    ip=packet[offset:];ihl=(ip[0]&15)*4;total=struct.unpack('!H',ip[2:4])[0]
    if ip[0]>>4!=4 or ihl<20 or total>len(ip) or ip[9]!=17:raise ValueError('invalid IPv4 UDP')
    if struct.unpack('!H',ip[6:8])[0]&0x3fff:raise ValueError('fragmented UDP not supported')
    if ip[12:16]!=bytes([192,168,1,222]):raise ValueError('unexpected source IP')
    udp=ip[ihl:total]
    if len(udp)<8:raise ValueError('short UDP')
    src,dst,length,_=struct.unpack('!HHHH',udp[:8])
    if dst!=2368 or length!=len(udp):raise ValueError('unexpected destination/truncated UDP')
    payload=udp[8:]
    if len(payload)!=1206:raise ValueError('not N301 1206-byte packet')
    return payload

class Oracle:
    def __init__(self, library_path=None):
        self.lib=ctypes.CDLL(str(Path(library_path).resolve() if library_path is not None else P/'vendor_oracle.so'))
        ip=ctypes.POINTER(ctypes.c_int);fp=ctypes.POINTER(ctypes.c_float)
        self.lib.decode.argtypes=[ctypes.POINTER(ctypes.c_ubyte),ip,fp,fp];self.lib.decode.restype=ctypes.c_int
        self.lib.rebin.argtypes=[ip,fp,fp,ctypes.c_int,fp,fp,fp];self.lib.rebin.restype=ctypes.c_int
    def decode(self,payload):
        b=np.frombuffer(payload,np.uint8).copy();az=np.empty(360,np.int32);r=np.empty(360,np.float32);i=np.empty(360,np.float32)
        n=self.lib.decode(b.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),az.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if n!=360:raise ValueError('vendor N301 packet validity rejected')
        return az,r,i
    def rebin(self,points):
        az=np.ascontiguousarray([p[0] for p in points],dtype=np.int32);r=np.ascontiguousarray([p[1] for p in points],dtype=np.float32);i=np.ascontiguousarray([p[2] for p in points],dtype=np.float32)
        ranges=np.empty(2000,np.float32);ints=np.empty(2000,np.float32);cloud=np.empty((len(points),4),np.float32)
        ip=lambda v:v.ctypes.data_as(ctypes.POINTER(ctypes.c_int));fp=lambda v:v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n=self.lib.rebin(ip(az),fp(r),fp(i),len(points),fp(ranges),fp(ints),fp(cloud))
        if n<0:raise ValueError('unexpected native rebin size')
        return ranges,ints,cloud[:n]

def sweeps(rows,oracle):
    groups=[];pending=[];started=False;last=None;errors=[];multiwrap=[]
    for index,row in enumerate(rows):
        try:az,rs,ins=oracle.decode(udp_payload(row['bytes']))
        except ValueError as e:errors.append({'packet_index':index,'reason':str(e)});continue
        wraps=0
        for a,r,i in zip(az,rs,ins):
            a=int(a)
            if last is not None and abs(a-last)>35000:
                wraps+=1
                if started and pending:groups.append(pending)
                pending=[];started=True
            if started:pending.append((a,float(r),float(i),index,row['kernel_stamp_ns']))
            last=a
        if wraps>1:multiwrap.append(index)
    return groups,{'packet_decode_errors':errors,'multiwrap_packet_indices':multiwrap,'boundary_partial_sweeps_discarded':True}

def analyze(directory, library_path=None, output_path=None):
    directory=Path(directory);rows,pcap=read_pcap(directory/'lidar.pcap')
    if not rows:raise ValueError('empty PCAP: no acquisition evidence')
    if (rows[-1]['kernel_stamp_ns']-rows[0]['kernel_stamp_ns'])>8000000000:raise ValueError('PCAP span exceeds 8 seconds')
    oracle=Oracle(library_path);groups,quality=sweeps(rows,oracle)
    ros=json.loads((directory/'ros_window.json').read_text());scans=ros['scans'];clouds=ros['clouds'];npz=np.load(directory/'scans.npz',allow_pickle=False)
    if len(scans)>90 or len(clouds)>90 or len(ros['times'])>90:raise ValueError('ROS bound exceeded')
    output=[];paired_scans=set();paired_clouds=set()
    for idx,group in enumerate(groups):
        ranges,ints,cloud=oracle.rebin(group);matches=[]
        for j,scan in enumerate(scans):
            actual=npz[scan['ranges_key']];intensity=npz[scan['intensities_key']]
            if actual.shape!=ranges.shape:continue
            if np.array_equal(np.isfinite(actual),np.isfinite(ranges)) and np.allclose(actual,ranges,atol=1e-6,rtol=1e-6,equal_nan=True) and intensity.shape==ints.shape and np.allclose(intensity,ints,atol=0,rtol=0,equal_nan=True):
                matches.append({'scan_index':j,'source_stamp_ns':scan['stamp_ns'],'bitwise_ranges_match':actual.tobytes()==ranges.tobytes(),'stamp_minus_last_packet_s':(scan['stamp_ns']-group[-1][4])/1e9});paired_scans.add(j)
        chash=sha(np.ascontiguousarray(cloud,dtype='<f4').tobytes());cm=[j for j,c in enumerate(clouds) if c['canonical_xyzi_sha256']==chash];paired_clouds.update(cm)
        finite=ranges[np.isfinite(ranges)]
        associations=[{'scan_index':m['scan_index'],'cloud_index':j,'scan_minus_cloud_stamp_ns':m['source_stamp_ns']-clouds[j]['stamp_ns']} for m in matches for j in cm if 'stamp_ns' in clouds[j]]
        output.append({'sweep_index':idx,'packet_index_bounds':[group[0][3],group[-1][3]],'kernel_stamp_ns_bounds':[group[0][4],group[-1][4]],'raw_point_count':len(group),'valid_cloud_points':len(cloud),'minimum_range_m':float(finite.min()) if len(finite) else None,'raw_distance_0_504_count':sum(abs(p[1]-.504)<1e-6 for p in group),'cloud_matches':cm,'scan_matches':matches,'same_geometry_source_stamp_associations':associations,'ranges_sha256':sha(ranges.tobytes())})
    stamp_regressions={kind:[i for i in range(1,len(ros[kind])) if ros[kind][i]['stamp_ns']<ros[kind][i-1]['stamp_ns']] for kind in ['scans','clouds']}
    time_associations=[{'scan_index':i,'time_topic_indices_within_1us_float_representation':[j for j,t in enumerate(ros['times']) if abs(t['source_time_s']-s['stamp_ns']/1e9)<=1e-6]} for i,s in enumerate(scans)]
    receipt={'schema':'nso.passive_raw_scan_comparison.v1','data_kind':ros.get('data_kind','unspecified'),'pcap':pcap,'complete_raw_sweeps':output,'quality':quality,'ROS_counts':{k:len(ros[k]) for k in ['scans','clouds','times']},'source_stamp_regressions':stamp_regressions,'time_topic_associations':time_associations,'matched_scan_indices':sorted(paired_scans),'matched_cloud_indices':sorted(paired_clouds),'unmatched_scan_indices':[i for i in range(len(scans)) if i not in paired_scans],'limits':['Only this finite window; nonrecurrence excludes no failure hypothesis','Raw-packet echo presence and sweep/header mispairing can coexist','Full-geometry match can be ambiguous if consecutive sweeps repeat','Host PCAP kernel timestamps retained; driver poll-time/interpolated scan stamp is not assumed equal','Float64 time_topic comparison permits 1us representation difference only; not beam-time accuracy','No per-beam acquisition time or deskew invented; no physical identity/noise classification','Boundary partial sweeps, packet loss and unsupported input are disclosed rather than treated as driver faults']}
    payload=json.dumps(receipt,indent=2,allow_nan=False)+'\n'
    if output_path is None:
        (directory/'comparison.json').write_text(payload)
    else:
        # Explicit offline outputs cannot overwrite retained measurements/results.
        with Path(output_path).open('x') as target:
            target.write(payload)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--oracle',type=Path,help='Explicit offline library built from the preserved C++ snapshot')
    p.add_argument('--output',type=Path,help='New JSON result path; refuses to overwrite an existing file')
    a=p.parse_args();r=analyze(a.directory,library_path=a.oracle,output_path=a.output)
    print(json.dumps({'packets':r['pcap']['packets'],'complete_sweeps':len(r['complete_raw_sweeps']),'matched_scans':len(r['matched_scan_indices'])}))
