"""Receive-only aggregate CAN telemetry; no command publisher or policy hook."""
import argparse,json,socket,struct,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
s.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,b''.join(struct.pack('=II',i,0xc00007ff) for i in (0x111,0x211,0x221)))
s.setsockopt(socket.SOL_SOCKET,35,1);s.bind(('can0',));s.settimeout(.2)
start=time.time_ns();end=time.monotonic()+700;finished=None;stats={};errors=0;last_state=None
while time.monotonic()<end:
 if (a.report/'result.json').exists():
  if finished is None:finished=time.monotonic()+3
  if time.monotonic()>=finished:break
 try:r,anc,flags,_=s.recvmsg(16,128)
 except socket.timeout:continue
 if len(r)!=16 or flags&(8|32):errors+=1;continue
 cid,n,d=struct.unpack('=IB3x8s',r)
 if n!=8:errors+=1;continue
 stamp=next((struct.unpack('@ll',v[:struct.calcsize('@ll')]) for lev,kind,v in anc if lev==socket.SOL_SOCKET and kind==35),None)
 if stamp is None:errors+=1;continue
 ns=stamp[0]*1000000000+stamp[1]
 row=dict(can_id=hex(cid),kernel_stamp_ns=ns,payload_hex=d.hex(),socket_flags=flags)
 st=stats.setdefault(hex(cid),dict(count=0,first_stamp_ns=ns,last_stamp_ns=ns,nonincreasing=0,max_gap_ns=0))
 if st['count']:
  dt=ns-st['last_stamp_ns'];st['nonincreasing']+=int(dt<=0);st['max_gap_ns']=max(st['max_gap_ns'],dt)
 st['count']+=1;st['last_stamp_ns']=ns
 if cid==0x211:last_state=dict(state=d[0],mode=d[1],error=struct.unpack('>H',d[4:6])[0],**row)
 else:
  v,w=[x*.001 for x in struct.unpack('>hh',d[:4])];row.update(v=v,w=w)
  for label,value in [('abs_v',abs(v)),('abs_w',abs(w))]:
   if label not in st or value>st[label]['value']:st[label]=dict(value=value,frame=row)
  st['min_v']=min(st.get('min_v',v),v);st['max_v']=max(st.get('max_v',v),v)
  st['last']=row
s.close()
out=dict(schema='nso.passive_CAN_maxima.v1',started_unix_ns=start,finished_unix_ns=time.time_ns(),report=str(a.report),terminal_seen=finished is not None,counts_and_maxima=stats,last_status=last_state,invalid_receive_count=errors,receive_only=True,commands='Other-socket SocketCAN local loopback frames; not independent proof of controller acceptance',scope='All received frames from this observer start, including terminal3s; source stamps preserved; no continuous raw recording',stop_rules_affected=False)
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'output':str(a.output),'terminal_seen':out['terminal_seen']}))
