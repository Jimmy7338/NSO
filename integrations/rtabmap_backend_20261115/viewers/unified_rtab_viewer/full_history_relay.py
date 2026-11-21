"""One current-session GetMap(global=True), then original live MapData relay.
Visualization only; no commands, TF, external DB, sensor fabrication or diskdump.
"""
import argparse,json,threading,time
from pathlib import Path

def build_request():
    from rtabmap_msgs.srv import GetMapRequest
    req=GetMapRequest();req.global_=True;req.optimized=True;req.graphOnly=False
    return req

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--expected-database',required=True);ap.add_argument('--epoch-id',required=True);ap.add_argument('--epoch-start-s',type=float,required=True);ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args()
    import rospy
    from rtabmap_msgs.msg import MapData
    from rtabmap_msgs.srv import GetMap,GetMapRequest
    rospy.init_node('nso_unified_rtab_history_relay')
    expected=Path(a.expected_database).resolve();actual=Path(rospy.get_param('/nso_loop/rtabmap/database_path')).expanduser().resolve()
    if actual!=expected or a.epoch_id not in expected.parts:raise ValueError('current owned session database mismatch')
    pub=rospy.Publisher('/nso_unified_rtab/mapData',MapData,queue_size=1,latch=True)
    lock=threading.Lock();state={'ready':False,'last':-1}
    def live(m):
        if m._connection_header.get('callerid')!='/nso_loop/rtabmap':return
        if m.header.frame_id!='nso_loop_map' or m.header.stamp.to_sec()<a.epoch_start_s:return
        with lock:
            if state['ready'] and m.header.stamp.to_nsec()>state['last']:
                pub.publish(m);state['last']=m.header.stamp.to_nsec()
                # First live message replaces latched fullsnapshot and releases its payload.
    sub=rospy.Subscriber('/nso_loop/mapData',MapData,live,queue_size=1,buff_size=16*1024*1024)
    deadline=time.monotonic()+30
    while not pub.get_num_connections() and not rospy.is_shutdown():
        if time.monotonic()>deadline:raise RuntimeError('assembler not subscribed; no fullpayload requested')
        rospy.rostime.wallsleep(.1)
    rospy.wait_for_service('/nso_loop/get_map_data',timeout=10)
    req=build_request()
    response=rospy.ServiceProxy('/nso_loop/get_map_data',GetMap)(req);data=response.data
    if data.header.frame_id!='nso_loop_map' or data.header.stamp.to_sec()<a.epoch_start_s:raise ValueError('history wrong frame/epoch')
    receipt=dict(schema='nso.unified_rtab.full_history.v1',epoch_id=a.epoch_id,database_path=str(expected),
        service='/nso_loop/get_map_data',request=dict(global_map=True,optimized=True,graphOnly=False),
        source_stamp_ns=data.header.stamp.to_nsec(),frame=data.header.frame_id,signature_count=len(data.nodes),
        optimized_pose_count=len(data.graph.posesId),optimized_ids=list(data.graph.posesId),
        RGBD_signature_count=sum(bool(n.data.left_compressed and n.data.right_compressed) for n in data.nodes),
        compressed_RGBD_bytes=sum(len(n.data.left_compressed)+len(n.data.right_compressed) for n in data.nodes),
        full_current_session_requested=True,all_DB_signatures_returned_claim=False,
        source_header_unmodified=True,navigation_authority=False,old_TSDF_modified=False,physical_accuracy_verified=False)
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    with lock:pub.publish(data);state.update(ready=True,last=data.header.stamp.to_nsec())
    del data,response
    rospy.spin()
if __name__=='__main__':main()
