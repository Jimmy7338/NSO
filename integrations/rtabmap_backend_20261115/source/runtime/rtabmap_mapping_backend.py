#!/usr/bin/env python3
"""Explicit RTAB grid/graph epoch alias; no registration, occupancy or policy edits.

Exact-stamp grid+graph epochs are accepted; an explicit hash-bound recorded graph
may bootstrap a matching real live grid after backend restart. Public TF has odom parent
and nso_pair_map child, preserving the existing global tree. The epoch correction
is held at real CSM odometry sample timestamps (derived TF, not sensor restamping).
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from collections import OrderedDict
import numpy as np

PROFILE = 'rtabmap_grid_graph_exact_epoch_v1'


def stamp_ns(msg):
    return msg.header.stamp.to_nsec()


def caller(msg):
    return getattr(msg, '_connection_header', {}).get('callerid')


def finite_transform(t):
    p, q = t.translation, t.rotation
    v = (p.x,p.y,p.z,q.x,q.y,q.z,q.w)
    return all(math.isfinite(x) for x in v) and abs(sum(x*x for x in v[3:])-1.) < 1e-5


def transform_record(t):
    p,q=t.translation,t.rotation
    return dict(translation=[p.x,p.y,p.z],rotation_xyzw=[q.x,q.y,q.z,q.w])


def matrix(record):
    x,y,z,w=record['rotation_xyzw'];p=record['translation']
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),p[0]],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),p[1]],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),p[2]],
            [0.,0.,0.,1.]]


def inverse(t):
    result=copy.deepcopy(t)
    p,q=t.translation,t.rotation
    x,y,z,w=q.x,q.y,q.z,q.w
    r=((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
       (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
       (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    v=(p.x,p.y,p.z)
    result.translation.x,result.translation.y,result.translation.z=(-sum(r[j][i]*v[j] for j in range(3)) for i in range(3))
    result.rotation.x,result.rotation.y,result.rotation.z,result.rotation.w=-x,-y,-z,w
    return result


class EpochAdapter:
    def __init__(self, source_caller='/nso_loop/rtabmap', source_frame='nso_loop_map',
                 output_frame='nso_pair_map', odom_caller='/nso_csm_odometry_bridge'):
        self.source_caller=source_caller;self.source_frame=source_frame
        self.output_frame=output_frame;self.odom_caller=odom_caller
        self.lock=threading.RLock();self.grids=OrderedDict();self.graphs=OrderedDict()
        self.epoch=None;self.tf_epoch=None;self.latest_odom=None;self.nodes={};self.graph_sources={};self.initial_metadata_receipt=None;self.recorded_graph_bindings={}

    def load_initial_metadata(self,path,expected_database_path,expected_epoch):
        """Load an explicitly captured metadata-only GetMap2 response, atomically."""
        raw=Path(path).read_bytes();data=json.loads(raw)
        expected_request=dict(global_=True,optimized=False,with_images=False,with_scans=False,
            with_user_data=False,with_grids=False,with_words=False,with_global_descriptors=False)
        request=dict(data.get('request',{}))
        if 'global' in request:request['global_']=request.pop('global')
        if (data.get('schema')!='rtabmap_node_metadata/1'
            or not expected_database_path or expected_database_path=='unspecified'
            or not expected_epoch or expected_epoch=='unspecified'
            or data.get('source_database_path')!=expected_database_path
            or data.get('source_epoch')!=expected_epoch
            or data.get('source_service')!='/nso_loop/get_map_data2'
            or data.get('source_caller')!=self.source_caller
            or request!=expected_request):
            raise ValueError('initial metadata provenance/request mismatch')
        nodes={}
        for row in data['nodes']:
            node_id=row['id'];stamp=row['stamp_s'];record=row['raw_odom_pose']
            values=list(record['translation'])+list(record['rotation_xyzw'])
            if (type(node_id) is not int or node_id<=0 or node_id in nodes
                or type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp<=0
                or len(record['translation'])!=3 or len(record['rotation_xyzw'])!=4
                or not all(type(v) in (int,float) and math.isfinite(v) for v in values)
                or abs(sum(v*v for v in record['rotation_xyzw'])-1.)>=1e-5):
                raise ValueError('invalid initial node metadata')
            nodes[node_id]=dict(id=node_id,map_id=int(row['map_id']),stamp_s=stamp,
                raw_odom_pose=copy.deepcopy(record),source='metadata-only GetMap2 original Node.pose')
        receipt=dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),
            source_database_path=expected_database_path,source_epoch=expected_epoch,
            source_service=data['source_service'],source_caller=data['source_caller'],
            request=request,node_count=len(nodes),schema=data['schema'])
        with self.lock:
            self.nodes.update(nodes);self.initial_metadata_receipt=receipt
        return copy.deepcopy(receipt)

    def load_initial_epoch_graph(self,path,expected_database_path,expected_epoch,graph_factory,manifest_path=None):
        """Import recorded serialized graph, never impersonating a live publisher."""
        raw=Path(path).read_bytes()
        data=json.loads(Path(manifest_path or str(path)+'.json').read_text())
        if (data.get('schema')!='rtabmap_recorded_epoch_graph/1'
            or not expected_database_path or expected_database_path=='unspecified'
            or not expected_epoch or expected_epoch=='unspecified'
            or data.get('source_database_path')!=expected_database_path
            or data.get('source_epoch')!=expected_epoch
            or data.get('actual_recorded_caller')!='/nso_rtabmap_mapping_backend'
            or data.get('original_source_caller')!=self.source_caller
            or data.get('graph_sha256')!=hashlib.sha256(raw).hexdigest()
            or not isinstance(data.get('grid_data_sha256'),str)
            or len(data['grid_data_sha256'])!=64
            or any(c not in '0123456789abcdef' for c in data['grid_data_sha256'])):
            raise ValueError('recorded epoch graph provenance/hash mismatch')
        graph=graph_factory();graph.deserialize(raw)
        ns=stamp_ns(graph)
        if (type(data.get('graph_stamp_ns')) is not int or data['graph_stamp_ns']!=ns or ns<=0
            or graph.header.frame_id!=self.source_frame
            or not finite_transform(graph.mapToOdom)
            or len(graph.posesId)!=len(graph.poses) or len(set(graph.posesId))!=len(graph.posesId)):
            raise ValueError('invalid recorded epoch graph')
        for pose in graph.poses:
            proxy=type('Transform',(),dict(translation=pose.position,rotation=pose.orientation))()
            if not finite_transform(proxy):raise ValueError('invalid recorded graph pose')
        for link in graph.links:
            if not finite_transform(link.transform) or not all(math.isfinite(v) for v in link.information):
                raise ValueError('invalid recorded graph link')
        with self.lock:
            self.graphs[ns]=graph
            self.graph_sources[ns]='recorded exact generation'
            self.recorded_graph_bindings[ns]=dict(data,path=str(Path(path).resolve()))
        return copy.deepcopy(data)

    def receive(self, kind, msg, graph_source_topic='/nso_loop/mapGraph'):
        if caller(msg)!=self.source_caller or msg.header.frame_id!=self.source_frame or stamp_ns(msg)<=0:
            raise ValueError('unexpected RTAB source/frame/stamp')
        if kind=='grid':
            i=msg.info
            if i.width<=0 or i.height<=0 or not math.isfinite(i.resolution) or i.resolution<=0 or len(msg.data)!=i.width*i.height:
                raise ValueError('invalid RTAB occupancy geometry')
            values=np.asarray(msg.data)
            if not np.issubdtype(values.dtype,np.integer) or values.min() < -1 or values.max() > 100:
                raise ValueError('invalid occupancy value')
            pose=i.origin
            proxy=type('Transform',(),dict(translation=pose.position,rotation=pose.orientation))()
            if not finite_transform(proxy):raise ValueError('invalid grid origin')
        elif kind=='graph':
            if not finite_transform(msg.mapToOdom) or len(msg.posesId)!=len(msg.poses) or len(set(msg.posesId))!=len(msg.posesId):
                raise ValueError('invalid RTAB graph')
            for p in msg.poses:
                proxy=type('Transform',(),dict(translation=p.position,rotation=p.orientation))()
                if not finite_transform(proxy):raise ValueError('invalid graph pose')
        else:raise ValueError('unknown input')
        with self.lock:
            cache=self.grids if kind=='grid' else self.graphs
            ns=stamp_ns(msg);cache[ns]=msg  # ROS callback payload retained read-only; no map-array copy.
            if kind=='graph':
                self.graph_sources[ns]=graph_source_topic
                self.recorded_graph_bindings.pop(ns,None)
            while len(cache)>64:
                old,_=cache.popitem(last=False)
                if kind=='graph':self.graph_sources.pop(old,None)
            if ns not in self.grids or ns not in self.graphs:return None
            if self.epoch is not None and ns<=self.epoch[0]:return None
            grid,graph=self.grids[ns],self.graphs[ns]
            recorded=self.recorded_graph_bindings.get(ns)
            if recorded is not None and hashlib.sha256(np.asarray(grid.data,dtype=np.int8).tobytes()).hexdigest()!=recorded['grid_data_sha256']:
                raise ValueError('recorded epoch graph grid binding mismatch')
            # Complete immutable-by-ownership correction is built before atomic publication.
            held_inverse=inverse(graph.mapToOdom)
            self.epoch=(ns,grid,graph)
            self.tf_epoch=(ns,held_inverse)
            out=copy.copy(grid);out.header=copy.copy(grid.header);out.header.frame_id=self.output_frame
            # rospy deserializes int8[] as an immutable tuple. Test/injected mutable
            # arrays get an independent C-level copy, never Python deep-copy per cell.
            if not isinstance(grid.data,(tuple,bytes)):
                out.data=copy.copy(grid.data)
            receipt=dict(schema='rtabmap_mapping_backend/1',profile=PROFILE,
                grid_source_topic='/nso_loop/grid_map',graph_source_topic=self.graph_sources[ns],
                source_caller=self.source_caller,source_frame=self.source_frame,
                alias_frame=self.output_frame,alias_transform='identity; same RTAB optimized coordinates',
                grid_source_stamp_ns=ns,graph_source_stamp_ns=ns,
                grid_data_sha256=hashlib.sha256(np.asarray(grid.data,dtype=np.int8).tobytes()).hexdigest(),
                map_to_odom=transform_record(graph.mapToOdom),
                graph_pose_count=len(graph.posesId),graph_link_count=len(graph.links),
                occupancy='unchanged including unknown; header frame only aliased',
                tf_semantics='inverse epoch mapToOdom held at real odom sample stamp; source epochs retained',
                history='new optimized graph supersedes older node poses; consumers must replay node-associated observations')
            if recorded is not None:receipt['recorded_epoch_graph']=copy.deepcopy(recorded)
            return out,copy.deepcopy(graph),receipt  # Graph copy is small; no grid-array deepcopy.

    def node_metadata(self, msg):
        if caller(msg)!=self.source_caller or msg.header.frame_id!=self.source_frame:
            raise ValueError('unexpected MapData source/frame')
        with self.lock:
            for node in msg.nodes:
                proxy=type('Transform',(),dict(translation=node.pose.position,rotation=node.pose.orientation))()
                if node.id>0 and math.isfinite(node.stamp) and node.stamp>0 and finite_transform(proxy):
                    self.nodes[node.id]=dict(id=node.id,map_id=node.map_id,stamp_s=node.stamp,
                        raw_odom_pose=transform_record(proxy),source='MapData.nodes: Node.pose uncorrected odometry')
            graph=copy.copy(msg.graph)
            # Installed mapDataToROS does not populate nested graph Header.
            # Its enclosing MapData header is the actual shared processing epoch.
            graph.header=copy.copy(msg.header)
            graph._connection_header={'callerid':self.source_caller}
            return graph

    def graph_snapshot(self, graph, receipt):
        with self.lock:
            node_metadata=dict(self.nodes)
            initial_metadata_receipt=copy.deepcopy(self.initial_metadata_receipt)
        poses=[]
        for node_id,pose in zip(graph.posesId,graph.poses):
            proxy=type('Transform',(),dict(translation=pose.position,rotation=pose.orientation))()
            poses.append(dict(id=node_id,optimized_pose=transform_record(proxy),
                node_metadata=copy.deepcopy(node_metadata.get(node_id))))
        links=[dict(from_id=l.fromId,to_id=l.toId,type=l.type,
            transform=transform_record(l.transform),information=list(l.information)) for l in graph.links]
        stable=dict(poses=sorted([dict(id=x['id'],pose=x['optimized_pose']) for x in poses],key=lambda x:x['id']),
            links=sorted(links,key=lambda x:(x['from_id'],x['to_id'],x['type'])))
        revision=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        nodes=[]
        for p in poses:
            meta=p['node_metadata']
            nodes.append(dict(id=int(p['id']),stamp_ns=None if meta is None else int(round(meta['stamp_s']*1e9)),
                odom_from_base=None if meta is None else matrix(meta['raw_odom_pose']),
                map_from_base=matrix(p['optimized_pose'])))
        return dict(receipt,initial_metadata_receipt=initial_metadata_receipt,graph_revision_sha256=revision,map_stamp_ns=receipt['grid_source_stamp_ns'],
            map_frame=self.source_frame,map_from_odom=matrix(receipt['map_to_odom']),nodes=nodes,
            optimized_nodes=poses,links=links,
            metadata_coverage=dict(known=sum(x['node_metadata'] is not None for x in poses),total=len(poses)),
            replay_semantics='optimized_node_pose * inverse(node_raw_odom_pose) * actual_frame_raw_odom_pose; predecessor node by real source time; not independently optimized frame pose')


    def odom(self,msg):
        if caller(msg)!=self.odom_caller or msg.header.frame_id!='odom' or msg.child_frame_id!='base_link' or stamp_ns(msg)<=0:
            raise ValueError('unexpected CSM odometry source/frame/stamp')
        # No map/graph lock on the real-source TF hot path.
        self.latest_odom=copy.deepcopy(msg.header)
        return self.held_transform(msg.header)

    def held_transform(self, header):
        epoch=self.tf_epoch  # One complete atomic tuple, never mixed epochs.
        if epoch is None or header.stamp.to_nsec()<epoch[0]:return None
        return copy.deepcopy(epoch[1]),copy.deepcopy(header),epoch[0]


def main():
    import rospy
    from nav_msgs.msg import OccupancyGrid,MapMetaData,Odometry
    from rtabmap_msgs.msg import MapGraph,MapData
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage
    from std_msgs.msg import String
    rospy.init_node('nso_rtabmap_mapping_backend')
    adapter=EpochAdapter()
    snapshot_path=rospy.get_param('~latest_snapshot_path',rospy.get_param('~graph_snapshot_path',''))
    source_database_path=rospy.get_param('~source_database_path','unspecified')
    source_epoch=rospy.get_param('~source_epoch','unspecified')
    initial_metadata_path=rospy.get_param('~initial_metadata_path','')
    if initial_metadata_path:
        adapter.load_initial_metadata(initial_metadata_path,source_database_path,source_epoch)
    initial_epoch_graph_path=rospy.get_param('~initial_epoch_graph_path','')
    if initial_epoch_graph_path:
        adapter.load_initial_epoch_graph(initial_epoch_graph_path,source_database_path,source_epoch,MapGraph,
            rospy.get_param('~initial_epoch_graph_manifest_path','') or None)
    maps=rospy.Publisher('/nso_pair/map',OccupancyGrid,queue_size=1,latch=True)
    metadata=rospy.Publisher('/nso_pair/map_metadata',MapMetaData,queue_size=1,latch=True)
    graphs=rospy.Publisher('/nso_pair/optimized_graph',MapGraph,queue_size=1,latch=True)
    provenance=rospy.Publisher('/nso_pair/mapping_provenance',String,queue_size=1,latch=True)
    tfs=rospy.Publisher('/tf',TFMessage,queue_size=100)
    def publish_tf(result):
        if result is None:return
        transform,header,epoch=result
        t=TransformStamped();t.header=header;t.header.frame_id='odom'
        t.child_frame_id=adapter.output_frame;t.transform=transform
        tfs.publish(TFMessage(transforms=[t]))
    def write_snapshot(graph,receipt):
        if not snapshot_path:return
        path=Path(snapshot_path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
        temporary.write_text(json.dumps(adapter.graph_snapshot(graph,receipt),sort_keys=True))
        os.replace(str(temporary),str(path))
    latest_receipt=[None]
    publication_lock=threading.RLock()
    def receive(kind,msg,graph_source_topic='/nso_loop/mapGraph'):
        try:
            with publication_lock:
                result=adapter.receive(kind,msg,graph_source_topic)
                if result is None:return
                grid,graph,receipt=result
                if adapter.epoch[0]!=stamp_ns(grid):return  # A newer exact epoch already superseded this one.
                receipt.update(source_database_path=source_database_path,source_epoch=source_epoch)
                latest_receipt[0]=receipt
                write_snapshot(graph,receipt)
                # Both original source graph and generation receipt are latched; no all-history pull.
                graphs.publish(graph);provenance.publish(String(data=json.dumps(receipt,sort_keys=True)))
                # Publish exact epoch and latest real odom evaluation; never use wall-now/future dating.
                publish_tf((inverse(graph.mapToOdom),copy.deepcopy(graph.header),stamp_ns(graph)))
                if adapter.latest_odom is not None:publish_tf(adapter.held_transform(adapter.latest_odom))
                maps.publish(grid);metadata.publish(grid.info)
        except ValueError as exc:rospy.logerr_throttle(5.,'RTAB backend rejected input: %s',exc)
    def odom(msg):
        try:
            publish_tf(adapter.odom(msg))
        except ValueError as exc:rospy.logerr_throttle(5.,'RTAB backend rejected odom: %s',exc)
    def map_data(msg):
        try:
            with publication_lock:
                receive('graph',adapter.node_metadata(msg),'/nso_loop/mapData.graph:enclosing_header')
                # MapGraph may arrive before metadata at the same epoch.
                if adapter.epoch is not None and latest_receipt[0] is not None:
                    write_snapshot(adapter.epoch[2],latest_receipt[0])
        except ValueError as exc:rospy.logerr_throttle(5.,'RTAB backend rejected metadata: %s',exc)
    subscribers=[rospy.Subscriber('/nso_loop/grid_map',OccupancyGrid,lambda m:receive('grid',m),queue_size=2),
                 rospy.Subscriber('/nso_loop/mapGraph',MapGraph,lambda m:receive('graph',m),queue_size=2),
                 rospy.Subscriber('/nso_loop/mapData',MapData,map_data,queue_size=1),
                 rospy.Subscriber('/nso_csm/odom',Odometry,odom,queue_size=100)]
    rospy.spin()


if __name__=='__main__':main()
