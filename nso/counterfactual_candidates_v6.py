"""Label-agnostic additions to an immutable counterfactual route pool.

This changes candidate sufficiency only. No semantic/quality score or evaluator
is consulted. Observed-cluster support in the camera frustum defines directional
representation; it is not an occlusion test or a promise of new information.
"""
import hashlib
import json
import numbers

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label
from scipy.sparse.csgraph import dijkstra

from env.grid_exploration import GridConfig
from env.virtual3d import camera_pose
from nso.quality_coverage3d import QualityCoveragePolicy
from nso.route_coverage_v2 import orientation_graph
from nso.semantic_completion_v3 import ObjectCompletionModel
from utils.grid_geometry import DIRECTIONS


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _observed_support(mapper, objects):
    """Recover the exact geometric components used by the frozen V3 model."""
    quality=mapper.quality_evidence(max_points=10000)
    if quality is None:return []
    points=quality['point'];r=mapper.shape[0]-1-np.floor(points[:,1]/.2).astype(int)
    c=np.floor(points[:,0]/.2).astype(int)
    inside=(r>=0)&(r<mapper.shape[0])&(c>=0)&(c<mapper.shape[1])
    occupied=np.zeros(mapper.shape,bool);occupied[r[inside],c[inside]]=True
    groups,_=label(binary_closing(occupied,structure=np.ones((3,3))))
    ids=np.zeros(len(points),int);ids[inside]=groups[r[inside],c[inside]]
    support=[]
    for group in sorted(set(ids)-{0}):
        cloud=points[ids==group]
        if len(cloud)<8:continue
        low,high=np.quantile(cloud,[.05,.95],axis=0);span=high-low
        if max(span[:2])>2.2 or span[2]<.25:continue
        support.append(cloud)
    if len(support)!=len(objects) or any(len(p)!=o['observed_points'] for p,o in zip(support,objects)):
        raise ValueError('observed support no longer matches geometric object model')
    return support


def _support_in_frustum(points, pose, config):
    local=(points-pose[:3,3])@pose[:3,:3];z=local[:,2]
    tangent=np.tan(np.deg2rad(config.fov_deg/2))
    valid=(z>.15)&(z<=config.max_depth_m)
    valid&=(np.abs(local[:,0])<z*tangent)
    valid&=(np.abs(local[:,1])<z*tangent*config.height_px/config.width_px)
    return int(valid.sum())


def _sector(position_xy, center_xy):
    delta=position_xy-center_xy
    return int(np.floor(((np.arctan2(delta[1],delta[0])+np.pi/8)%(2*np.pi))/(np.pi/4)))%8


def _path(previous,start,target,cells,reverse=False):
    chain=[int(target)]
    while chain[-1]!=start:
        nxt=int(previous[chain[-1]])
        if nxt<0 or len(chain)>len(previous):raise ValueError('unreachable orientation path')
        chain.append(nxt)
    if not reverse:chain.reverse()
    return [(int(cells[s//4,0]),int(cells[s//4,1]),s%4) for s in chain]


def _actions(states):
    actions=[]
    for a,b in zip(states,states[1:]):
        if a[:2]==b[:2]:
            turn=(b[2]-a[2])%4
            if turn not in (1,3):raise ValueError('unpaid or invalid turn')
            actions.append('right' if turn==1 else 'left')
        else:
            dr,dc=DIRECTIONS[a[2]]
            if b!=(a[0]+dr,a[1]+dc,a[2]):raise ValueError('invalid directed movement')
            actions.append('forward')
    return actions


def augment_candidate_routes(mapper, obs, original_routes, *, max_actions=48, max_extra=4):
    """Return ``(extra_routes, audit)``; never alter original_routes.

    Append the returned routes to the existing pool before shared scoring and
    physical execution. Extra IDs start after the largest existing integer ID.
    The default adds at most four; an empty/insufficient geometric pool is valid.

    Rule: preserve all 3-radius x 8-angle object proposals and their provenance;
    project at most two cells to known reachable safe space; enumerate all four
    camera headings; reject original endpoint duplicates and >budget real round
    trips. Each view is associated with every observed cluster having measured
    support in its actual camera frustum, independently of the proposing cluster.
    Prefer the least represented object, then its least represented relative
    octant; ties use camera-axis/target-center angle, cost, pose and object index.
    Representation starts from original endpoints, counting only clusters whose
    support lies in that endpoint's frustum. It updates after each addition.
    This is geometry diversity, not visibility/semantic/utility maximization.
    """
    c=mapper.config
    if not np.isclose(c.resolution_m,.2):raise ValueError('frozen object model requires 0.2m grid')
    if (not isinstance(max_actions,numbers.Integral) or isinstance(max_actions,bool) or max_actions<1
            or not isinstance(max_extra,numbers.Integral) or isinstance(max_extra,bool) or not 0<=max_extra<=4):
        raise ValueError('positive action budget and 0..4 extra candidates required')
    initial_hash=_hash(original_routes)
    identifiers=[row['candidate_id'] for row in original_routes]
    if any(not isinstance(i,numbers.Integral) or isinstance(i,bool) for i in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('original candidate IDs must be unique integers')
    grid=GridConfig(resolution_m=c.resolution_m,robot_radius_m=c.robot_radius_m,
                    sensor_range_m=c.max_depth_m,sensor_fov_deg=360,max_steps=c.max_steps)
    safe=QualityCoveragePolicy(grid,c,'coverage')._safe(obs)
    start_pose=(*map(int,obs.position),int(obs.heading))
    graph,cells,ids=orientation_graph(safe);start=int(ids[obs.position])*4+obs.heading
    outward,previous=dijkstra(graph,directed=True,indices=start,return_predecessors=True)
    backward,backprevious=dijkstra(graph.T.tocsr(),directed=True,indices=start,return_predecessors=True)
    minimum=outward.reshape(-1,4).min(axis=1);valid=np.isfinite(minimum)
    distance=np.full(safe.shape,-1.);distance[tuple(cells[valid].T)]=minimum[valid]
    reachable=safe&(distance>=0)
    originals=set()
    for row in original_routes:
        raw=np.asarray(row['states'])
        if raw.ndim!=2 or raw.shape[1]!=3 or len(raw)<2 or not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():
            raise ValueError('invalid original route states')
        states=[tuple(map(int,s)) for s in raw]
        if states[0]!=start_pose or states[-1]!=start_pose:raise ValueError('original route must return to the shared initial pose')
        for r,col,h in states:
            if not(0<=r<safe.shape[0] and 0<=col<safe.shape[1] and h in range(4) and safe[r,col]):
                raise ValueError('original route leaves the shared known-safe graph')
        actions=_actions(states)
        if row.get('cost',len(actions))!=len(actions) or len(actions)>max_actions:raise ValueError('original route budget mismatch')
        if 'actions' in row and row['actions']!=actions:raise ValueError('original route action mismatch')
        pose=tuple(map(int,row['pose']))
        arrival=int(row.get('arrival_action',states.index(pose) if pose in states else -1))
        if arrival<0 or arrival>=len(states) or states[arrival]!=pose:raise ValueError('original endpoint absent from route')
        originals.add(pose)
    model=ObjectCompletionModel(mapper,False);support=_observed_support(mapper,model.objects)
    retreat,nearest=distance_transform_edt(~reachable,return_indices=True)
    provenance={}; rejected=[]
    for oi,obj in enumerate(model.objects):
        for radius in (.9,1.4,2.):
            for angle_index in range(8):
                angle=angle_index*np.pi/4;xy=obj['center'][:2]+radius*np.array([np.cos(angle),np.sin(angle)])
                r=safe.shape[0]-1-int(np.floor(xy[1]/.2));col=int(np.floor(xy[0]/.2))
                rec={'object_index':oi,'radius_m':radius,'angle_index':angle_index,'raw_cell':[r,col]}
                if not(0<=r<safe.shape[0] and 0<=col<safe.shape[1]):
                    rejected.append(rec|{'reason':'outside_map'});continue
                if retreat[r,col]>2:
                    rejected.append(rec|{'reason':'retreat_over_2cells','retreat_cells':float(retreat[r,col])});continue
                cell=tuple(map(int,nearest[:,r,col]));rec['retreat_cells']=float(retreat[r,col])
                provenance.setdefault(cell,[]).append(rec)

    def memberships(pose):
        matrix=camera_pose(pose[:2],pose[2],c,safe.shape[0]);result=[]
        for oi,(obj,cloud) in enumerate(zip(model.objects,support)):
            count=_support_in_frustum(cloud,matrix,c)
            if not count:continue
            delta=obj['center'][:2]-matrix[:2,3];norm=np.linalg.norm(delta)
            cosine=float(np.clip(matrix[:2,2]@delta/max(norm,1e-12),-1,1))
            result.append({'object_index':oi,'sector':_sector(matrix[:2,3],obj['center'][:2]),
                           'support_points_in_fov':count,'axis_angle_rad':float(np.arccos(cosine))})
        return result

    counts=np.zeros((len(model.objects),8),int);original_memberships=[]
    def mark(rows):
        for row in rows:counts[row['object_index'],row['sector']]+=1
    for pose in sorted(originals):
        rows=memberships(pose);mark(rows)
        original_memberships.append({'pose':list(pose),'memberships':rows})
    initial_counts=counts.copy();pool=[];heading_audit=[]
    for cell,origins in sorted(provenance.items()):
        for heading in range(4):
            pose=(*cell,heading);state=int(ids[cell])*4+heading
            cost=outward[state]+backward[state]
            rec={'pose':list(pose),'proposal_objects':sorted({r['object_index'] for r in origins})}
            if not np.isfinite(cost) or outward[state]<1 or cost>max_actions:
                heading_audit.append(rec|{'reason':'unreachable_or_outside_paid_round_trip_budget'});continue
            rows=memberships(pose)
            rec.update(cost=int(cost),memberships=rows)
            if pose in originals:
                heading_audit.append(rec|{'reason':'original_endpoint'});continue
            if not rows:
                heading_audit.append(rec|{'reason':'no_observed_cluster_support_in_fov'});continue
            heading_audit.append(rec|{'reason':'eligible'})
            pool.append({'pose':pose,'state':state,'cost':int(cost),'memberships':rows,'provenance':origins})

    extras=[];selections=[]
    while pool and len(extras)<max_extra:
        def key(row):
            values=[]
            for membership in row['memberships']:
                oi,sector=membership['object_index'],membership['sector']
                values.append((int(counts[oi].sum()),int(counts[oi,sector]),membership['axis_angle_rad'],
                               row['cost'],row['pose'],oi,sector))
            return min(values)
        chosen=min(pool,key=key);selection_key=key(chosen);pool.remove(chosen)
        out=_path(previous,start,chosen['state'],cells)
        back=_path(backprevious,start,chosen['state'],cells,reverse=True)
        states=out+back[1:];actions=_actions(states)
        assert states[0]==states[-1]==start_pose and len(actions)==chosen['cost']
        cid=max(identifiers,default=-1)+1+len(extras)
        extra={'candidate_id':int(cid),'group':'object_diversity_v6','pose':list(chosen['pose']),
               'states':[list(s) for s in states],'actions':actions,'cost':len(actions),
               'arrival_action':len(out)-1,'is_coverage_anchor':False,
               'proposal_provenance':chosen['provenance'],'visible_support_memberships':chosen['memberships']}
        extras.append(extra)
        selections.append({'candidate_id':int(cid),'pose':list(chosen['pose']),
                           'selected_object_index':selection_key[-2],'selected_sector':selection_key[-1],
                           'representation_before':[selection_key[0],selection_key[1]],
                           'axis_angle_rad':selection_key[2],'cost':chosen['cost']})
        mark(chosen['memberships'])
    assert _hash(original_routes)==initial_hash
    audit={'version':'v6_geometry_diversity_additions','original_routes_sha256':initial_hash,
           'extra_routes_sha256':_hash(extras),'source_objects':[{'object_index':i,'center':o['center'].tolist(),
               'observed_points':o['observed_points']} for i,o in enumerate(model.objects)],
           'safe_sha256':hashlib.sha256(safe.tobytes()).hexdigest(),'max_actions':int(max_actions),
           'max_extra':int(max_extra),'added':len(extras),'ring_cells':len(provenance),
           'original_endpoint_memberships':original_memberships,'initial_representation_counts':initial_counts.tolist(),
           'final_representation_counts':counts.tolist(),'rejected_ring_proposals':rejected,
           'heading_candidates':heading_audit,'selection_audit':selections,
           'semantics_read_for_selection':False,'predicted_gain_used':False,'GT_used':False,
           'fov_scope':'actual measured support in optical frustum; no occlusion or new-information claim',
           'original_unchanged':True,'new_navigation_executed':False}
    return extras,audit
