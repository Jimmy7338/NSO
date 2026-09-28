"""Development-only conditional view-response features and ridge ranking.

Geometry and routes are measured inputs. Categories modify conditional response
features, never the measured map, route pool or target definition. Training
targets are supplied separately by an offline development evaluator. Predicted
values are relative within-history ranks, not calibrated areas or uncertainty.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_closing, label
from env.virtual3d import camera_pose


FEATURE_NAMES = ('radar_area_rate', 'camera_area_rate', 'quality_rate', 'inverse_cost',
                 'support_angular_exposure_rate', 'support_angular_novelty_rate',
                 'support_proximity_rate', 'support_opposite_face_rate',
                 'class_angular_exposure_rate', 'class_angular_novelty_rate',
                 'class_proximity_rate', 'class_opposite_face_rate')


def observed_clusters(mapper):
    q=mapper.quality_evidence(max_points=10000)
    if q is None:return []
    points=q['point'];rows=mapper.shape[0]-1-np.floor(points[:,1]/.2).astype(int)
    cols=np.floor(points[:,0]/.2).astype(int)
    inside=(rows>=0)&(rows<mapper.shape[0])&(cols>=0)&(cols<mapper.shape[1])
    occupancy=np.zeros(mapper.shape,bool);occupancy[rows[inside],cols[inside]]=True
    groups,_=label(binary_closing(occupancy,structure=np.ones((3,3))))
    ids=np.zeros(len(points),int);ids[inside]=groups[rows[inside],cols[inside]]
    accepted=[]
    for group in sorted(set(ids)-{0}):
        ix=np.flatnonzero(ids==group);cloud=points[ix]
        lo,hi=np.quantile(cloud,[.05,.95],axis=0);span=hi-lo
        if len(ix)<8 or max(span[:2])>2.2 or span[2]<.25:continue
        accepted.append({'point':cloud,'normal':q['normal'][ix],
                         'bits':q['bits'][ix],'label':q['label'][ix]})
    return accepted


def response_features(mapper,routes,predictions):
    """Return common-route matrices G/O/S/X/M/N using only prefix information.

    Exposure is a descriptor of measured support in a candidate frustum. It is
    deliberately NOT called visible hidden area: no completion mesh or GT is
    consulted. Scene occlusion can invalidate it; this must be learned/tested.
    """
    clusters=observed_clusters(mapper)
    assert len(clusters)==len(predictions['objects']['G'])
    for cluster,obj in zip(clusters,predictions['objects']['G']):
        assert len(cluster['point'])==obj['observed_points']
    states=sorted({tuple(state) for route in routes for state in route['states'][1:]})
    poses={s:camera_pose(s[:2],s[2],mapper.config,mapper.shape[0]) for s in states}
    cache=[]
    for cluster in clusters:
        cloud=cluster['point'];center=np.median(cloud,axis=0)
        inward=cluster['normal'].mean(axis=0);inward/=max(np.linalg.norm(inward),1e-12)
        history=np.array([np.mean((cluster['bits']&(1<<b))!=0) for b in range(8)])
        # A measured voxel support descriptor, not an exterior-area estimator.
        support=len(cloud)*.15**2
        if len(cloud)>128:cloud=cloud[np.linspace(0,len(cloud)-1,128,dtype=int)]
        entries={}
        for state,pose in poses.items():
            local=(cloud-pose[:3,3])@pose[:3,:3];z=local[:,2]
            tangent=np.tan(np.deg2rad(mapper.config.fov_deg/2))
            fov=(z>.2)&(z<mapper.config.max_depth_m)&(np.abs(local[:,0])<z*tangent)
            fov&=np.abs(local[:,1])<z*tangent*mapper.config.height_px/mapper.config.width_px
            exposure=float(np.mean(fov));delta=pose[:3,3]-center;distance=np.linalg.norm(delta)
            az=np.arctan2(delta[1],delta[0]);sector=int(np.floor((az+np.pi)/(2*np.pi)*8))%8
            entries[state]=(sector,exposure,exposure/(1+distance**2),
                            exposure*max(0.,float(delta@inward))/max(distance,.2))
        cache.append((support,history,entries))
    marked=any(np.any((c['label']==2)|(c['label']==3)) for c in clusters)
    marker_domain=(getattr(mapper.config,'semantic_source',None)=='rgb_marker'
                   and getattr(mapper.config,'appearance',None)=='marked')
    output={name:[] for name in ('G','O','S','X','M','N')};descriptors=[]
    by_id={row['candidate_id']:row for row in predictions['candidates']}
    for route in routes:
        row=by_id[route['candidate_id']]['scores']['G'];cost=route['cost']
        base=np.array([row['radar_coverage_m2']/cost,row['camera_coverage_m2']/cost,
                       row['observed_quality_gain_per_point']/cost,1./cost])
        object_features=[]
        for support,history,entries in cache:
            angular=np.zeros(8);near=opposite=0.
            for raw in route['states'][1:]:
                sector,exposure,proximity,back=entries[tuple(raw)]
                angular[sector]=max(angular[sector],exposure)
                near=max(near,proximity);opposite=max(opposite,back)
            object_features.append(support*np.array([angular.sum(),angular@(1-history),near,opposite])/cost)
        descriptors.append([f.tolist() for f in object_features])
        for name in output:
            total=np.zeros(4);conditional=np.zeros(4)
            if name!='N':
                for i,f in enumerate(object_features):
                    known=predictions['objects']['G'][i]['marker_supported']
                    weight=.25 if name in ('O','S','X') and marker_domain and marked and not known else 1.
                    source='G' if name in ('G','O') else name
                    probability=predictions['objects'][source][i]['shelf_probability']
                    total+=weight*f;conditional+=weight*(2*probability-1)*f
            output[name].append(np.r_[base,total,conditional])
    matrices={name:np.asarray(rows,float).reshape(-1,len(FEATURE_NAMES)) for name,rows in output.items()}
    np.testing.assert_array_equal(matrices['G'],matrices['M'])
    return matrices,{'feature_names':list(FEATURE_NAMES),'object_descriptors':descriptors,
                     'observed_clusters':len(clusters),'objectness_weight_unmarked':.25,
                     'descriptor_is_hidden_visibility':False,'target_used_for_features':False}


@dataclass
class RelativeResponseRidge:
    coefficient: np.ndarray
    scale: np.ndarray
    alpha: float

    @classmethod
    def fit(cls,x,y,history_ids,alpha=1.):
        x=np.asarray(x,float);y=np.asarray(y,float);ids=np.asarray(history_ids)
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(ids)!=len(x):raise ValueError('unaligned training arrays')
        if alpha<=0 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid training data')
        xc=np.empty_like(x);yc=np.empty_like(y);weight=np.empty(len(x))
        for h in np.unique(ids):
            mask=ids==h;xc[mask]=x[mask]-x[mask].mean(axis=0)
            yc[mask]=y[mask]-y[mask].mean(axis=0);weight[mask]=1/np.sqrt(mask.sum())
        scale=np.sqrt(np.mean(xc**2,axis=0));scale[scale<1e-8]=1.
        design=xc/scale*weight[:,None];target=yc*weight[:,None]
        coefficient=np.linalg.solve(design.T@design+alpha*np.eye(x.shape[1]),design.T@target)
        return cls(coefficient,scale,float(alpha))

    def predict(self,x):
        x=np.asarray(x,float)
        # Test candidate means use only X, never observed held-out outcomes.
        return (x-x.mean(axis=0))/self.scale@self.coefficient
