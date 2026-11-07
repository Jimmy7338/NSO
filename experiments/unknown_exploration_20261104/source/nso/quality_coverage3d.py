"""Joint coverage/inspection prototype using only mapped sensor evidence.

Angular observation deficit is a proxy, not measured 3D error or a calibrated
uncertainty posterior. Semantic weighting is bounded and has no effect once the
view deficit is exhausted. All variants share navigation and reconstruction.
"""
from collections import deque
import numpy as np
from scipy.sparse.csgraph import dijkstra
from nso.navigable_frontier_v2 import ProjectedCoveragePolicy
from nso.route_coverage_v2 import orientation_graph,recover_actions
from env.virtual3d import camera_pose

METHODS=('coverage','geometric_quality','semantic_static','semantic_quality')


class QualityCoveragePolicy(ProjectedCoveragePolicy):
    def __init__(self,grid_config,virtual_config,method='coverage',quality_weight=1.,max_candidates=16):
        if method not in METHODS or quality_weight<0:raise ValueError('invalid method')
        self.virtual_config=virtual_config;self.method=method;self.quality_weight=quality_weight
        self.surface=(np.empty((0,3)),np.empty(0,int),np.empty(0,int))
        super().__init__(grid_config,max_candidates,bundle=False,path_gain=False)

    def set_surface_evidence(self,evidence):
        points,bits,labels=evidence
        keep=(points[:,2]>.1)&(points[:,2]<1.8)
        points,bits,labels=points[keep],bits[keep],labels[keep]
        if len(points)>1800:
            indices=np.linspace(0,len(points)-1,1800,dtype=int)
            points,bits,labels=points[indices],bits[indices],labels[indices]
        self.surface=points.copy(),bits.copy(),labels.copy()

    def _select(self,obs,safe):
        graph,cells,ids=orientation_graph(safe)
        start=int(ids[obs.position])*4+obs.heading
        costs,previous=dijkstra(graph,directed=True,indices=start,return_predecessors=True)
        minimum=costs.reshape(-1,4).min(axis=1)
        distances=np.full(safe.shape,-1.)
        reachable=np.isfinite(minimum)
        distances[tuple(cells[reachable].T)]=minimum[reachable]
        pool=set(self._candidates(obs,safe,distances))
        if self.method!='coverage':
            # Add inspection positions throughout observed reachable space.
            # No oracle object centers, room labels or GT visibility are used.
            for r,c in cells[reachable]:
                if r%4==0 and c%4==0:pool.add((int(r),int(c)))
        pool=sorted(pool,key=lambda p:(distances[p],p))
        if len(pool)>48:
            pool=[pool[i] for i in np.linspace(0,len(pool)-1,48,dtype=int)]
        eligible=self._gain_mask(obs);points,bits,labels=self.surface
        if len(points):
            pc=np.floor(points[:,0]/self.config.resolution_m).astype(int)
            pr=safe.shape[0]-1-np.floor(points[:,1]/self.config.resolution_m).astype(int)
            inside=(pr>=0)&(pr<safe.shape[0])&(pc>=0)&(pc<safe.shape[1])
            pr=np.clip(pr,0,safe.shape[0]-1);pc=np.clip(pc,0,safe.shape[1]-1)
            counts=np.asarray([int(b).bit_count() for b in bits])
        candidates=[];remaining=self.config.max_steps-obs.step
        for cell in pool:
            for heading in range(4):
                state=int(ids[cell])*4+heading;cost=costs[state]
                if not 0<cost<=remaining:continue
                view=self._visible(obs,cell,heading);mask=view&eligible
                quality=np.zeros(len(points),float)
                if len(points) and self.method!='coverage':
                    pose=camera_pose(cell,heading,self.virtual_config,safe.shape[0])
                    local=(points-pose[:3,3])@pose[:3,:3]
                    z=local[:,2];fx=self.virtual_config.width_px/(2*np.tan(np.deg2rad(self.virtual_config.fov_deg/2)))
                    visible=inside&view[pr,pc]&(z>.15)&(z<self.virtual_config.max_depth_m)
                    visible&=(np.abs(local[:,0])<z*self.virtual_config.width_px/(2*fx))
                    visible&=(np.abs(local[:,1])<z*self.virtual_config.height_px/(2*fx))
                    az=np.arctan2(pose[1,3]-points[:,1],pose[0,3]-points[:,0])
                    viewbin=np.floor((az+np.pi)/(2*np.pi)*8).astype(int)%8
                    novel=(bits&(1<<viewbin))==0
                    if self.method=='semantic_static':quality=visible*(labels==3)*.15**2
                    else:
                        quality=visible*novel*np.maximum(0,3-counts)/3*.15**2
                        if self.method=='semantic_quality':quality*=1+.75*(labels==3)
                reward=float(mask.sum()*self.config.resolution_m**2+self.quality_weight*quality.sum())
                if reward<=0:continue
                candidates.append(dict(cell=cell,heading=heading,state=state,cost=int(cost),
                    mask=mask,quality=quality,reward=reward,score=reward/(cost+1)))
        candidates.sort(key=lambda c:(-c['score'],c['cell'],c['heading']))
        candidates=candidates[:self.max_candidates]
        if not candidates:return False
        between=dijkstra(graph,directed=True,indices=[c['state'] for c in candidates])
        horizon=min(32,remaining)
        rows=[]
        for i,first in enumerate(candidates):
            # Transfers outside the lookahead retain a positive rate score.
            value=first['reward']/(first['cost']+1)
            successor=None
            if first['cost']<horizon:
                best=first['reward']*(horizon-first['cost'])/horizon
                for j,second in enumerate(candidates):
                    arrival=first['cost']+between[i,second['state']]
                    if i==j or arrival>=horizon:continue
                    gain=float(np.count_nonzero(second['mask']&~first['mask'])*self.config.resolution_m**2)
                    # Conservative: a surface contributes at most once in this
                    # two-view plan, even if a second direction could help again.
                    gain+=self.quality_weight*float(second['quality'][first['quality']==0].sum())
                    reward=(first['reward']*(horizon-first['cost'])+gain*(horizon-arrival))/horizon
                    if reward>best:best,successor=reward,j
                value=best
            rows.append(dict(step=obs.step,goal=list(first['cell']),heading=first['heading'],
                estimated_actions=first['cost'],coverage_gain_m2=float(first['mask'].sum()*self.config.resolution_m**2),
                quality_proxy=float(first['quality'].sum()),score=float(value),
                next_goal=list(candidates[successor]['cell']) if successor is not None else None,selected=False))
        index=min(range(len(candidates)),key=lambda i:(-rows[i]['score'],candidates[i]['cost'],i))
        chosen=candidates[index];rows[index]['selected']=True;self.selection_audit.extend(rows)
        self.goal=chosen['cell'];self.goal_heading=chosen['heading'];self._score=rows[index]['score']
        self._gain=int(chosen['mask'].sum());self._distance=chosen['cost']
        actions=recover_actions(previous,start,chosen['state'],cells)
        self._actions=deque(actions);self.goal_count+=1
        self._active=dict(start_step=obs.step,goal=list(self.goal),planned_actions=len(actions),predicted_gain=self._gain)
        return True
