"""Known-map 3D visibility, separate completion/precision surrogates and routes.

Scores are bounded sensor-derived surrogates, NOT GT F1 or calibrated posteriors.
Regions are spatial bins of observed candidates, never simulator room IDs.
"""
from collections import deque
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import dijkstra
from nso.navigable_frontier_v2 import ProjectedCoveragePolicy
from nso.route_coverage_v2 import orientation_graph,recover_actions
from env.virtual3d import camera_pose
from utils.reconstruction_metrics import ray_scene


class JointPlannerV2(ProjectedCoveragePolicy):
    def __init__(self,grid_config,virtual_config,semantic=True,route=True,quality_weight=1.,semantic_weight=.5):
        self.virtual_config=virtual_config;self.semantic=semantic;self.route=route
        self.quality_weight=quality_weight;self.semantic_weight=semantic_weight
        self.quality=None;self.ray=None
        super().__init__(grid_config,max_candidates=20,bundle=False,path_gain=False)

    def set_mapping(self,mapper):
        # Only the sensor mapper is supplied. No simulator, GT map or evaluator.
        self.mapper=mapper

    def _quality_gain(self,cell,heading):
        q=self.quality
        if q is None:return np.zeros(0),0.,0.
        points=q['point'];pose=camera_pose(cell,heading,self.virtual_config,self.mapper.shape[0])
        local=(points-pose[:3,3])@pose[:3,:3];z=local[:,2]
        c=self.virtual_config;tan=np.tan(np.deg2rad(c.fov_deg/2))
        visible=(z>.2)&(z<c.max_depth_m)&(np.abs(local[:,0])<z*tan)&(np.abs(local[:,1])<z*tan*c.height_px/c.width_px)
        delta=pose[:3,3]-points;distance=np.linalg.norm(delta,axis=1)
        indices=np.flatnonzero(visible)
        if self.ray is not None and len(indices):
            rays=np.column_stack([np.tile(pose[:3,3],(len(indices),1)),-delta[indices]]).astype(np.float32)
            hit=self.ray.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
            # Known reconstructed surfaces block rays. Unknown geometry is not
            # certified visible; it remains optimistic completion prediction.
            visible[indices]&=(hit>=1-.12/np.maximum(distance[indices],.2))
        az=np.arctan2(delta[:,1],delta[:,0]);bins=np.floor((az+np.pi)/(2*np.pi)*8).astype(int)%8
        novel=(q['bits']&(1<<bins))==0
        count=np.array([int(b).bit_count() for b in q['bits']])
        # Direction novelty models completion; approaching with a more favorable
        # incidence/range models precision, with diminishing count contribution.
        completion=novel*np.maximum(0,2-count)/2
        incidence=np.abs(np.sum(q['normal']*delta,axis=1))/np.maximum(distance,.2)
        information=incidence**2/np.maximum(distance**2,.25)
        info_gain=np.clip((information-q['information'])/np.maximum(q['information'],.1),0,2)/2
        uncertainty=np.clip(.25/np.sqrt(q['n'])+q['residual']/.001+q['normal_dispersion']*.25,0,1)
        precision=info_gain*uncertainty
        complexity=1+.5*np.clip(q['normal_dispersion'],0,1)
        # Inspection controls showed that shelves benefit from approaching,
        # whereas flat boxes benefit mostly from a newly exposed side. Use the
        # category as a bounded *view-type* prior, not static region importance.
        # The term vanishes once the better range/incidence has been observed.
        if self.semantic:
            completion+=.5*self.semantic_weight*(q['label']==3)*info_gain
        completion=visible*completion*complexity
        precision=visible*precision*complexity
        gain=.65*completion+.35*precision
        return gain,float(completion.sum()),float(precision.sum())

    def _select(self,obs,safe):
        self.quality=self.mapper.quality_evidence()
        mesh=self.mapper.mesh();self.ray=ray_scene(mesh) if len(mesh.triangles) else None
        graph,cells,ids=orientation_graph(safe)
        start=int(ids[obs.position])*4+obs.heading
        costs,previous=dijkstra(graph,directed=True,indices=start,return_predecessors=True)
        minimum=costs.reshape(-1,4).min(axis=1);reachable=np.isfinite(minimum)
        distances=np.full(safe.shape,-1.);distances[tuple(cells[reachable].T)]=minimum[reachable]
        pool=set(self._candidates(obs,safe,distances))
        inspection=[tuple(map(int,cell)) for cell in cells[reachable] if cell[0]%5==0 and cell[1]%5==0]
        if len(inspection)>20:inspection=[inspection[i] for i in np.linspace(0,len(inspection)-1,20,dtype=int)]
        pool.update(inspection)
        unknown=self._gain_mask(obs);remaining=self.config.max_steps-obs.step
        normalizer=max(1,np.count_nonzero(obs.belief!=1))
        c0=np.count_nonzero(obs.belief==0)/normalizer
        q0=.5
        if self.quality is not None:
            count=np.array([int(b).bit_count() for b in self.quality['bits']])
            q0=float(np.mean(np.minimum(count/2,1)))*.7+.3
        npoints=max(1,0 if self.quality is None else len(self.quality['point']))
        def utility(mask,qgain):
            dc=np.count_nonzero(mask)/normalizer
            # Conservative budget allocation while large portions remain
            # unknown. c0 comes from observed occupancy and declared bounds,
            # never the evaluator's reachable-area denominator.
            coverage_gate=.05+.95*c0**3
            dq=coverage_gate*self.quality_weight*float(qgain.sum())/npoints
            return q0*dc+c0*dq+dc*dq
        candidates=[]
        for cell in sorted(pool):
            # Lidar gain is omnidirectional; compute once for all headings.
            mask=self._visible(obs,cell,0)&unknown
            for heading in range(4):
                state=int(ids[cell])*4+heading;cost=costs[state]
                if not 0<cost<=remaining:continue
                gain,completion,precision=self._quality_gain(cell,heading)
                reward=utility(mask,gain)
                if reward<=1e-10:continue
                candidates.append(dict(cell=cell,heading=heading,state=state,cost=int(cost),mask=mask,
                    gain=gain,reward=reward,completion=completion,precision=precision,
                    region=(cell[0]//20,cell[1]//20),rate=reward/(cost+1)))
        if not candidates:return False
        candidates.sort(key=lambda r:(-r['rate'],r['cost'],r['cell'],r['heading']))
        # Region representatives retain distant targets for route planning.
        reps={}
        for candidate in candidates:
            if candidate['region'] not in reps:reps[candidate['region']]=candidate
        selected=list(reps.values())
        for candidate in candidates:
            if not any(candidate is r for r in selected):selected.append(candidate)
            if len(selected)>=self.max_candidates:break
        candidates=selected[:self.max_candidates]
        between=dijkstra(graph,directed=True,indices=[r['state'] for r in candidates])
        best_index=0;route_indices=[0];best_score=candidates[0]['rate']
        if self.route:
            # Beam search over an ordered budgeted tour, one representative per
            # region first, then refresh online after reaching the first target.
            beams=[]
            for i,r in enumerate(candidates):
                value=r['reward']*(remaining-r['cost'])/max(remaining,1)
                beams.append((value,[i],r['cost'],r['mask'].copy(),r['gain'].copy(),{r['region']}))
            best=max(beams,key=lambda b:b[0])
            for depth in range(1,min(8,len(reps))):
                expansions=[]
                for value,path,elapsed,seen,quality,regions in sorted(beams,key=lambda b:-b[0])[:6]:
                    for j,r in enumerate(candidates):
                        if r['region'] in regions:continue
                        arrival=elapsed+between[path[-1],r['state']]
                        if arrival>=remaining:continue
                        newmask=r['mask']&~seen
                        newquality=np.maximum(0,r['gain']-quality)
                        added=utility(newmask,newquality)*(remaining-arrival)/remaining
                        expansions.append((value+added,path+[j],arrival,seen|r['mask'],np.maximum(quality,r['gain']),regions|{r['region']}))
                if not expansions:break
                beams=sorted(expansions,key=lambda b:-b[0])[:6]
                if beams[0][0]>best[0]:best=beams[0]
            best_score,route_indices=best[0],best[1]
            best_index=route_indices[0]
        else:
            best_index=max(range(len(candidates)),key=lambda i:candidates[i]['rate'])
            best_score=candidates[best_index]['rate'];route_indices=[best_index]
        chosen=candidates[best_index]
        for i,r in enumerate(candidates):
            self.selection_audit.append(dict(step=obs.step,goal=list(r['cell']),heading=r['heading'],
                estimated_actions=r['cost'],coverage_gain_cells=int(r['mask'].sum()),
                completion_proxy=r['completion'],precision_proxy=r['precision'],reward=r['reward'],
                selected=i==best_index,region=list(r['region'])))
        self.selection_audit.append(dict(step=obs.step,region_route=[list(candidates[i]['region']) for i in route_indices],
            view_route=[list(candidates[i]['cell'])+[candidates[i]['heading']] for i in route_indices],route_score=float(best_score)))
        self.goal=chosen['cell'];self.goal_heading=chosen['heading'];self._score=best_score
        self._gain=int(chosen['mask'].sum());self._distance=chosen['cost']
        actions=recover_actions(previous,start,chosen['state'],cells)
        self._actions=deque(actions);self.goal_count+=1
        self._active=dict(start_step=obs.step,goal=list(self.goal),planned_actions=len(actions),predicted_gain=self._gain)
        return True
