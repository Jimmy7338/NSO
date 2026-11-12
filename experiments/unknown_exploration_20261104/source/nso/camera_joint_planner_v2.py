"""Joint camera/lidar coverage and surface quality, with observed-region tours.

The F1-shaped objective uses a planar RGB-D footprint and conditional surface
quality as a recall surrogate, with assumed precision 1. It is not evaluator F1.
Both sensors have separate observed masks; all hypothetical gains are deduplicated.
"""
from collections import deque
import numpy as np
from scipy.sparse.csgraph import dijkstra
from nso.joint_planner_v2 import JointPlannerV2
from nso.route_coverage_v2 import orientation_graph,recover_actions
from utils.reconstruction_metrics import ray_scene
from utils.grid_geometry import visible_mask


class CameraJointPlannerV2(JointPlannerV2):
    def _recall_increment(self,qgain,npoints,camera_fraction,coverage_gate,q0):
        dq=coverage_gate*self.quality_weight*float(qgain.sum())/npoints
        return camera_fraction*min(max(0.,1-q0),dq)

    def _inspection_cells(self,safe,distances):
        cells=np.argwhere(safe&(distances>=0))
        candidates=[tuple(map(int,cell)) for cell in cells if cell[0]%5==0 and cell[1]%5==0]
        if len(candidates)>20:candidates=[candidates[i] for i in np.linspace(0,len(candidates)-1,20,dtype=int)]
        return candidates

    def _select(self,obs,safe):
        self.quality=self.mapper.quality_evidence()
        mesh=self.mapper.mesh();self.ray=ray_scene(mesh) if len(mesh.triangles) else None
        graph,cells,ids=orientation_graph(safe)
        start=int(ids[obs.position])*4+obs.heading
        costs,previous=dijkstra(graph,directed=True,indices=start,return_predecessors=True)
        minimum=costs.reshape(-1,4).min(axis=1);reachable=np.isfinite(minimum)
        distances=np.full(safe.shape,-1.);distances[tuple(cells[reachable].T)]=minimum[reachable]
        pool=set(self._candidates(obs,safe,distances))
        inspection=self._inspection_cells(safe,distances)
        pool.update(inspection)
        unknown=self._gain_mask(obs);remaining=self.config.max_steps-obs.step
        normalizer=max(1,np.count_nonzero(obs.belief!=1))
        c0=np.count_nonzero(obs.belief==0)/normalizer
        q0=.5
        camera_fraction=np.count_nonzero(self.mapper.camera_seen & (obs.belief!=1))/normalizer
        if self.quality is not None:
            count=np.array([int(b).bit_count() for b in self.quality['bits']])
            reliability=self.quality['information']/(self.quality['information']+.25+100*self.quality['residual'])
            q0=float(.55*np.mean(np.minimum(count/2,1))+.45*np.mean(reliability))
        npoints=max(1,0 if self.quality is None else len(self.quality['point']))
        def utility(mask,qgain,camera_mask):
            dc=np.count_nonzero(mask)/normalizer
            # Conservative budget allocation while large portions remain
            # unknown. c0 comes from observed occupancy and declared bounds,
            # never the evaluator's reachable-area denominator.
            coverage_gate=.05+.95*c0**3
            dcamera=np.count_nonzero(camera_mask)/normalizer
            r0=camera_fraction*q0
            r1=min(1.,r0+q0*dcamera+self._recall_increment(qgain,npoints,camera_fraction,coverage_gate,q0))
            return min(1.,c0+dc)*2*r1/(1+r1)-c0*2*r0/(1+r0)
        candidates=[]
        for cell in sorted(pool):
            # Lidar gain is omnidirectional; compute once for all headings.
            mask=self._visible(obs,cell,0)&unknown
            for heading in range(4):
                state=int(ids[cell])*4+heading;cost=costs[state]
                if not 0<cost<=remaining:continue
                gain,completion,precision=self._quality_gain(cell,heading)
                camera_mask=visible_mask(obs.belief==1,cell,heading,int(self.config.sensor_range_m/self.config.resolution_m),self.virtual_config.fov_deg)
                camera_mask &= ~self.mapper.camera_seen & (obs.belief!=1)
                reward=utility(mask,gain,camera_mask)
                if reward<=1e-10:continue
                candidates.append(dict(cell=cell,heading=heading,state=state,cost=int(cost),mask=mask,
                    gain=gain,camera_mask=camera_mask,reward=reward,completion=completion,precision=precision,
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
                beams.append((value,[i],r['cost'],r['mask'].copy(),r['gain'].copy(),{r['region']},r['camera_mask'].copy()))
            best=max(beams,key=lambda b:b[0])
            for depth in range(1,min(8,len(reps))):
                expansions=[]
                for value,path,elapsed,seen,quality,regions,camera_seen in sorted(beams,key=lambda b:-b[0])[:6]:
                    for j,r in enumerate(candidates):
                        if r['region'] in regions:continue
                        arrival=elapsed+between[path[-1],r['state']]
                        if arrival>=remaining:continue
                        newmask=r['mask']&~seen
                        newquality=np.maximum(0,r['gain']-quality)
                        merged_seen=seen|r['mask'];merged_quality=np.maximum(quality,r['gain']);merged_camera=camera_seen|r['camera_mask']
                        added=(utility(merged_seen,merged_quality,merged_camera)-utility(seen,quality,camera_seen))*(remaining-arrival)/remaining
                        expansions.append((value+added,path+[j],arrival,seen|r['mask'],np.maximum(quality,r['gain']),regions|{r['region']},merged_camera))
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
                completion_proxy=r['completion'],precision_proxy=r['precision'],camera_gain_cells=int(r['camera_mask'].sum()),reward=r['reward'],
                selected=i==best_index,region=list(r['region'])))
        self.selection_audit.append(dict(step=obs.step,region_route=[list(candidates[i]['region']) for i in route_indices],
            view_route=[list(candidates[i]['cell'])+[candidates[i]['heading']] for i in route_indices],route_score=float(best_score)))
        self.goal=chosen['cell'];self.goal_heading=chosen['heading'];self._score=best_score
        self._gain=int(chosen['mask'].sum());self._distance=chosen['cost']
        actions=recover_actions(previous,start,chosen['state'],cells)
        self._actions=deque(actions);self.goal_count+=1
        self._active=dict(start_step=obs.step,goal=list(self.goal),planned_actions=len(actions),predicted_gain=self._gain)
        return True
