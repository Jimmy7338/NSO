#!/usr/bin/env python3
"""Paired next-view counterfactuals from real saved development prefixes.

All alternative scorers receive identical past sensor frames and candidate
poses. Simulator truth is used only AFTER ranking to measure TSDF outcomes.
Endpoint captures isolate sensing; this is not a navigation experiment.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys,zipfile
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d_v2 import VirtualConfigV2,VirtualWorldV2
from env.grid_exploration import GridConfig
from nso.semantic_completion_v3 import SemanticHistoryMapperV3,ObjectCompletionModel
from nso.joint_planner_v2 import JointPlannerV2
from nso.route_coverage_v2 import orientation_graph
from utils.reconstruction_metrics import ReconstructionEvaluator,ray_scene
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.cpu_protocol import file_hash


def task(source,result,index,output):
    config=json.loads((source/'config.json').read_text())
    entry=next(x for x in config['scenes'] if x['seed']==result['seed'] and x['layout']==result['layout'])
    c=VirtualConfigV2(**(config['environment']|entry.get('environment',{})))
    folder=source/result['artifact_dir'];shape=np.load(folder/'maps.npz')['final_belief'].shape
    frames=[RGBDFrame.load(folder/'frames'/f'{i:04d}.npz') for i in range(index+1)]
    scans=[PlanarScan.load(folder/'scans'/f'{i:04d}.npz') for i in range(index+1)]
    def mapped(label_condition='aligned'):
        mapper=SemanticHistoryMapperV3(shape,c,c.truncation_m)
        for frame,scan in zip(frames,scans):
            if label_condition=='shuffled':
                labels=frame.semantic.copy();labels[frame.semantic==2]=3;labels[frame.semantic==3]=2
                frame=replace(frame,semantic=labels)
            if label_condition=='absent':frame=replace(frame,semantic=np.zeros_like(frame.semantic))
            mapper.update(frame,scan)
        return mapper
    mapper=mapped();pose=np.load(folder/'maps.npz')['poses'][index]
    obs=mapper.observation(tuple(pose[:2]),int(pose[2]),index)
    policy=JointPlannerV2(GridConfig(max_steps=c.max_steps),c,semantic=False)
    safe=policy._safe(obs);graph,cells,ids=orientation_graph(safe)
    costs=dijkstra(graph,directed=True,indices=int(ids[obs.position])*4+obs.heading)
    minimum=costs.reshape(-1,4).min(axis=1);distances=np.full(shape,-1.)
    finite=np.isfinite(minimum);distances[tuple(cells[finite].T)]=minimum[finite]
    models={'geometry':ObjectCompletionModel(mapper,False),'semantic':ObjectCompletionModel(mapper,True),
            'shuffled':ObjectCompletionModel(mapped('shuffled'),True),'absent':ObjectCompletionModel(mapped('absent'),True)}
    ray=ray_scene(mapper.mesh());candidates=[]
    for cell in models['geometry'].candidates(safe,distances,limit=20):
        for heading in range(4):
            cost=float(costs[int(ids[cell])*4+heading])
            if not 0<cost<=c.max_steps-index:continue
            scores={name:float(model.gain(cell,heading,ray).sum()) for name,model in models.items()}
            candidates.append(dict(cell=cell,heading=heading,cost=cost,**scores))
    # Union of top alternatives from EVERY scorer plus fixed geometric order.
    # Candidate screening never looks at future outcomes.
    chosen=set(range(0,len(candidates),max(1,len(candidates)//4)))
    for name in models:
        chosen.update(sorted(range(len(candidates)),key=lambda j:(-candidates[j][name],j))[:4])
    name=f"{result['layout']}_{result['seed']}_step{index}"
    out=output/name;out.mkdir()
    world=VirtualWorldV2(c,seed=result['seed'],layout=result['layout'])
    ref=np.load(source/(result['scene']+'_reference.npz'))
    evaluator=ReconstructionEvaluator.__new__(ReconstructionEvaluator)
    evaluator.reference=ref['points'];evaluator.classes=ref['classes'];evaluator.truth=ray_scene(world.mesh)
    before=evaluator.evaluate(mapper.mesh(),1.,(.02,.05))
    outcomes=[]
    for j in sorted(chosen):
        candidate=candidates[j];branch=mapped()
        world.position=tuple(candidate['cell']);world.heading=candidate['heading']
        for k in range(1,5):
            world.step_count=index+k;frame=world.sense();branch.update(frame)
            frame.save(out/f'candidate{j:03d}_frame{k}.npz')
        after=evaluator.evaluate(branch.mesh(),1.,(.02,.05))
        outcomes.append(dict(candidate=j,**candidate,f1_gain=after['f1_05cm']-before['f1_05cm'],
            recall_gain=after['recall_05cm']-before['recall_05cm'],f1_02cm_gain=after['f1_02cm']-before['f1_02cm']))
    summary=dict(scene=name,source=str(source),prefix_frames=len(frames),candidates=len(candidates),evaluated=len(outcomes),scorers={})
    for scorer in models:
        best=max(outcomes,key=lambda x:(x[scorer],-x['candidate']))
        correlation=spearmanr([x[scorer] for x in outcomes],[x['f1_gain'] for x in outcomes]).statistic
        summary['scorers'][scorer]=dict(chosen=best['candidate'],f1_gain=best['f1_gain'],recall_gain=best['recall_gain'],
            ranking=float(correlation) if np.isfinite(correlation) else None,
            regret=max(x['f1_gain'] for x in outcomes)-best['f1_gain'])
    (out/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    (out/'outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    names=[str(p.relative_to(ROOT)) for folder in ('env','nso','utils') for p in (ROOT/folder).glob('*.py')]+['scripts/check_semantic_signal_v3.py']
    signatures={name:file_hash(ROOT/name) for name in names}
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:archive.write(ROOT/name,name)
    records=[json.loads(line) for line in (source/'episodes.jsonl').read_text().splitlines()]
    records=[r for r in records if r['method']=='full_v2']
    summaries=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(task,source,r,step,output) for r in records for step in (80,160)]
        for future in as_completed(futures):
            row=future.result();summaries.append(row);print(json.dumps(row),flush=True)
    assert all(file_hash(ROOT/name)==sha for name,sha in signatures.items()),'sources changed during diagnostic'
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (output/'metadata.json').write_text(json.dumps(dict(scope='development endpoint counterfactuals, not independent navigation evidence',
        files_sha256=signatures,source=str(source.resolve()),prefix_steps=[80,160],new_frames=4),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
