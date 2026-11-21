import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).parent;P=Path('/home/ouc/nso_robot/analyses/reproduction_20261104');R=P/'simulation_publication_stage/experiments/unknown_exploration_20261104/source';sys.path.insert(0,str(R))
from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules,ExplorationJointRuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform,SensorPacket
old=json.loads((P/'deployed_policy_matrix/shelf_west_96_S_steps.json').read_text());choices=json.loads((P/'deployed_policy_matrix/shelf_west_96_S_choices.json').read_text());w=create_competition_world('Q0','shelf_west');tr=GridTransform(tuple(w.shape),w.config.resolution_m);f=w.sense();sc=w.scan();p=SensorPacket('Q0_shelf_west','fresh-first-frame','frame-0',0,f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config)
args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode='S',cpu_disable_feedback=False,cpu_max_candidates=12,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='fresh-S');c=NSO_Components(args);c.initialize('cpu',1,*w.shape,*w.shape);c._cpu_backend=ExplorationJointCPUFourModules(args,1,w.shape);c.capabilities=dict(c._cpu_backend.capabilities);rt=ExplorationJointRuntimeIntegration(c,1,w.shape);rt.start_sensor_episode(0,config=w.config,transform=tr,packets=[p],total_budget=96,return_anchor=(*p.position,p.heading),paid_prefix_actions=0)
for row in old[:41]:
 action=rt.next_local_action(0);assert action==row['action'];f,col,done=w.step(action);sc=w.scan();p=SensorPacket('Q0_shelf_west','fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',action,bool(col),bool(done)).validate(tr,w.config);assert p.sha256()==row['packet_sha256'];rt.observe(0,w.step_count,None,None,None,sensor_packet=p)
s=c._cpu_backend.scenes[0];assets=s['assets'];audit=choices[2]['outputs']['score_audit'];out=[]
for i,a in enumerate(assets):
 width,depth,height,vote=map(float,[a['measured_width_m'],a['measured_depth_m'],a['measured_height_m'],a['class_vote']]);marked=int(a['marked_points']);generic=width*height+3*width*depth;semantic=width*height+3*(1-vote)*width*depth if marked else generic
 out.append(dict(index=i,group=int(a['group']),width=width,depth=depth,height=height,class_vote=vote,marked_points=marked,generic=generic,semantic=semantic))
cs=[]
for i in (2,6):
 a=audit[i];factors=a['incremental_aperture'];terms=[dict(index=j,factor=float(v),generic_contribution=float(v*out[j]['generic']),semantic_contribution=float(v*out[j]['semantic'])) for j,v in enumerate(factors)];assert abs(sum(x['generic_contribution'] for x in terms)-a['prior_potentials']['G'])<1e-8 and abs(sum(x['semantic_contribution'] for x in terms)-a['prior_potentials']['S'])<1e-8;cs.append(dict(candidate_id=i,coverage_only_proxy=a['coverage_only_proxy'],factors=terms,prior_G=a['prior_potentials']['G'],prior_S=a['prior_potentials']['S']))
(D/'assets_audit.json').write_text(json.dumps(dict(scope='hash-checked replay of archived 41-action S prefix only; no new continuation',packet_sha256=p.sha256(),assets=out,candidates=cs),indent=2));print(json.dumps(dict(assets=out,candidates=cs),indent=2))
