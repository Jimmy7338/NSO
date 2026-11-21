#!/usr/bin/env python3
"""Plot one recorded episode; no ROS, inference, planning or GT input."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group();mode.add_argument('--report',type=Path,default=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261107/exploration_resume_S40_01'));mode.add_argument('--bundle',type=Path,help='Published bundle containing source/result.json, decisions.jsonl and frames_manifest.json');ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    root=a.bundle if a.bundle else a.report
    result_path=root/'source/result.json' if a.bundle else root/'result.json'
    decisions_path=root/'decisions.jsonl' if a.bundle else root/'joint/decisions.jsonl'
    result=json.loads(result_path.read_text());records=[json.loads(l) for l in decisions_path.read_text().splitlines()]
    obs=[r for r in records if r['request']['method']=='observe' and r['response'].get('ok') and r['response']['result'].get('accepted')]
    arrivals=result['successful_arrivals'];assert len(obs)==41 and len(arrivals)==40 and all(x['verified'] for x in arrivals)
    if a.bundle:
        manifest=json.loads((root/'frames_manifest.json').read_text());frames=manifest['frames'];assert len(frames)==41
        first=next(f for f in frames if f['action_id']==0)
        assert first['frame_id']==obs[0]['response']['result']['state']['frame_id']
        bootstrap_path=root/first['file'];assert hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()==first['sha256']
    else:
        bootstrap_path=Path(obs[0]['request']['params']['path'])
    with np.load(bootstrap_path,allow_pickle=False) as z:
        if a.bundle:
            for key in ('base_odom_pose','rgbd_timestamp_s'):
                assert hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest()==first['arrays'][key]['sha256']
        bootstrap=np.asarray(z['base_odom_pose']).ravel().tolist();stamp0=float(z['rgbd_timestamp_s'])
    supports=set();data=[]
    for i,r in enumerate(obs):
        d=r['response']['result'];f=d['summary']['runtime']['modules']['feedback'];assert d['state']['action_id']==i and f['paid_actions']==i
        supports.add((f['fixed_old_support_count'],f['fixed_old_support_sha256']))
        pose=bootstrap if i==0 else arrivals[i-1]['after']['pose'];stamp=stamp0 if i==0 else arrivals[i-1]['after']['source_stamp_s']
        data.append(dict(paid_action=i,aligned_odom_x_m=pose[0],aligned_odom_y_m=pose[1],aligned_odom_yaw_rad=pose[2],pose_source_stamp_s=stamp,observation_frame_id=d['state']['frame_id'],observed_area_m2=f['observed_2d_area_m2'],fixed_old_quality_proxy=f['old_quality_proxy']))
    assert len(supports)==1
    with (a.output/'episode_data.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=data[0]);w.writeheader();w.writerows(data)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(9.1,2.65),layout='constrained',gridspec_kw={'width_ratios':[1.05,1,1]})
    x=np.array([d['aligned_odom_x_m'] for d in data]);y=np.array([d['aligned_odom_y_m'] for d in data]);paid=np.arange(41)
    axes[0].plot(x,y,'-',color='#28658c',lw=1.1);axes[0].scatter(x,y,c=paid,cmap='viridis',s=11,zorder=3);axes[0].scatter(x[0],y[0],marker='o',s=55,facecolors='none',edgecolors='black',label='Start',zorder=4);axes[0].scatter(x[-1],y[-1],marker='x',s=35,color='#b04a35',label='End',zorder=5);axes[0].set(aspect='equal',xlabel='Aligned odom x (m)',ylabel='Aligned odom y (m)',title='(a) Recorded endpoints');axes[0].legend(frameon=False,fontsize=7,loc='best')
    for ax,key,color,title,ylabel in [(axes[1],'observed_area_m2','#28658c','(b) Observed area','Observed 2D area (m²)'),(axes[2],'fixed_old_quality_proxy','#b04a35','(c) Fixed-support quality','Old-quality proxy (unitless)')]:
        values=[d[key] for d in data];ax.plot(paid,values,color=color,lw=1.5);ax.scatter([0,40],[values[0],values[-1]],s=16,color=color);ax.set(xlabel='Paid action',ylabel=ylabel,title=title,xlim=(0,40));ax.set_xticks([0,10,20,30,40]);ax.grid(axis='y',alpha=.2,lw=.5);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    for ext in ('pdf','svg','png'):fig.savefig(a.output/f'real_episode_S40_20261107.{ext}',dpi=230)
    plt.close(fig)
    metadata={'episode':str(root),'input_mode':'published_bundle' if a.bundle else 'local_report','independent_episodes':1,'observations':41,'paid_actions':40,'fixed_old_support_count':next(iter(supports))[0],'area_initial_final_m2':[data[0]['observed_area_m2'],data[-1]['observed_area_m2']],'quality_initial_final':[data[0]['fixed_old_quality_proxy'],data[-1]['fixed_old_quality_proxy']],'trajectory':'bootstrap RGBD base_odom_pose plus 40 successful_arrivals.after.pose; straight segments connect sampled endpoints, not a dense path','limits':['Aligned CSM odometry, no external trajectory truth','Area is internal observed-grid proxy, not independently measured coverage','Quality is fixed-support internal proxy, not reconstruction accuracy or F1','One S episode, no matched G comparison or error bars','Camera extrinsics unverified; historical mask disabled'],'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [result_path,decisions_path]},'bootstrap_file_sha256':hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()}
    (a.output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
