#!/usr/bin/env python3
"""Plot terminal physical C/S records, or redraw their frozen CSV only.

No inference, ROS, policy replay, smoothing or independent ground truth.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract(report,arm):
    result=json.loads((report/'result.json').read_text())
    records=[json.loads(l) for l in (report/'joint/decisions.jsonl').read_text().splitlines()]
    config=next(r['request']['params']['configuration'] for r in records if r['request']['method']=='init')
    assert config['score_mode']==arm and config['total_budget']==40
    obs=[r for r in records if r['request']['method']=='observe' and r['response'].get('ok') and r['response']['result'].get('accepted')]
    assert result.get('status') and obs
    rows=[]
    for i,r in enumerate(obs):
        value=r['response']['result'];feedback=value['summary']['runtime']['modules']['feedback']
        assert value['state']['action_id']==i and feedback['paid_actions']==i
        with np.load(r['request']['params']['path'],allow_pickle=False) as z:
            pose=z['base_odom_pose'].tolist();stamp=float(z['rgbd_timestamp_s'])
        rows.append(dict(arm=arm,paid_action=i,x_m=pose[0],y_m=pose[1],yaw_rad=pose[2],pose_stamp_s=stamp,observation_frame_id=value['state']['frame_id'],observed_area_m2=feedback['observed_2d_area_m2']))
    return rows,{'report':str(report),'status':result['status'],'reason':result.get('reason'),'source_sha256':{k:hashlib.sha256((report/k).read_bytes()).hexdigest() for k in ['result.json','joint/decisions.jsonl']}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--c-report',type=Path);p.add_argument('--s-report',type=Path);p.add_argument('--csv',type=Path,help='Redraw frozen extracted table only; does not revalidate raw extraction');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.csv:
        assert not a.c_report and not a.s_report
        with a.csv.open() as f:rows=list(csv.DictReader(f))
        provenance={'mode':'frozen_csv_redraw','input_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest()}
    else:
        assert a.c_report and a.s_report,'Both terminal reports are required'
        c,cr=extract(a.c_report,'C');s,sr=extract(a.s_report,'S');rows=c+s
        provenance={'mode':'terminal_raw_report_extraction','sources':{'C':cr,'S':sr}}
        with (a.output/'pair_observations.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.75),layout='constrained')
    for arm,color in [('C','#28658c'),('S','#b04a35')]:
        rs=[r for r in rows if r['arm']==arm];paid=[int(r['paid_action']) for r in rs];assert paid==list(range(len(rs)))
        x=[float(r['x_m']) for r in rs];y=[float(r['y_m']) for r in rs];area=[float(r['observed_area_m2']) for r in rs]
        axes[0].plot(x,y,color=color,lw=1.2,label=arm);axes[0].scatter([x[0]],[y[0]],facecolors='none',edgecolors=color,s=40,zorder=4);axes[0].scatter([x[-1]],[y[-1]],color=color,marker='x',s=30,zorder=5)
        axes[1].plot(paid,area,color=color,lw=1.5,label=arm)
    axes[0].set(aspect='equal',title='(a) Recorded observation poses',xlabel='Aligned odom x (m)',ylabel='Aligned odom y (m)');axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(title='(b) Internally observed area',xlabel='Paid action',ylabel='Observed 2D area (m²)',xlim=(0,40));axes[1].set_xticks([0,10,20,30,40]);axes[1].grid(axis='y',alpha=.2);axes[1].legend(frameon=False,fontsize=8)
    for ext in ['pdf','png','svg']:fig.savefig(a.output/('physical_C_S_pair_20261107.'+ext),dpi=230)
    plt.close(fig)
    provenance.update(limits=['One scene, one ordered C-then-S pair; two dependent episodes','Distinct actual bootstraps, not byte-identical initial information','Connected RGBD-time base-odom samples are not a dense or independently ground-truthed trajectory','Area is the internal observed-grid proxy','No cross-bootstrap quality/accuracy claim; no error bars'])
    (a.output/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()
