#!/usr/bin/env python3
"""Read sealed trajectories and draw article figures; no simulation or rescoring.

The complete four V36 pairs and six reviewed nominal multi-instance G/S pairs
are included. Display geometry is offline context only. No old input is edited.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-article-trajectories-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/thesis/figures/article_trajectories_20260928'
LOCAL = Path('audit_results/v36_online_confirmation_20260918')
LARGE = Path('audit_results/semantic_development_acquisition_20260923')
ASSETS = Path('audit_results/semantic_scene_assets_20260923')
MAX_BYTES = 12 * 1024**2
RESERVE_BYTES = 100 * 1024**2
COLORS = {'G': '#0072B2', 'S': '#D55E00'}
ACTION_COLORS = ['#BDD5E2', '#4E5963', '#D9A157']
SOURCES = {}
CHECKS = []
TRAJECTORIES = []
SUMMARIES = []
PAIRS = []
OUTPUTS = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def read_bytes(path, expected=None):
    path = ROOT / path
    data = path.read_bytes()
    value = sha(data)
    if expected is not None and value != expected:
        raise ValueError(f'Sealed input hash differs: {path}')
    SOURCES[relative(path)] = {'sha256': value, 'bytes': len(data),
                               'existing_seal_verified': expected is not None}
    return data


def read_json(path, expected=None):
    data = read_bytes(path, expected)
    if str(path).endswith('.gz'):
        data = gzip.decompress(data)
    return json.loads(data)


def check(condition, message):
    if not condition:
        raise ValueError(message)
    CHECKS.append(message)


def rotate_xy(points, turns):
    value = np.asarray(points, dtype=float).copy()
    for _ in range(turns):
        x, y = value[..., 0].copy(), value[..., 1].copy()
        value[..., 0], value[..., 1] = y, -x
    return value


def device_xy(points, parent):
    return rotate_xy(np.asarray(points) - parent['device_frame']['origin_xyz'][:2],
                     parent['device_frame']['quarter_turns_ccw'])


def normalize_action(action):
    return {'turn_left': 'left', 'turn_right': 'right',
            'initial_observation': 'initial', None: 'initial'}.get(action, action)


def summarize(record):
    rows = record['rows']
    p = np.asarray([[r['source_x_m'], r['source_y_m']] for r in rows])
    headings = np.asarray([[r['heading_dx'], r['heading_dy']] for r in rows])
    distances = np.linalg.norm(np.diff(p, axis=0), axis=1)
    actions = [r['action'] for r in rows[1:]]
    check(len(rows) == rows[-1]['paid_step'] + 1, record['run_id'] + ': contiguous paid records')
    check(all(a in ('forward', 'left', 'right', 'observe') for a in actions),
          record['run_id'] + ': supported executed primitive actions only')
    check(all(d < 1e-8 for d, a in zip(distances, actions) if a != 'forward'),
          record['run_id'] + ': turns and observes do not translate')
    check(all(d > 1e-8 for d, a in zip(distances, actions) if a == 'forward'),
          record['run_id'] + ': every saved forward action translates')
    check(np.allclose(p[0], p[-1], atol=1e-8) and np.allclose(headings[0], headings[-1], atol=1e-8),
          record['run_id'] + ': returned to initial position and heading')
    turns = sum(a in ('left', 'right') for a in actions)
    observations = actions.count('observe')
    forward = actions.count('forward')
    check(forward + turns + observations == rows[-1]['paid_step'],
          record['run_id'] + ': primitive cost decomposition closes')
    cumulative = 0.
    for index, row in enumerate(rows):
        if index:
            cumulative += float(distances[index-1])
        row['cumulative_translation_m'] = cumulative
    valid_slack = [r['return_slack_actions'] for r in rows if r['return_slack_actions'] is not None]
    check(all(s >= 0 for s in valid_slack), record['run_id'] + ': recorded return reserve is nonnegative')
    output = dict(cohort=record['cohort'], parent=record['parent'], configuration=record['configuration'],
        method=record['method'], run_id=record['run_id'], budget=record['budget'],
        paid_actions=rows[-1]['paid_step'], translation_m=round(float(distances.sum()), 9),
        forward_actions=forward, turn_actions=turns, left_turn_actions=actions.count('left'),
        right_turn_actions=actions.count('right'), explicit_observe_actions=observations,
        saved_sensor_frames=len(rows), returned_xy_and_heading=True,
        unused_budget_at_return=record['budget']-rows[-1]['paid_step'],
        minimum_recorded_return_slack_actions=min(valid_slack) if valid_slack else None,
        return_slack_record_count=len(valid_slack), return_slack_scope=record['slack_scope'],
        metric_name=record['metric_name'], saved_joint_metric=record['metric'],
        first_geometry_feedback_step=record.get('first_feedback'),
        selected_diagnostic_decision_steps=';'.join(str(s) for s in record.get('diagnostic_steps', [])))
    record['summary'] = output
    SUMMARIES.append(output)
    TRAJECTORIES.extend(rows)


def row_base(record, step, action, xy, display, direction, source_path, slack):
    return dict(cohort=record['cohort'], parent=record['parent'], configuration=record['configuration'],
        method=record['method'], run_id=record['run_id'], paid_step=step, action=normalize_action(action),
        source_x_m=float(xy[0]), source_y_m=float(xy[1]), display_x_m=float(display[0]),
        display_y_m=float(display[1]), heading_dx=float(direction[0]), heading_dy=float(direction[1]),
        return_slack_actions=slack, source_file=str(source_path))


def load_local():
    config = read_json(LOCAL/'config.json')
    scene = read_json('configs/virtual3d/v33_direction_scene_r1_20260917.json')
    parents = {p['id']: p for p in scene['parents']}
    records = {}
    for cell in config['physical_cases']:
        parent, hypothesis, mode = cell['parent'], cell['hypothesis'], cell['mode']
        folder = LOCAL/f"case{cell['index']:02d}"
        seal = read_json(folder/'main_seal.json')
        trace = read_json(folder/'trace.json', seal['trace.json'])
        controller = read_json(folder/'controller.json', seal['controller.json'])
        result = read_json(folder/'result.json', seal['result.json'])
        check(result['physical_case'] == cell, str(folder)+': matching physical case')
        states = {r['step']: r for r in controller['calls'] if r['operation']=='update_public_topology_state'}
        feedback = [r['step'] for r in controller['calls'] if r['module']=='IGCR'
                    and abs(r.get('geometry_applied_log_odds', 0)) > 0]
        record = dict(cohort='V36_local_confirmation', parent=parent, configuration=f'h{hypothesis}',
            method=mode, run_id=f'V36_case{cell["index"]:02d}', budget=42, rows=[],
            metric_name='J5', metric=result['stages']['final']['measurement']['05cm']['joint'],
            first_feedback=min(feedback) if feedback else None,
            slack_scope='remaining budget minus saved exact-return distance at current state')
        for t in trace:
            xy = t['pose_v33'][:2]
            direction = [(0.,1.),(1.,0.),(0.,-1.),(-1.,0.)][t['pose_v33'][2]]
            direction = rotate_xy(direction, parents[parent]['device_frame']['quarter_turns_ccw'])
            s = states[t['paid']]
            record['rows'].append(row_base(record, t['paid'], t['action'], xy,
                device_xy(xy, parents[parent]), direction, folder/'trace.json',
                s['remaining_budget']-s['return_distance']))
        check(len(trace)==43 and trace[0]['pose_v33']==trace[18]['pose_v33']==trace[42]['pose_v33'],
              record['run_id']+': declared common prefix and 42-action exact return')
        summarize(record)
        records[parent, hypothesis, mode] = record
    return parents, records


def load_large():
    progress = read_json(LARGE/'progress_summary_010.json',
        'ce26d8cc42464af8ebb981199dd48200e51d8c7588175f07c761624acbadfaff')
    design = read_json(ASSETS/'design.json')
    parents = {p['parent_id']: p for p in design['parents']}
    assets = read_json(ASSETS/'manifest.json')
    records, geometries = {}, {}
    for index in range(6):
        parent = f'SEM_P{index:02d}'
        asset = f'{parent}__nominal_relationship'
        geometry_path = ASSETS/asset/'renderer_private/geometry.npz'
        body = read_bytes(geometry_path, assets['artifact_sha256'][f'{asset}/renderer_private/geometry.npz'])
        with np.load(io.BytesIO(body), allow_pickle=False) as arrays:
            geometries[parent] = {key: arrays[key].copy() for key in ('vertices','triangles','triangle_instance_id')}
        for mode in ('G', 'S'):
            run_id = f'core_P{index:02d}_nom_{mode}_b120_lexicographic'
            saved = next(r for r in progress['episodes'] if r['run_id']==run_id)
            check(saved['complete_episode'] and saved['review']['saved_policy_replay_verified']
                  and saved['returned_xy_and_yaw'], run_id+': complete previously reviewed returned trajectory')
            folder = LARGE/'episodes'/run_id
            manifest = read_json(folder/'artifact_manifest.json', saved['episode_manifest_sha256'])
            review = saved['review']
            read_bytes(review['path'], review['sha256'])
            result = read_json(folder/'result.json', manifest['files']['result.json']['sha256'])
            actions = result['actions']
            check(len(actions)==saved['executed_paid_actions'], run_id+': paid count agrees with summary')
            record = dict(cohort='six_parent_nominal_development', parent=parent, configuration='nominal',
                method=mode, run_id=run_id, budget=saved['budget'], rows=[],
                metric_name='J_nav', metric=saved['J_nav'], diagnostic_steps=[],
                first_feedback=saved.get('first_events',{}).get('informative_paid_geometry_residual_step'),
                slack_scope='remaining budget minus next paid action and saved return cost after that action')
            for step in range(len(actions)+1):
                path = folder/'steps'/f'{step:03d}.json.gz'
                d = read_json(path, manifest['files'][f'steps/{step:03d}.json.gz']['sha256'])
                check(d['accounting']['paid_step']==step, run_id+f': frame {step} has matching step')
                if step:
                    check(d['accounting']['observation_sha256']==actions[step-1]['observation_sha256']
                          and normalize_action(d['accounting']['action'])==normalize_action(actions[step-1]['controller_action']),
                          run_id+f': frame {step} matches executed action receipt')
                camera = np.asarray(d['mapper']['world_from_camera'], dtype=float)
                xy, direction = camera[:2,3], camera[:2,2]
                decision = d['decision']
                returning = decision.get('routing',{}).get('return_cost_after_action')
                slack = saved['budget']-step-1-returning if returning is not None else None
                selected = (decision.get('global_selection') or {}).get('selected') or {}
                if decision.get('global_replanned') and selected.get('kind')=='diagnose_then_observe':
                    record['diagnostic_steps'].append(step)
                record['rows'].append(row_base(record, step, d['accounting']['action'], xy, xy,
                    direction, path, slack))
            summarize(record)
            records[parent, mode] = record
    return parents, geometries, records


def pairing(g, s):
    ga, sa = g['rows'], s['rows']
    differing = [i for i,(a,b) in enumerate(zip(ga,sa)) if a['action']!=b['action'] or
                 not np.allclose([a['source_x_m'],a['source_y_m'],a['heading_dx'],a['heading_dy']],
                                 [b['source_x_m'],b['source_y_m'],b['heading_dx'],b['heading_dy']],atol=1e-8)]
    first = differing[0] if differing else None
    row = dict(cohort=g['cohort'], parent=g['parent'], configuration=g['configuration'],
        geometry_run=g['run_id'], semantic_run=s['run_id'], first_executed_divergence_step=first,
        identical_saved_actions_and_poses=not differing and len(ga)==len(sa),
        metric_name=g['metric_name'], geometry_metric=g['metric'], semantic_metric=s['metric'],
        saved_metric_delta=s['metric']-g['metric'],
        translation_delta_m=s['summary']['translation_m']-g['summary']['translation_m'])
    PAIRS.append(row)
    return first


def xy(record):
    return np.asarray([[r['display_x_m'],r['display_y_m']] for r in record['rows']])


def path_layer(ax, record, start=0, direction_interval=12):
    points = xy(record)
    mode, color = record['method'], COLORS[record['method']]
    ax.plot(points[start:,0],points[start:,1],color=color,ls='--' if mode=='G' else '-',
            lw=1.5 if mode=='G' else 2.2,zorder=7 if mode=='G' else 6)
    turn_points = sorted({tuple(points[i]) for i,r in enumerate(record['rows'])
                          if i>=start and r['action'] in ('left','right')})
    if turn_points:
        t = np.asarray(turn_points)
        ax.scatter(t[:,0],t[:,1],s=23 if mode=='G' else 39,facecolors='none',edgecolors=color,lw=.7,zorder=8)
    observations = sorted({tuple(points[i]) for i,r in enumerate(record['rows']) if i>=start and r['action']=='observe'})
    if observations:
        t = np.asarray(observations)
        ax.scatter(t[:,0],t[:,1],marker='s',s=10,color=color,zorder=9)
    for i in range(start, len(points), direction_interval):
        r=record['rows'][i]
        delta=.30*np.asarray([r['heading_dx'],r['heading_dy']])
        ax.annotate('',xy=points[i]+delta,xytext=points[i],
            arrowprops=dict(arrowstyle='-|>',color=color,lw=.85,mutation_scale=7),zorder=10)


def map_style(ax, limits, labels=('x (m)','y (m)')):
    ax.set(xlim=limits[0],ylim=limits[1],aspect='equal',xlabel=labels[0],ylabel=labels[1])
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,labelsize=7)
    ax.grid(color='#EDF0F2',lw=.55,zorder=0)
    ax.set_axisbelow(True)


def action_strip(ax, g, s, divergence, prefix=0):
    matrix=[]
    for record in (g,s):
        matrix.append([0 if r['action']=='forward' else 2 if r['action']=='observe' else 1
                       for r in record['rows'][1:]])
    width=max(len(a) for a in matrix)
    check(all(len(a)==width for a in matrix),g['run_id']+': paired action-strip lengths agree')
    ax.imshow(matrix,cmap=ListedColormap(ACTION_COLORS),vmin=0,vmax=2,aspect='auto',interpolation='nearest',
              extent=(.5,width+.5,1.5,-.5))
    if width < g['budget']:
        ax.add_patch(Rectangle((width+.5,-.5),g['budget']-width,2,
            facecolor='#FAFAFA',edgecolor='#CFD4D7',hatch='////',lw=.5))
    ax.set_xlim(.5,g['budget']+.5)
    if prefix:
        ax.axvline(prefix+.5,color='white',lw=1.4)
    if divergence is not None:
        ax.axvline(divergence,color='#7B3294',lw=1.0)
    ax.set_yticks([0,1],['G','S'])
    ax.set_xticks([1,prefix or 40,g['budget']] if prefix else [1,40,80,g['budget']])
    ax.tick_params(axis='both',length=0,labelsize=6.5,pad=3)
    ax.set_xlabel('Executed paid action',fontsize=6.7,labelpad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def common_legend(fig, ypos, large=False):
    handles=[Line2D([],[],color=COLORS['G'],ls='--',lw=1.5,label='G: geometry'),
             Line2D([],[],color=COLORS['S'],lw=2.2,label='S: semantic'),
             Line2D([],[],marker='D',color='#22323D',lw=0,ms=5,label='Start / return'),
             Line2D([],[],marker='o',color='#4E5963',mfc='none',lw=0,ms=5,label='In-place turn'),
             Patch(fc=ACTION_COLORS[0],label='Translate'),Patch(fc=ACTION_COLORS[1],label='Turn'),
             Patch(fc=ACTION_COLORS[2],label='Paid observe')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,ypos),ncols=4,frameon=False,
               fontsize=7,handlelength=1.8,columnspacing=1.3,labelspacing=.7)


def save(fig, name):
    for extension in ('pdf','svg','png'):
        path=OUT/f'{name}.{extension}'
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None}
        fig.savefig(path,dpi=260,facecolor='white',metadata=metadata)
        OUTPUTS.append(path)
    plt.close(fig)


def local_figure(parents, records):
    fig=plt.figure(figsize=(8.25,10.2))
    outer=fig.add_gridspec(2,2,left=.075,right=.985,bottom=.15,top=.905,hspace=.35,wspace=.20)
    for index,(p,h) in enumerate(itertools.product(('P00','P01'),(0,1))):
        grid=outer[index//2,index%2].subgridspec(2,1,height_ratios=(8,1),hspace=.28)
        ax=fig.add_subplot(grid[0]); strip=fig.add_subplot(grid[1])
        parent=parents[p];g,s=records[p,h,'G'],records[p,h,'S']
        nav=device_xy(parent['nav_cells'],parent)
        ax.scatter(nav[:,0],nav[:,1],s=4,color='#CCD3D7',zorder=1)
        for bi,box in enumerate(parent['hypotheses'][h]['assets'][0]['boxes']):
            x0,x1,y0,y1,_,_=box
            corners=device_xy(list(itertools.product((x0,x1),(y0,y1))),parent)
            lo,hi=corners.min(axis=0),corners.max(axis=0)
            ax.add_patch(Rectangle(lo,*(hi-lo),facecolor='#E8DEC9' if bi==0 else '#C0CFD3',
                edgecolor='#7D8C92',lw=.6,hatch='////' if bi==0 else None,zorder=2))
        prefix=xy(g)[:19]
        ax.plot(prefix[:,0],prefix[:,1],color='#DEE2E5',lw=5.5,zorder=3)
        path_layer(ax,s,18,6);path_layer(ax,g,18,6)
        anchor=xy(g)[0];ax.scatter(*anchor,marker='D',s=28,color='#22323D',zorder=12)
        ax.annotate('Start / return',anchor,xytext=(0,-18),textcoords='offset points',ha='center',fontsize=7)
        first=pairing(g,s)
        check(first==(19 if h else None),f'{p}/h{h}: expected full executed-pose pairing')
        if first:
            for record in (g,s):
                r=record['rows'][first]
                target=anchor+.72*np.asarray([r['heading_dx'],r['heading_dy']])
                ax.annotate('',target,anchor,arrowprops=dict(arrowstyle='-|>',lw=1.8,color=COLORS[record['method']]),zorder=14)
            ax.text(.02,.97,'First different action: 19',transform=ax.transAxes,va='top',fontsize=7.1,
                    bbox=dict(facecolor='white',edgecolor='none',pad=2))
        else:
            ax.text(.02,.97,'Identical G/S trajectory',transform=ax.transAxes,va='top',fontsize=7.1,
                    bbox=dict(facecolor='white',edgecolor='none',pad=2))
        for record in (g,s):
            f=record['first_feedback'];point=xy(record)[f]
            ax.scatter(*point,marker='D',s=33,facecolors='white',edgecolors=COLORS[record['method']],lw=1,zorder=12)
        ax.text(.02,.02,f"First residual update: G {g['first_feedback']}, S {s['first_feedback']}",
                transform=ax.transAxes,fontsize=6.6,bbox=dict(facecolor='white',edgecolor='none',pad=1.7))
        delta=s['metric']-g['metric']
        ax.set_title(f'({chr(97+index)}) {p} / h{h}     '+r'$\Delta J_5$'+f' = {delta:+.4f}',loc='left',fontsize=9.3,fontweight='bold',pad=7)
        map_style(ax,((-3.65,3.65),(-1.2,6.1)),('Equipment u (m)','Equipment v (m)'))
        action_strip(strip,g,s,first,prefix=18)
        summary=g['summary']; ss=s['summary']
        strip.text(.5,-1.45,f"G / S: {summary['translation_m']:.0f} / {ss['translation_m']:.0f} m  ·  "
            f"{summary['turn_actions']} / {ss['turn_actions']} turns  ·  43 / 43 sensor frames",
            transform=strip.transAxes,ha='center',fontsize=6.8)
    fig.text(.075,.966,'Where the observation route changes',fontsize=14,fontweight='bold',color='#23333C')
    fig.text(.075,.938,'Four complete V36 pairs · shared 18-action prefix · 42 paid actions · identical metric scale',fontsize=8.3,color='#52606A')
    common_legend(fig,.035)
    fig.text(.075,.014,'P01 uses its declared equipment coordinates. Open diamonds: first nonzero applied geometry residual.\n'
        'Grey path: shared prefix. Arrows: recorded camera heading. All moves and turns also appear in the action strips.',fontsize=6.7,color='#52606A')
    save(fig,'local_paired_trajectories')


def scene_figure(parents, geometries, records):
    names=['Two rooms','Switchback corridor','Central island','Two dead-end branches','Parallel aisles','Open gallery']
    fig=plt.figure(figsize=(11.0,9.9))
    outer=fig.add_gridspec(2,3,left=.055,right=.985,bottom=.16,top=.90,hspace=.38,wspace=.20)
    for index in range(6):
        grid=outer[index//3,index%3].subgridspec(2,1,height_ratios=(8,1),hspace=.29)
        ax=fig.add_subplot(grid[0]);strip=fig.add_subplot(grid[1])
        p=f'SEM_P{index:02d}';geo=geometries[p];g,s=records[p,'G'],records[p,'S']
        facets=geo['vertices'][geo['triangles']]
        mask=facets[:,:,2].max(axis=1)>.2
        ids=geo['triangle_instance_id'][mask]
        ax.add_collection(PolyCollection(facets[mask,:,:2],facecolors=np.where((ids<0)[:,None],
            np.array([[.85,.87,.89,1.]]),np.array([[.73,.79,.81,1.]])),edgecolors='none',zorder=2))
        bounds=parents[p]['room_size_m']
        ax.add_patch(Rectangle((0,0),*bounds,fill=False,edgecolor='#77858C',lw=.65,zorder=3))
        path_layer(ax,s,0,18);path_layer(ax,g,0,18)
        ax.scatter(*xy(g)[0],marker='D',s=25,color='#22323D',zorder=13)
        first=pairing(g,s)
        check(first==(71 if index==2 else None),p+': complete nominal trajectory comparison agrees with frozen report')
        if index==2:
            check(70 in s['diagnostic_steps'] and s['rows'][71]['action']=='right' and s['rows'][72]['action']=='observe',
                  p+': paid diagnosis 70 -> turn 71 -> observe 72 verified')
            point=xy(s)[71]
            ax.scatter(*point,marker='o',s=62,facecolors='none',edgecolors='#7B3294',lw=1.1,zorder=14)
            ax.annotate('71: first divergence\n70 → diagnosis; 72 → observe',point,xytext=(4.4,8.6),
                fontsize=6.6,ha='center',arrowprops=dict(arrowstyle='-',color='#7B3294',lw=.8),
                bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=15)
        descriptor='same saved path' if first is None else f'first divergence at {first}'
        ax.set_title(f'({chr(97+index)}) P{index:02d} · {names[index]}',loc='left',fontsize=8.9,fontweight='bold',pad=7)
        ax.text(.02,.975,descriptor,transform=ax.transAxes,va='top',fontsize=6.9,
            bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=16)
        ax.text(.02,.025,r'$\Delta J_{\mathrm{nav}}$'+f" = {s['metric']-g['metric']:+.4f}",transform=ax.transAxes,
            fontsize=7.1,bbox=dict(facecolor='white',edgecolor='none',pad=1.8),zorder=16)
        map_style(ax,((-.25,10.5),(-.25,10.5)))
        ax.set_xticks([0,5,10]);ax.set_yticks([0,5,10])
        action_strip(strip,g,s,first)
        gs,ss=g['summary'],s['summary']
        strip.text(.5,-1.7,f"G / S: {gs['translation_m']:.2f} / {ss['translation_m']:.2f} m\n"
            f"{gs['turn_actions']} / {ss['turn_actions']} turns; {gs['explicit_observe_actions']} / {ss['explicit_observe_actions']} observes",
            transform=strip.transAxes,ha='center',fontsize=6.8,linespacing=1.3)
    fig.text(.055,.966,'Scene-scale execution across all six development layouts',fontsize=14,fontweight='bold',color='#23333C')
    fig.text(.055,.939,'Saved nominal G/S runs · budget of 120 actions · provided navigation graph · exact simulated poses',fontsize=8.4,color='#52606A')
    common_legend(fig,.040,large=True)
    fig.text(.055,.014,'Five pairs have identical routes; P02 differs and has a lower semantic endpoint. This figure documents execution and its limits.\n'
        'Common 10.75 m square window. P04 returns after 115 actions (hatched: unused); other runs use 120. Grey: offline geometry.',fontsize=6.8,color='#52606A')
    save(fig,'six_parent_saved_trajectories')


def write_csv(name, rows):
    path=OUT/name
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    OUTPUTS.append(path)


def main():
    if shutil.disk_usage(ROOT).free < MAX_BYTES+RESERVE_BYTES:
        raise RuntimeError('Need 12 MiB bounded output plus 100 MiB free reserve')
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.65,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'svg.hashsalt':'nso-article-trajectories-20260928','savefig.facecolor':'white'})
    local_parents,local_records=load_local()
    large_parents,geometries,large_records=load_large()
    local_figure(local_parents,local_records)
    scene_figure(large_parents,geometries,large_records)
    write_csv('trajectory_costs.csv',SUMMARIES)
    write_csv('paired_trajectory_comparisons.csv',PAIRS)
    write_csv('saved_pose_actions.csv',TRAJECTORIES)
    captions={
        'local_paired_trajectories': {
            'caption_zh':'四个V36确认条件的完整G/S保存轨迹及逐动作带。各图采用同一米制尺度，P01仅按预定义设备坐标作刚性旋转；灰色路径为共同18动作前缀，之后由蓝虚线和橙实线分别表示G与S。圆环标记真实原地转向位置，箭头表示抽样记录的相机朝向，空心菱形标记首次非零实测几何残差更新，实心菱形为起终点。两个h0条件完全同轨迹，两个h1条件首次执行差异在第19步。动作带完整区分平移与转向，避免二维重叠路径隐藏动作成本。每步均有保存传感，包括免费初始帧；本组无额外observe原子动作。J5直接读取封存结果，无新评价。',
            'caption_en':'All four V36 confirmation pairs with saved routes and complete primitive-action strips. Panels share a metric scale; P01 is displayed in its declared equipment frame. Grey marks the common 18-action prefix. Open circles denote in-place turn positions; arrows show sampled recorded headings. Open diamonds indicate the first nonzero applied geometric residual, and filled diamonds the shared start/return. Both h0 trajectories coincide; both h1 pairs first diverge at executed action 19. Every action obtains a saved sensor frame; these runs have no extra observe primitive. J5 is copied from sealed endpoints.'},
        'six_parent_saved_trajectories': {
            'caption_zh':'已独立复核的六父正常关系开发组G/S保存轨迹，均预算120动作；P04两条实际115动作返航并剩余5动作，其他均120动作返航。斜线动作带表示未使用预算。所有地图使用相同10.75m见方坐标窗口，灰色几何仅用于离线解释；共同运行条件含公开安全导航图与精确仿真位姿。五对执行轨迹完全相同，P02在第70步选择诊断，71步转向产生首次执行差异，72步执行observe；该条件S的J_nav低于G。方点显示真实observe位置，圆环表示原地转向位置，动作带保留全部付费动作。该图用于说明场景级执行及已有边界，不声称六场景语义优势。J_nav与局部实验J5采用不同定义，不作混合平均。',
            'caption_en':'Previously reviewed nominal G/S trajectories on all six development layouts under a 120-action budget. Both P04 runs return after 115 actions, leaving five unused actions (hatched strips); all others use 120. All panels share a 10.75 m square display window. Geometry is offline explanatory context; the runs use a provided navigation graph and exact simulated poses. Five pairs execute identical trajectories. P02 selects a diagnostic at step 70, first diverges during the turn at step 71, and executes observe at step 72; its semantic endpoint is lower. Squares denote paid observe locations and circles in-place turns. This is execution and boundary evidence, not a claim of six-scene semantic superiority. J_nav is not pooled with local J5.'}}
    path=OUT/'captions.json';path.write_text(json.dumps(captions,ensure_ascii=False,indent=2)+'\n');OUTPUTS.append(path)
    read_bytes(Path(__file__).resolve())
    # Re-check only the files actually used for this derived figure, not old experiment source closures.
    for path,entry in SOURCES.items():
        if sha((ROOT/path).read_bytes()) != entry['sha256']:
            raise ValueError('Input changed while making figure: '+path)
    output_bytes=sum(p.stat().st_size for p in OUTPUTS)
    manifest=dict(schema='article.saved_trajectories.v1',date='2026-09-28',
        command='OPENBLAS_NUM_THREADS=1 .venv/bin/python -B scripts/plot_article_trajectories_20260928.py',
        sources=SOURCES,outputs={relative(p):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in OUTPUTS},
        trajectory_count=20,local_confirmation_pairs=4,six_parent_development_pairs=6,
        checks_passed=len(CHECKS),checks=CHECKS,
        scope=dict(new_worlds=0,new_sensor_packets=0,new_policy_runs=0,new_tsdf_integrations=0,
            new_quality_evaluations=0,new_independent_samples=0,geometry_role='offline explanatory background only',
            path_interpolation=False,path_smoothing=False,position_jitter=False,
            pooled_J5_and_J_nav=False,natural_recognition_claim=False,
            trajectories_shown='all local V36 pairs and all six reviewed nominal G/S pairs; no performance selection'),
        statistic_definitions=dict(translation_m='sum Euclidean XY distances of successive saved poses',
            turn_actions='paid left/right primitives; local V36 uses 90 degrees, six-parent uses 30 degrees',
            explicit_observe_actions='extra paid observe primitive; every motion also captures sensors',
            saved_sensor_frames='one saved observation per paid step plus initial observation',
            return_slack='recorded budget reserve, cohort-specific scope in each CSV row; not a newly computed shortest path',
            unused_budget_at_return='declared budget minus executed paid actions at verified final home pose'),
        output_bytes_before_manifest=output_bytes,maximum_output_bytes=MAX_BYTES)
    manifest_bytes=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    if output_bytes+len(manifest_bytes)>MAX_BYTES:
        raise RuntimeError('Generated figure package exceeds declared 12 MiB limit')
    (OUT/'manifest.json').write_bytes(manifest_bytes)
    print(json.dumps({'output':relative(OUT),'bytes':output_bytes+len(manifest_bytes),
        'checks':len(CHECKS),'trajectories':20,'new_worlds':0,'new_quality_evaluations':0}))


if __name__=='__main__':
    main()
