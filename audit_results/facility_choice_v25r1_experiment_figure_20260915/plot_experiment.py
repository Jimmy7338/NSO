#!/usr/bin/env python3
"""Plot sealed declarations/catalog only; no simulator/planner/metric imports."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/dev/shm/nso_v25_figure_mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COST = ROOT/'audit_results/facility_choice_v25_service_cost_20260915'
STATIC = ROOT/'audit_results/facility_choice_v25_static_r1_20260915'
CAP = 2*1024*1024
RESERVE = 32*1024*1024

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

sources = {}
for folder in (COST, STATIC):
    sealed = read(folder/'artifact_hashes.json')
    for name in ('result.json', 'manifest.json'):
        p = folder/name
        assert sha(p) == sealed[name], str(p)
        sources[str(p.relative_to(ROOT))] = sha(p)
    sources[str((folder/'artifact_hashes.json').relative_to(ROOT))] = sha(folder/'artifact_hashes.json')
cost, static = read(COST/'result.json'), read(STATIC/'result.json')
assert cost['status'] == 'complete_static_cost_only'
assert static['status'] == 'complete_static_only'
assert read(COST/'manifest.json')['status'] == read(STATIC/'manifest.json')['status'] == 'complete'
protected = read(COST/'manifest.json')['protected_source_sha256']
for name in ('env/facility_choice_v24.py', 'env/facility_choice_v24_1.py', 'env/facility_choice_v25_r1.py'):
    assert sha(ROOT/name) == protected[name], name
    sources[name] = sha(ROOT/name)

# Execute only the literal dict expression and the pure panel declaration AST.
# World constructors, module imports, occupancy, ray casting and routes never run.
base_ast = ast.parse((ROOT/'env/facility_choice_v24.py').read_text())
layout_node = next(n.value for n in base_ast.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'LAYOUTS_V24' for t in n.targets))
layouts = eval(compile(ast.Expression(layout_node), '<frozen literal layout>', 'eval'), {'__builtins__': {}, 'dict': dict})
spec = layouts['D24-P00']
r1_ast = ast.parse((ROOT/'env/facility_choice_v25_r1.py').read_text())
fn = next(n for n in r1_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'partition_boxes_v25')
assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(fn))
ns = {'__builtins__': {}, 'enumerate': enumerate, 'sorted': sorted, 'dict': dict,
      'LAYOUTS_V25': {'D25-P00': spec}, 'PARTITION_HEIGHT_M': 2.4}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<frozen pure panels>', 'exec'), ns)
panels = ns['partition_boxes_v25']('D25-P00')

cls = next(n for n in base_ast.body if isinstance(n, ast.ClassDef) and n.name == 'FacilityChoiceWorldV24')
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
def box_calls(block):
    return [n.value for n in block if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name) and n.value.func.id == 'box']
def literal_box(call, env):
    values = [eval(compile(ast.Expression(a), '<frozen box expression>', 'eval'), {'__builtins__': {}}, env) for a in call.args]
    return values + ([1] if len(values) == 6 else [])
w,h = spec['width'],spec['height']
x,y,sx,sy = spec['divider']
walls = [literal_box(n, locals()) for n in box_calls(init.body)]
walls = [p for p in walls if p[2] >= 0]  # omit the floor from the plan view
loop = next(n for n in init.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
asset_boxes, backstops = [], []
for i,(cx,front) in enumerate(spec['fronts']):
    env = dict(i=i,cx=cx,front=front,owner=100+i)
    boxes = [literal_box(n, env) for n in box_calls(loop.body)]
    assert len(boxes) == 3
    backstops.extend(boxes[:2]); asset_boxes.append(boxes[2])
# v24.1 only increases these four backstop heights; XY footprints stay identical.
assert len(walls) == 5 and len(backstops) == 4 and len(panels) == 10

parent = next(p for p in cost['parents'] if p['parent'] == 'D25-P00')
catalog = parent['route_catalog']
assert parent['total_budget'] == 400 and parent['paid_prefix_actions'] == 234
assert catalog['A']['prefix_states'] == catalog['B']['prefix_states']
assert catalog['A']['prefix_actions'] == catalog['B']['prefix_actions']
for arm, expected in (('A',354),('B',340)):
    c = catalog[arm]
    assert c['paid_total'] == expected and len(c['actions']) == expected
    assert len(c['states']) == expected+1 and c['states'][0] == c['states'][-1] == parent['anchor']
    assert c['scripted_development_arm'] and not c['unknown_map_runtime']
    assert not c['shape_quality_or_completion_guaranteed']
case_rows = [c for c in static['cases'] if c['parent'] == 'D25-P00']
assert len(case_rows) == 2
resolution = case_rows[0]['original_sensor_config']['resolution_m']
def xy(states):
    a = np.asarray(states, dtype=float)
    return np.column_stack(((a[:,1]+.5)*resolution, h-(a[:,0]+.5)*resolution))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10, 'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig = plt.figure(figsize=(14.4,7.4), facecolor='white')
colors = {'A':'#1966ad','B':'#d76c15'}
fig.suptitle('P00: paired facility layouts and fixed acquisition routes', fontsize=17, y=.972, weight='bold')
fig.text(.5,.927,'Same partitions and routes in both layouts; only the complex facility changes sides',ha='center',fontsize=11,color='#44515e')
for idx,case in enumerate(case_rows):
    ax = fig.add_axes([.045+idx*.50,.31,.43,.56])
    for p in walls+backstops:
        ax.add_patch(Rectangle((p[0],p[1]),p[3],p[4],facecolor='#9fa8b0',edgecolor='#737d85',lw=.4,zorder=2))
    for panel in panels:
        p=panel['primitive']
        ax.add_patch(Rectangle((p[0],p[1]),p[3],p[4],facecolor='#333e48',edgecolor='#202a32',lw=.45,zorder=3))
    for p in asset_boxes:
        ax.add_patch(Rectangle((p[0],p[1]),p[3],p[4],facecolor='#d2e6df',edgecolor='#2b6855',lw=1.0,zorder=4))
    for row in case['attachment_visibility']:
        p=row['physical_primitive']
        ax.add_patch(Rectangle((p[0],p[1]),p[3],p[4],facecolor='#598d77',edgecolor='#154d37',lw=.7,zorder=5))
    a=xy(catalog['A']['prefix_states'])
    ax.plot(*a.T,color='#adb7bf',lw=3.5,ls=(0,(3,2)),zorder=1)
    for arm in ('A','B'):
        c=catalog[arm]
        for key,style in (('continuation_states','-'),('return_states',(0,(2,2)))):
            a=xy(c[key]);ax.plot(*a.T,color=colors[arm],lw=1.4,ls=style,zorder=6,alpha=.94)
        views=xy(c['selected_view_states'])
        for j,(vx,vy) in enumerate(views):
            ax.plot(vx,vy,'o',color=colors[arm],ms=5,mec='white',mew=.6,zorder=8)
            ax.annotate('',xy=(vx,vy-.40),xytext=(vx,vy-.08),arrowprops=dict(arrowstyle='-|>',color=colors[arm],lw=1.1),zorder=8)
            ax.annotate(arm+str(j+1),xy=(vx,vy),xytext=(vx+(-.23 if j==0 else .23),9.24),
                ha='center',fontsize=8.5,color=colors[arm],weight='bold',arrowprops=dict(arrowstyle='-',color=colors[arm],lw=.5))
    for i,(cx,front) in enumerate(spec['fronts']):
        is_complex = any(a['asset_id'] == i for a in case['attachment_visibility'])
        ax.text(cx,front+.34,'AB'[i],ha='center',va='center',fontsize=11,weight='bold',color='#154d37',zorder=9)
        ax.text(cx,front-.62,'complex' if is_complex else 'simple',ha='center',fontsize=8.5,color='#344f42',
                bbox=dict(facecolor='white',edgecolor='none',pad=.6,alpha=.88),zorder=9)
    anchor=xy([parent['anchor']])[0]
    ax.plot(*anchor,marker='*',ms=12,color='#101e2a',zorder=10)
    ax.annotate('Original anchor\n(start / prefix end / finish)',xy=anchor,xytext=(8.1,2.17),ha='center',va='bottom',fontsize=8,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1),arrowprops=dict(arrowstyle='-',color='#253644',lw=.65),zorder=10)
    ax.annotate('',xy=(anchor[0],anchor[1]-.5),xytext=(anchor[0],anchor[1]-.1),arrowprops=dict(arrowstyle='-|>',lw=1.1,color='#101e2a'))
    ax.set(xlim=(0,16),ylim=(0,10),xticks=range(0,17,2),yticks=range(0,11,2),xlabel='x (m)',ylabel='y (m)')
    ax.set_aspect('equal');ax.grid(color='#d7dde1',alpha=.65,lw=.5);ax.set_axisbelow(True)
    ax.set_title(('(a) A complex / B simple','(b) A simple / B complex')[idx],fontsize=12,pad=8,weight='bold')
handles=[Patch(facecolor='#d2e6df',edgecolor='#2b6855',label='Cabinet body'),Patch(facecolor='#598d77',label='Actual external attachment'),
    Patch(facecolor='#9fa8b0',label='Walls / existing backstops'),Patch(facecolor='#333e48',label='V25 r1 physical partitions'),
    Line2D([],[],color='#adb7bf',lw=3,ls='--',label='Common paid prefix'),Line2D([],[],color=colors['A'],label='A supplemental route'),
    Line2D([],[],color=colors['B'],label='B supplemental route'),Line2D([],[],color='#526574',ls=':',label='Return segment (arm color)')]
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.50,.242),ncol=4,frameon=False,fontsize=9,columnspacing=1.7,handlelength=2)
tax=fig.add_axes([.24,.085,.52,.105]);tax.axis('off')
rows=[[arm,catalog[arm]['paid_prefix_actions'],catalog[arm]['supplemental_paid_actions'],catalog[arm]['paid_total'],parent['total_budget']] for arm in ('A','B')]
table=tax.table(cellText=rows,colLabels=['Option','Shared prefix','Extra incl. return','Total actions','Budget'],cellLoc='center',loc='center',colWidths=[.11,.21,.28,.22,.18])
table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.27)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('#d4dce2');cell.set_linewidth(.5)
    if r==0:cell.set_facecolor('#edf2f5');cell.set_text_props(weight='bold')
fig.text(.5,.055,'All turns and return actions are paid. Repeated positions overlap in this plan view.',ha='center',fontsize=8.5,color='#44515e')
fig.text(.5,.023,'Fixed GT-designed acquisition, not an autonomous policy. Active geometric probing may also reveal the extra information.',ha='center',fontsize=9,color='#293844')
blobs={}
for ext in ('png','pdf'):
    b=io.BytesIO();fig.savefig(b,format=ext,dpi=160,facecolor='white',metadata={'Title':'P00 fixed acquisition route design'});blobs[ext]=b.getvalue()
plt.close(fig)
assert sum(map(len,blobs.values())) < CAP
assert os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize-sum(map(len,blobs.values())) >= RESERVE
for ext,blob in blobs.items():
    (OUT/f'p00_experiment_design.{ext}').write_bytes(blob)
receipt=dict(status='complete_static_figure',parent='D25-P00',source_sha256=sources,
    geometry_extraction='Only frozen literal layout, pure partition function AST and literal box argument AST; attachments from frozen static metadata',
    route_source='Frozen route_catalog, no route computation; exact coordinates without offset',
    physical_actions=0,new_world_instances=0,sensor_calls=0,tsdf_fusions=0,new_cost_computations=0,new_quality_evaluations=0,
    not_autonomous_or_semantic_effectiveness_evidence=True,paired_layouts=list(c['assignment'] for c in case_rows),
    counts=dict(walls_and_divider=len(walls),backstops=len(backstops),r1_partitions=len(panels),cabinet_bodies=len(asset_boxes)),
    routes={arm:{k:catalog[arm][k] for k in ('paid_total','paid_prefix_actions','supplemental_paid_actions','selected_view_states','planned_leg_costs','actions_sha256')} for arm in ('A','B')},
    total_budget=parent['total_budget'],figure_bytes={ext:len(b) for ext,b in blobs.items()},figure_byte_cap=CAP)
(OUT/'source_and_zero_action_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'figure_bytes':receipt['figure_bytes'],'physical_actions':0}))
