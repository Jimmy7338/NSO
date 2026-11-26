"""Editable schematic of the evaluated acquired-state loop, no new modules."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
HERE=Path(__file__).resolve().parent
def main():
 out=HERE/'method_figure_final';out.mkdir(exist_ok=False);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.8,'svg.fonttype':'none','pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(6.8,2.8));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');blue='#335f8c';teal='#177e89';ink='#233746';line='#738b9a'
 def box(x,y,title,body,color=blue,w=.19,h=.23):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.012',lw=.9,ec=color,fc='#f5f9fb'if color==blue else'#f0f8f7'))
  ax.text(x+w/2,y+h-.055,title,ha='center',va='center',fontsize=8.0,fontweight='bold',color=color)
  ax.text(x+w/2,y+.079,body,ha='center',va='center',fontsize=6.8,color=ink,linespacing=1.45)
 def arrow(points,color=line):
  for i,(a,b)in enumerate(zip(points[:-1],points[1:])):
   if i==len(points)-2:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.9,color=color,shrinkA=0,shrinkB=0))
   else:ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.9)
 box(.015,.6,'Acquired data','RGB-D + scan\nPose + labels')
 box(.27,.6,'Measured map','Occupancy + TSDF\nInstance tracks')
 box(.525,.6,'Routes','Heading paths\nShared scores')
 box(.78,.6,'Execution','Motion primitive\nNew observation')
 box(.27,.16,'Fitted bank','Closed / open\nDepth + placement',teal)
 box(.525,.16,'Gain model','V: visibility\nA: RGB-D + TSDF',teal)
 box(.78,.16,'Joint score','Coverage + quality\n+ object surface',teal)
 arrow([(.205,.715),(.26,.715)]);arrow([(.46,.715),(.515,.715)])
 arrow([(.365,.59),(.365,.40)],teal);arrow([(.46,.275),(.515,.275)],teal);arrow([(.62,.59),(.62,.40)],teal);arrow([(.715,.275),(.77,.275)],teal);arrow([(.875,.40),(.875,.59)],teal)
 arrow([(.715,.715),(.748,.715),(.748,.36),(.77,.36)],line)
 arrow([(.875,.84),(.875,.94),(.11,.94),(.11,.84)],blue);ax.text(.493,.956,'Acquire the next observation after each executed action',ha='center',va='bottom',fontsize=7.6,color=blue)
 for suffix in ('pdf','svg','png'):fig.savefig(out/('acquired_loop.'+suffix),bbox_inches='tight',dpi=250)
 plt.close(fig);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=dict(status='editable_evaluated_method_schematic',new_policy_modules=False,source=str(Path(__file__).resolve()),source_sha256=sha(Path(__file__)),source_geometry_input_to_actor=False,positive_performance_encoded=False,alternatives='Visibility and acquisition replace one gain term on a shared full bank',output_sha256={p.name:sha(p)for p in out.iterdir()if p.is_file()});(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
