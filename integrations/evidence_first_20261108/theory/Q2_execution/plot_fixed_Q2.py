"""Plot saved primary coverage curves only; no experiment execution."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent/'Q2_execution'
rows=list(csv.DictReader(open(D/'coverage_curves.csv')))
fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
styles=[('C','#777777','-', 'x'),('G','#0072B2','-',None),('S','#D55E00','-',None),('X','#009E73','--',None),('G_lambda_dev7','#CC79A7','-.',None)]
for ax,arr in zip(axes,['shelf_west','shelf_east']):
 for mode,color,style,marker in styles:
  r=[z for z in rows if z['arrangement']==arr and z['condition']==mode]
  ax.plot([int(z['paid_action'])for z in r],[100*float(z['coverage_fraction'])for z in r],label='G × 7'if mode=='G_lambda_dev7'else mode,color=color,ls=style,lw=1.5,marker=marker,markevery=12,ms=4)
 ax.set_xlabel('Paid actions');ax.set_ylabel(('West'if arr=='shelf_west'else'East')+' observed area (%)');ax.set_xlim(0,96);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[1].legend(frameon=False,fontsize=8,ncol=2,loc='lower right');fig.savefig(D/'coverage_curves.pdf');fig.savefig(D/'coverage_curves.png',dpi=200)
