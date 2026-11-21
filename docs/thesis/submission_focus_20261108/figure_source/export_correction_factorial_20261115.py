"""Export measured factorial rows; no simulation or outcome selection.
Usage: python export_correction_factorial_20261115.py --summary PATH [--output PATH]
"""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).with_name('correction_factorial_20261115.csv'));a=p.parse_args()
s=json.loads(a.summary.read_text());assert s['parent_layouts']==1 and s['related_configurations']==2 and s['actual_factorial_routes']==8
rows=s['rows'];assert len(rows)==8 and len({(r['h'],r['observed'],r['forecast']) for r in rows})==8
with a.output.open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
