#!/usr/bin/env python3
"""Wrap unchanged audited row exports in complete TeX tabular environments."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/thesis/tables/v39'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    specs=(('external','v39_external/table_method_rows.tex','lrrrrr',
        r'方法 & $\C$ & $\F$ & $\J$ & 动作数 & 合格/计划'),
        ('tare','v39_tare/table_rows.tex','lrrrrrrr',
        r'条件 & $\C$ & $\F$ & $\J$ & 动作 & $N_{\rm native}$ & $N_{\rm wait}$ & $N_{\rm return}$'))
    inputs={};outputs={}
    for name,relative,columns,header in specs:
        source=ROOT/'docs/thesis/figures'/relative
        rows=source.read_text()
        text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+'\\\\\\midrule\n'+rows+'\\bottomrule\n\\end{tabular}\n'
        target=OUT/f'{name}.tex';target.write_text(text)
        inputs[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        outputs[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(dict(source_sha256=inputs,outputs=outputs,
        numerical_rows_unchanged=True,scope='TeX alignment wrapper only'),indent=2)+'\n')


if __name__=='__main__':main()
