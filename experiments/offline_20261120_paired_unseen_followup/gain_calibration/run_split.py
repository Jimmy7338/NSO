"""Sequential low-thread acquisition with all per-layout exit statuses retained."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=('calibration','heldout'))
    args = parser.parse_args()
    output = HERE / 'runs' / args.split
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in PROTOCOL['layouts']:
        if item['split'] != args.split:
            continue
        layout = item['id']
        with (output / f'{layout}.log').open('x') as log:
            result = subprocess.run([sys.executable, str(HERE/'experiment.py'), 'run',
                         '--layout', layout, '--output', str(output/layout)], stdout=log, stderr=subprocess.STDOUT)
        rows.append(dict(layout=layout, returncode=result.returncode))
        (output/'status.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(rows[-1]), flush=True)
    if any(row['returncode'] for row in rows):
        print('At least one declared layout failed. Evaluation preserves the failures.', flush=True)
    sys.exit(subprocess.run([sys.executable, str(HERE/'evaluate.py'), args.split,
                             '--run-root', str(output)]).returncode)


if __name__ == '__main__':
    main()
