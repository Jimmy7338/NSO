"""Catalog terminal and raw evidence by local path and SHA-256."""

import argparse
import json
from pathlib import Path

from measure import sha

STUDY = Path(__file__).resolve().parents[1]
SECTIONS = ('e1', 'mechanism', 'confirmatory')


def build_index():
    files = []
    for section in SECTIONS:
        for path in sorted((STUDY / section).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            rel = path.relative_to(STUDY)
            role = ('native_mesh_or_array' if path.suffix.lower() in {'.npz', '.ply', '.npy', '.glb', '.obj'}
                    else 'frozen_code_or_index' if path.suffix == '.py' or path.name == 'freeze.json'
                    else 'execution_or_evaluation_receipt')
            files.append(dict(path=str(rel), sha256=sha(path), bytes=path.stat().st_size, role=role))
    return dict(root=str(STUDY), files=files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    index = build_index()
    args.output.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'indexed': len(index['files'])}))
