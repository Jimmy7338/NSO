"""Fail-closed frozen CPU experiment config, code, model and geometry checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import platform
import scipy
import matplotlib
from env.grid_layouts import generate_layout


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def geometry_hash(world):
    world=np.asarray(world,dtype=np.uint8)
    return min(hashlib.sha256(str(view.shape).encode()+np.ascontiguousarray(view).tobytes()).hexdigest()
               for view in [np.rot90(world,k) for k in range(4)]+[np.rot90(world.T,k) for k in range(4)])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, matplotlib=matplotlib.__version__)


def verify_protocol(config, root):
    root=Path(root)
    manifest=json.loads((root/config['frozen_protocol']).read_text())
    if manifest['runtime'] != runtime_versions():
        raise ValueError('frozen runtime versions changed')
    if digest_json(config)!=manifest['config_sha256']:
        raise ValueError('frozen protocol config changed')
    for name, expected in manifest['files_sha256'].items():
        if file_hash(root/name)!=expected:
            raise ValueError(f'frozen protocol file changed: {name}')
    expected=manifest['maps']
    seen=set(manifest['excluded_geometry_hashes'])
    selected=set()
    for entry in config['maps']:
        name=f"{entry['layout']}_seed{entry['seed']}"
        world=generate_layout(entry['layout'],config['map_size'],entry['seed'],config['door_width'])
        value=geometry_hash(world)
        if value!=expected[name] or value in seen or value in selected:
            raise ValueError('frozen map mismatch, duplicate or previously used geometry')
        selected.add(value)
    if len(selected)!=len(expected):
        raise ValueError('frozen map matrix mismatch')
    return manifest
