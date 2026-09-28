"""Evaluation invariants shared by argument parsing and the runtime."""
from __future__ import annotations

import hashlib

TRAIN_FLAGS = ('train_global', 'train_local', 'train_slam',
               'train_goal_reachability', 'train_semantic')


def enforce_eval_mode(args):
    """Apply last, after presets, so --paper_mode cannot re-enable training."""
    if args.eval:
        for name in TRAIN_FLAGS:
            setattr(args, name, False)
    return args


def freeze_models(models):
    for model in models.values():
        if model is not None:
            model.eval()
            model.requires_grad_(False)


def model_fingerprints(models):
    """Hash parameters AND buffers; catches evaluation-time BatchNorm changes."""
    result = {}
    for name, model in models.items():
        if model is None:
            continue
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f'{key}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
            digest.update(tensor.numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def verify_frozen(models, before):
    after = model_fingerprints(models)
    if before != after:
        changed = [key for key in before if before[key] != after.get(key)]
        raise RuntimeError(f'Evaluation changed model state: {changed}')
    return after


def code_fingerprint(root):
    """Identify the working source, including uncommitted fixes."""
    from pathlib import Path
    import subprocess
    root = Path(root)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    paths = list(root.glob('*.py'))
    for directory in ('algo', 'env', 'loop', 'model', 'nso', 'semantic', 'utils'):
        paths.extend((root / directory).rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {'git_revision': revision, 'working_source_sha256': digest.hexdigest()}
