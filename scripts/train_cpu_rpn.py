#!/usr/bin/env python3
"""Offline bounded-controller candidate RPN; split entire maps before training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.cpu_reachability import FEATURES, CandidateRPN
from utils.eval_protocol import code_fingerprint


def metrics(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    bins, ece = [], 0.
    for i in range(10):
        mask = (p >= i/10) & (p < (i+1)/10)
        n = int(mask.sum())
        confidence = float(p[mask].mean()) if n else None
        accuracy = float(y[mask].mean()) if n else None
        if n:
            ece += n/len(y)*abs(confidence-accuracy)
        bins.append(dict(count=n, mean_probability=confidence, success_fraction=accuracy))
    return dict(count=len(y), positives=int(y.sum()), negatives=int(len(y)-y.sum()),
                bce=float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),
                brier=float(np.mean((p-y)**2)), ece=ece,
                accuracy=float(np.mean((p >= .5) == y)), reliability_bins=bins)


def train(source, output, validation_seeds, epochs=120, seed=7):
    source, output = Path(source), Path(output)
    meta = json.loads((source/'run_metadata.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('training collection must be complete')
    episodes = [json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    validation_worlds = {e['ground_truth_sha256'] for e in episodes if e['map_seed'] in validation_seeds}
    splits = {'train': [], 'validation': []}
    provenance, budgets, seen = [], set(), set()
    for episode in episodes:
        path = source/episode['artifact_dir']/'goal_attempts.jsonl'
        # Some layout generators ignore the seed. A seed split alone leaks maps.
        split = 'validation' if episode['ground_truth_sha256'] in validation_worlds else 'train'
        identity = (episode['ground_truth_sha256'], episode.get('semantic_sha256'),
                    episode['start_seed'], episode['method'])
        if identity in seen:
            continue
        seen.add(identity)
        if not path.exists():
            continue
        provenance.append(dict(episode_key=episode['episode_key'], map_id=episode['map_id'],
                               split=split, world_sha256=episode['ground_truth_sha256'],
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            budgets.add(row['budget'])
            if row['controller_success'] is not None:
                splits[split].append((row['features'], row['controller_success']))
    if len(budgets) != 1:
        raise ValueError('expected one controller budget')
    arrays = {}
    for split, rows in splits.items():
        if not rows or len(set(y for _, y in rows)) != 2:
            raise ValueError(f'{split} requires both observed success and failure labels')
        arrays[split] = (np.asarray([x for x, _ in rows]), np.asarray([y for _, y in rows], float))
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, .2, (5, 16)); b1 = np.zeros(16)
    w2 = rng.normal(0, .2, (16, 1)); b2 = np.zeros(1)
    x, y = arrays['train']
    history, start = [], time.perf_counter()
    # Fixed epochs; validation is reported once and never used for model selection.
    for epoch in range(epochs):
        for indices in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/64)))):
            xb, yb = x[indices], y[indices, None]
            h = np.tanh(xb @ w1 + b1)
            p = 1/(1+np.exp(-np.clip(h @ w2+b2, -40, 40)))
            dz = (p-yb)/len(indices)
            dh = (dz @ w2.T)*(1-h*h)
            dw1, db1, dw2, db2 = xb.T@dh, dh.sum(0), h.T@dz, dz.sum(0)
            for w, grad in zip((w1,b1,w2,b2), (dw1,db1,dw2,db2)):
                w -= .08 * grad
        p = (1/(1+np.exp(-np.clip(np.tanh(x@w1+b1)@w2+b2, -40,40)))).ravel()
        history.append(metrics(y,p)['bce'])
    np.savez_compressed(output/'model.npz', w1=w1,b1=b1,w2=w2,b2=b2,
                        features=np.array(FEATURES), goal_budget=next(iter(budgets)))
    model = CandidateRPN(output/'model.npz')
    report = dict(schema='cpu_candidate_rpn_v1', task='observed bounded controller success',
                  architecture='5 observed-path features, 16 tanh units, 1 sigmoid; 113 parameters',
                  training_source=str(source.resolve()), seed=seed, epochs=epochs,
                  validation_map_seeds=validation_seeds, provenance=provenance,
                  goal_budget=next(iter(budgets)), model_sha256=model.sha256,
                  source_code=code_fingerprint(ROOT), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_s=time.perf_counter()-start, train_bce_history=history,
                  excluded_labels='unattempted, path-invalidated and episode-censored targets',
                  limitations=['synthetic geometry, perfect odometry', 'candidate MLP, not original convolutional RPN or UQ',
                               'controller action cost is highly predictive in deterministic grids'])
    for split,(xs,ys) in arrays.items():
        report[split] = metrics(ys, model.predict(xs))
        report[split+'_constant_prior'] = metrics(ys, np.full(len(ys), y.mean()))
        # Available geometric/controller-cost comparator, no training or hidden map.
        report[split+'_action_budget_rule'] = metrics(ys, (xs[:,1] <= 1).astype(float))
    (output/'training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    axes[0].plot(history); axes[0].set(xlabel='Epoch',ylabel='Training BCE')
    bins = report['validation']['reliability_bins']
    axes[1].plot([b['mean_probability'] for b in bins if b['count']],
                 [b['success_fraction'] for b in bins if b['count']], 'o-')
    axes[1].plot([0,1],[0,1],'--',color='gray')
    axes[1].set(xlabel='Predicted probability',ylabel='Observed success fraction',xlim=(0,1),ylim=(0,1))
    fig.tight_layout(); fig.savefig(output/'training_validation.png',dpi=160); plt.close(fig)
    print(json.dumps({k:report[k] for k in ('elapsed_s','train','validation')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--validation-seeds',required=True,type=int,nargs='+')
    parser.add_argument('--epochs',type=int,default=120)
    args=parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    train(args.source,args.output,args.validation_seeds,args.epochs)
