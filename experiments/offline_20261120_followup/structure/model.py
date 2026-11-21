"""B9 modular structural temperature: acquired evidence only.

Keep B8 conditional geometry within each structure. Calibrate the structural
log-evidence difference separately, then apply the original category prior.
This is a declared modular posterior, not the original joint generative model.
"""
import numpy as np

TEMPERATURE_GRID = (.25, .5, 1., 2., 4., 8., 16., 32., 64., 128.)
SIGMA_M = .01


def components(energy, prior_mass, sigma_m=SIGMA_M):
    energy = np.asarray(energy, float)
    prior = np.asarray(prior_mass, float)
    if energy.shape != (len(prior), 2) or len(prior) == 0:
        raise ValueError('K by 2 evidence and nonempty K masses required')
    if not np.isfinite(energy).all() or not np.isfinite(prior).all():
        raise ValueError('finite inputs required')
    if np.any(prior < 0) or not np.isclose(prior.sum(), 1.):
        raise ValueError('normalized nonnegative geometric masses required')
    if not np.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError('positive sigma required')
    with np.errstate(divide='ignore'):
        a = np.log(prior[:, None]) - energy / (2 * sigma_m**2)
    maximum = a.max(axis=0)
    unnormalized = np.exp(a - maximum)
    totals = unnormalized.sum(axis=0)
    conditional = unnormalized / totals
    log_evidence = maximum + np.log(totals)
    return conditional, log_evidence, a


def posterior(energy, prior_mass, shape_prior=(.5, .5), temperature=1., sigma_m=SIGMA_M):
    shape = np.asarray(shape_prior, float)
    if shape.shape != (2,) or not np.isfinite(shape).all() or np.any(shape <= 0) or not np.isclose(shape.sum(), 1.):
        raise ValueError('two positive normalized shape priors required')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    conditional, log_evidence, a = components(energy, prior_mass, sigma_m)
    if temperature == 1.:
        # Exact B8 operation order, including underflow behavior.
        with np.errstate(divide='ignore'):
            logw = np.log(np.asarray(prior_mass, float)[:, None]) + np.log(shape[None, :])
        logw = logw - np.asarray(energy, float) / (2 * float(sigma_m)**2)
        logw -= np.max(logw)
        joint = np.exp(logw)
        joint /= joint.sum()
    else:
        logits = (log_evidence - log_evidence.max()) / temperature + np.log(shape)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        joint = conditional * weights[None, :]
    return dict(joint_probability_mass=joint,
                shape_probability_mass=joint.sum(axis=0),
                conditional_geometry_mass=conditional,
                neutral_log_evidence_difference=float(log_evidence[1] - log_evidence[0]),
                temperature=float(temperature), sigma_m=float(sigma_m))


def neutral_nll(log_evidence_difference, target, temperature):
    if target not in (0, 1):
        raise ValueError('binary target required')
    if not np.isfinite(log_evidence_difference) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('finite evidence and positive temperature required')
    signed = (1 - 2 * target) * float(log_evidence_difference) / float(temperature)
    return float(np.logaddexp(0., signed))


def select_temperature(rows):
    """Equal object/condition NLL; caller supplies calibration-only scalar rows."""
    if not rows or any(r['split'] != 'calibration' for r in rows):
        raise ValueError('calibration split only')
    seen = set()
    grouped = {}
    for row in rows:
        key = (row['object_id'], row['condition'])
        if key in seen:
            raise ValueError('duplicate object condition')
        seen.add(key)
        grouped.setdefault(row['object_id'], []).append(row)
    conditions = {'front', 'front_side', 'around', 'occluded'}
    if any({r['condition'] for r in block} != conditions for block in grouped.values()):
        raise ValueError('all four fixed conditions required for every object')
    curve = []
    for temperature in TEMPERATURE_GRID:
        values = [np.mean([neutral_nll(r['log_evidence_difference'], r['target'], temperature)
                           for r in block]) for block in grouped.values()]
        curve.append(dict(temperature=temperature, equal_object_nll=float(np.mean(values))))
    best = min(curve, key=lambda r: (r['equal_object_nll'], abs(np.log2(r['temperature'])), -r['temperature']))
    return dict(temperature=best['temperature'], criterion='calibration_equal_object_neutral_class_NLL',
                objects=len(grouped), cases=len(rows), curve=curve,
                tie_rule='exact NLL tie: closest log2 distance to1; then larger temperature',
                test_truth_used=False, sigma_m=SIGMA_M)
