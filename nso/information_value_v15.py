"""Evaluator-only value of observed semantic groups, never simulator IDs.

Variants with indistinguishable observed semantic features must choose the
same candidate. The latent per-world oracle is a separate diagnostic only.
This finite-pool screen does not bound multi-decision semantic policies.
"""
import math
import numpy as np


def observed_group_information(rewards, semantic_groups):
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError('finite nonempty world-by-candidate rewards required')
    groups = list(semantic_groups)
    if len(groups) != len(values) or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('one explicit nonempty observed semantic group per world required')
    n = len(values)
    mean = lambda rows: np.asarray([math.fsum(rows[:, i]) / len(rows)
                                   for i in range(values.shape[1])])
    blind = float(mean(values).max())
    records = []
    for group in sorted(set(groups)):
        indices = [i for i, g in enumerate(groups) if g == group]
        averages = mean(values[indices])
        records.append(dict(observed_group=group, world_indices=indices,
            probability=len(indices) / n, mean_candidate_rewards=averages.tolist(),
            best_mean_reward=float(averages.max())))
    informed = math.fsum(r['probability'] * r['best_mean_reward'] for r in records)
    latent = math.fsum(values.max(axis=1)) / n
    gap = informed - blind
    if gap < -1e-12 or informed > latent + 1e-12:
        raise ValueError('information value order violated')
    return dict(geometry_blind_oracle=blind, observed_semantic_oracle=informed,
        latent_world_oracle=latent, raw_information_value=gap,
        information_value=max(0., gap),
        relative_information_value=max(0., gap) / blind if blind > 1e-12 else None,
        groups=records, equal_weight_per_world=True,
        latent_oracle_is_semantic_evidence=False, semantic_model_efficacy_proven=False)
