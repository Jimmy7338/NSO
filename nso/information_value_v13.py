"""Evaluator-only finite-candidate information screen; never a planner input."""
import numpy as np


def summarize_paired_rewards(protocol, rows):
    lookup = {}
    for row in rows:
        key = (row["parent"], row["checkpoint_action"], row["arrangement"], row["candidate_id"])
        if key in lookup:
            raise ValueError("duplicate candidate outcome")
        if not np.isfinite(row["joint_gain"]):
            raise ValueError("non-finite candidate reward")
        lookup[key] = float(row["joint_gain"])
    used, pairs = set(), []
    tolerance = protocol["numerical_positive_tolerance"]
    for pair in protocol["eligible_pairs"]:
        parent, step = pair["parent"], pair["action_id"]
        sets = [{key[3] for key in lookup if key[:3] == (parent, step, arrangement)}
                for arrangement in protocol["arrangements"]]
        if any(ids != sets[0] or len(ids) != pair["candidates_per_arrangement"] for ids in sets):
            raise ValueError("missing or mismatched paired candidate outcomes")
        ids = sorted(sets[0])
        keys = [[(parent, step, arrangement, cid) for cid in ids]
                for arrangement in protocol["arrangements"]]
        values = np.array([[lookup[key] for key in line] for line in keys])
        used.update(key for line in keys for key in line)
        blind = float(values.mean(axis=0).max())
        informed = float(values.max(axis=1).mean())
        raw_gap = informed - blind
        if raw_gap < -tolerance:
            raise ValueError("finite candidate Jensen gap is unexpectedly negative")
        pairs.append(dict(parent=parent, action_id=step, candidate_ids=ids,
            rewards=values.tolist(), geometry_blind_oracle=blind,
            semantic_informed_oracle=informed, information_value=max(0., raw_gap),
            raw_information_value=raw_gap,
            optimal_ids_by_arrangement={a: [cid for cid, value in zip(ids, values[i])
                if value >= values[i].max() - tolerance]
                for i, a in enumerate(protocol["arrangements"])}))
    if used != set(lookup) or len(lookup) != protocol["expected_candidate_outcomes"]:
        raise ValueError("unexpected or incomplete outcome universe")
    parents = []
    for parent in protocol["parents"]:
        subset = [p for p in pairs if p["parent"] == parent]
        if not subset:
            raise ValueError("parent has no predeclared pairs")
        parents.append(dict(parent=parent,
            mean_information_value=float(np.mean([p["information_value"] for p in subset])),
            mean_geometry_blind_oracle=float(np.mean([p["geometry_blind_oracle"] for p in subset]))))
    mean_value = float(np.mean([p["mean_information_value"] for p in parents]))
    mean_blind = float(np.mean([p["mean_geometry_blind_oracle"] for p in parents]))
    screen = protocol["development_training_screen"]
    relative = mean_value / mean_blind if mean_blind > screen["minimum_geometry_blind_oracle_mean"] else None
    positive = sum(p["mean_information_value"] > tolerance for p in parents)
    return dict(pairs=pairs, parents=parents, mean_information_value=mean_value,
        mean_geometry_blind_oracle=mean_blind, relative_information_value=relative,
        positive_parent_count=positive, numerical_information_screen_passed=(
            relative is not None and relative >= screen["minimum_mean_value_relative_to_geometry_blind_oracle"]
            and positive >= screen["minimum_positive_parent_count"]))
