"""Small CPU-trainable semantic observation-gain network for V11.

The module deliberately contains no simulator or ground-truth access. Feature
construction consumes sealed decision-time audits; targets are supplied only
by the offline training script after a candidate has physically executed.
"""
from dataclasses import dataclass, asdict

import numpy as np


GEOMETRY_FEATURE_NAMES = (
    "common_proxy_per_action", "generic_potential_per_action", "cost_fraction",
    "inverse_cost", "aperture_sum", "aperture_max", "aperture_abs_difference",
    "target_aperture",
)
SEMANTIC_FEATURE_NAMES = (
    "confidence_signed_target_aperture",
    "confidence_signed_generic_opportunity_per_action",
    "confidence_signed_measured_support_per_action",
    "confidence_signed_aperture_square",
)
GEOMETRY_CAPACITY_FEATURE_NAMES = (
    "aperture_sum_square", "aperture_max_square",
    "aperture_abs_difference_square", "target_aperture_square",
)


@dataclass(frozen=True)
class TinyMLPConfig:
    input_dim: int = 12
    hidden_dim: int = 4
    epochs: int = 800
    learning_rate: float = 0.01
    l2: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


def _asset_confidence(asset):
    """Bounded marker-evidence proxy; it is not natural-detector confidence."""
    support = max(1, int(asset["support_points"]))
    marked_fraction = float(asset["marked_points"]) / support
    return abs(float(asset["class_vote"])) * min(1.0, 20.0 * marked_fraction)


def candidate_features(prediction_audit, candidate, measured_assets, *, confidence_scale=1.0):
    """Return matched semantic and geometry-capacity 12-D feature vectors."""
    cost = float(candidate["cost"])
    if cost <= 0 or len(measured_assets) != 2:
        raise ValueError("V11 development features require positive cost and two measured assets")
    aperture = np.asarray(prediction_audit["aperture_factors"], dtype=float)
    generic = np.asarray(prediction_audit["generic_area_proxies"], dtype=float)
    if aperture.shape != (2,) or generic.shape != (2,):
        raise ValueError("V11 development features require two aperture/generic values")
    votes = np.asarray([float(a["class_vote"]) for a in measured_assets])
    confidence_scale = float(confidence_scale)
    if not 0.0 <= confidence_scale <= 1.0:
        raise ValueError("semantic confidence scale must be in [0,1]")
    confidence = np.asarray([_asset_confidence(a) for a in measured_assets]) * confidence_scale
    support = np.asarray([float(a["support_m2_proxy"]) for a in measured_assets])
    asset_index = candidate.get("asset_index")
    target_aperture = float(aperture.mean() if asset_index is None else aperture[int(asset_index)])
    signed = votes * confidence
    if asset_index is None:
        signed_target = float(np.mean(signed * aperture))
    else:
        signed_target = float(signed[int(asset_index)] * target_aperture)
    common = np.asarray([
        float(prediction_audit["common_proxy_per_action"]),
        float(prediction_audit["potential_proxies_before_cost"]["G"]) / cost,
        cost / 48.0,
        1.0 / cost,
        float(aperture.sum()),
        float(aperture.max()),
        float(abs(aperture[0] - aperture[1])),
        target_aperture,
    ])
    semantic = np.asarray([
        signed_target,
        float(np.sum(signed * aperture * generic) / cost),
        float(np.sum(signed * aperture * support) / cost),
        float(np.sum(signed * aperture**2) / max(float(aperture.sum()), 1e-8)),
    ])
    geometry_capacity = np.asarray([
        common[4] ** 2, common[5] ** 2, common[6] ** 2, common[7] ** 2,
    ])
    semantic_x = np.r_[common, semantic]
    geometry_x = np.r_[common, geometry_capacity]
    if not np.isfinite(semantic_x).all() or not np.isfinite(geometry_x).all():
        raise ValueError("nonfinite V11 feature")
    return semantic_x, geometry_x, {
        "asset_confidence": confidence.tolist(),
        "semantic_feature_names": list(GEOMETRY_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES),
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES + GEOMETRY_CAPACITY_FEATURE_NAMES),
    }


class TinyMLP:
    """One-hidden-layer regressor with deterministic full-batch Adam."""

    def __init__(self, config=TinyMLPConfig()):
        self.config = config
        self.x_mean = None
        self.x_scale = None
        self.y_scale = None
        self.parameters = None
        self.diagnostics = None

    @property
    def parameter_count(self):
        c = self.config
        return c.input_dim * c.hidden_dim + c.hidden_dim + c.hidden_dim + 1

    def fit(self, x, y, seed):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        c = self.config
        if x.ndim != 2 or x.shape != (len(y), c.input_dim) or len(y) < 2:
            raise ValueError("invalid V11 training arrays")
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1e-9] = 1.0
        self.y_scale = max(float(y.std()), 1e-9)
        z = (x - self.x_mean) / self.x_scale
        target = y / self.y_scale
        rng = np.random.default_rng(int(seed))
        w1 = rng.normal(0.0, 0.2, (c.input_dim, c.hidden_dim))
        b1 = np.zeros(c.hidden_dim)
        w2 = rng.normal(0.0, 0.2, (c.hidden_dim, 1))
        b2 = np.zeros(1)
        params = [w1, b1, w2, b2]
        first_moment = [np.zeros_like(p) for p in params]
        second_moment = [np.zeros_like(p) for p in params]

        def objective():
            hidden = np.tanh(z @ w1 + b1)
            residual = (hidden @ w2 + b2)[:, 0] - target
            return float(np.mean(residual**2) + c.l2 * (np.sum(w1**2) + np.sum(w2**2)))

        initial_loss = objective()
        loss_curve = [initial_loss]
        for step in range(1, c.epochs + 1):
            hidden = np.tanh(z @ w1 + b1)
            prediction = (hidden @ w2 + b2)[:, 0]
            output_gradient = 2.0 * (prediction - target) / len(target)
            gradients = [None] * 4
            gradients[2] = hidden.T @ output_gradient[:, None] + 2.0 * c.l2 * w2
            gradients[3] = np.asarray([output_gradient.sum()])
            hidden_gradient = output_gradient[:, None] @ w2.T * (1.0 - hidden**2)
            gradients[0] = z.T @ hidden_gradient + 2.0 * c.l2 * w1
            gradients[1] = hidden_gradient.sum(axis=0)
            for index, (parameter, gradient) in enumerate(zip(params, gradients)):
                first_moment[index] = c.beta1 * first_moment[index] + (1.0 - c.beta1) * gradient
                second_moment[index] = c.beta2 * second_moment[index] + (1.0 - c.beta2) * gradient**2
                corrected_m = first_moment[index] / (1.0 - c.beta1**step)
                corrected_v = second_moment[index] / (1.0 - c.beta2**step)
                parameter -= c.learning_rate * corrected_m / (np.sqrt(corrected_v) + c.epsilon)
            loss_curve.append(objective())
        self.parameters = tuple(params)
        final_loss = objective()
        self.diagnostics = {
            "seed": int(seed), "rows": int(len(y)), "parameter_count": self.parameter_count,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "loss_decreased": final_loss < initial_loss,
            "loss_curve": loss_curve,
        }
        if not all(np.isfinite(p).all() for p in params) or not np.isfinite(final_loss):
            raise ValueError("nonfinite trained V11 network")
        return self

    def predict(self, x):
        if self.parameters is None:
            raise ValueError("V11 network is not fitted")
        x = np.asarray(x, dtype=float)
        w1, b1, w2, b2 = self.parameters
        z = (x - self.x_mean) / self.x_scale
        return ((np.tanh(z @ w1 + b1) @ w2 + b2)[:, 0] * self.y_scale)

    def to_dict(self):
        if self.parameters is None:
            raise ValueError("V11 network is not fitted")
        return {
            "config": asdict(self.config), "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(), "y_scale": self.y_scale,
            "parameters": [p.tolist() for p in self.parameters],
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data):
        config = TinyMLPConfig(**data["config"])
        model = cls(config)
        model.x_mean = np.asarray(data["x_mean"], dtype=float)
        model.x_scale = np.asarray(data["x_scale"], dtype=float)
        model.y_scale = float(data["y_scale"])
        model.parameters = tuple(np.asarray(p, dtype=float) for p in data["parameters"])
        model.diagnostics = data["diagnostics"]
        if (model.x_mean.shape != (config.input_dim,)
                or model.x_scale.shape != (config.input_dim,)
                or any(not np.isfinite(p).all() for p in model.parameters)
                or np.any(model.x_scale <= 0) or model.y_scale <= 0):
            raise ValueError("invalid frozen V11 member")
        return model


class FrozenSemanticGainEnsemble:
    """JSON-only V11 ensemble used inside STGHP; no training or truth access."""

    def __init__(self, data):
        self.guard = None
        if data.get("schema_version") == "semantic_gain_v11_integration_bundle/1":
            if data.get("status") != "frozen_development_model_with_input_guard":
                raise ValueError("unfrozen V11 integration bundle")
            self.guard = data["guard"]
            data = data["model"]
        if (data.get("schema_version") != "semantic_gain_v11_frozen_development_model/1"
                or data.get("status") != "frozen_after_grouped_development_gates"):
            raise ValueError("unrecognized or unfrozen V11 development model")
        self.semantic = [TinyMLP.from_dict(row) for row in data["semantic_members"]]
        self.geometry = [TinyMLP.from_dict(row) for row in data["geometry_members"]]
        if not self.semantic or len(self.semantic) != len(self.geometry):
            raise ValueError("V11 semantic/geometry ensembles must align")
        self.metadata = {key: data[key] for key in (
            "schema_version", "status", "architecture", "parameters_per_member",
            "ensemble_seeds", "protocol_sha256", "independent_confirmation")}

    def in_distribution(self, semantic_x, geometry_x):
        if self.guard is None:
            return True
        for name, values in (("semantic", semantic_x), ("geometry", geometry_x)):
            x = np.asarray(values, dtype=float)
            lower = np.asarray(self.guard[name]["lower"], dtype=float)
            upper = np.asarray(self.guard[name]["upper"], dtype=float)
            if np.any(x < lower) or np.any(x > upper):
                return False
        return True

    @staticmethod
    def _predict(models, x):
        members = np.asarray([model.predict(x) for model in models])
        return members.mean(axis=0), members.std(axis=0), members

    def predict(self, semantic_x, geometry_x, swapped_x):
        semantic = self._predict(self.semantic, semantic_x)
        geometry = self._predict(self.geometry, geometry_x)
        swapped = self._predict(self.semantic, swapped_x)
        return {"semantic": semantic, "geometry": geometry, "swapped": swapped}


class ConditionalGainResidual:
    """Observed camera-yield correction indexed by category, instance and view sector.

    The outcome is an online sensor proxy. It is not evaluator surface area or F1.
    Fractional Beta counts prevent a large predicted footprint from dominating a
    single action update. Disabled feedback still records the diagnostic outcome.
    """

    def __init__(self, *, feedback_enabled=True, prior_strength=8.0,
                 minimum_updates=2, maximum_adjustment=.25):
        self.feedback_enabled = bool(feedback_enabled)
        self.prior_strength = float(prior_strength)
        self.minimum_updates = int(minimum_updates)
        self.maximum_adjustment = float(maximum_adjustment)
        if self.prior_strength <= 0:
            raise ValueError("positive conditional residual prior required")
        if self.minimum_updates < 1 or not 0 <= self.maximum_adjustment <= 1:
            raise ValueError("invalid conditional residual evidence guard")
        self.rows = {}
        self.events = []

    @staticmethod
    def key(asset, sector):
        center = np.asarray(asset["aabb_center"], dtype=float)
        if center.shape != (3,) or int(sector) not in range(8):
            raise ValueError("conditional residual requires a measured instance and sector")
        instance = tuple(np.rint(center[:2] / .2).astype(int).tolist())
        category = int(np.sign(float(asset["class_vote"])))
        return (category, *instance, int(sector))

    def posterior(self, key):
        row = self.rows.get(tuple(key))
        return .5 if row is None else row["alpha"] / (row["alpha"] + row["beta"])

    def semantic_scale(self, key):
        row = self.rows.get(tuple(key))
        if row is None or row["updates"] < self.minimum_updates:
            return 1.0
        return float(np.clip(self.posterior(key) / .5,
                             1. - self.maximum_adjustment, 1. + self.maximum_adjustment))

    def observe(self, key, camera_outcome, *, action_id):
        key = tuple(key)
        predicted = int(camera_outcome["predicted_cells"])
        realized = int(camera_outcome["realized_predicted_cells"])
        rate = float(realized / predicted) if predicted else None
        consumed = bool(self.feedback_enabled and rate is not None)
        if consumed:
            row = self.rows.setdefault(key, {"alpha": self.prior_strength / 2.,
                                             "beta": self.prior_strength / 2., "updates": 0})
            row["alpha"] += rate
            row["beta"] += 1. - rate
            row["updates"] += 1
        event = {"action_id": int(action_id), "key": list(key),
                 "realization_rate": rate, "feedback_consumed_for_planning": consumed,
                 "posterior_after": self.posterior(key)}
        self.events.append(event)
        return event

    def snapshot(self):
        return {"schema_version": "conditional_gain_residual_v11/1",
            "feedback_enabled": self.feedback_enabled,
            "prior_strength": self.prior_strength,
            "minimum_updates": self.minimum_updates,
            "maximum_adjustment": self.maximum_adjustment,
            "posteriors": [{"key": list(key), "posterior": self.posterior(key), **value}
                           for key, value in sorted(self.rows.items())],
            "events": list(self.events), "observed_actions": len(self.events)}
