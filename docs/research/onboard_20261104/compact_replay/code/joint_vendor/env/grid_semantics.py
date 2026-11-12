"""Seeded synthetic salience; this is not an RGB-D or open-vocabulary sensor."""
from dataclasses import dataclass
import numpy as np
from env.grid_exploration import GridObservation


@dataclass(frozen=True)
class SemanticObservation(GridObservation):
    semantic: np.ndarray


def synthetic_semantics(shape, seed):
    rng = np.random.default_rng(seed)
    rows, cols = np.indices(shape)
    field = np.zeros(shape, dtype=np.float32)
    for _ in range(12):
        r, c = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        sigma = float(rng.uniform(2, 6))
        field = np.maximum(field, np.exp(-((rows-r)**2 + (cols-c)**2)/(2*sigma**2)))
    return field.astype(np.float32)
