"""Small candidate-level point-estimate RPN, NumPy inference, frozen weights.

Predicts bounded controller success, NOT ground-truth connectivity or RPN-UQ.
Architecture: five observed-path features -> 16 tanh units -> sigmoid.
"""
import hashlib
from pathlib import Path
import numpy as np

FEATURES = ['distance_over_budget', 'estimated_actions_over_budget',
            'min_known_obstacle_clearance_over_10', 'unknown_neighbor_path_fraction',
            'predicted_gain_over_1000']


class CandidateRPN:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            if data['features'].tolist() != FEATURES:
                raise ValueError('incompatible RPN features')
            self.weights = {k: data[k].copy() for k in ('w1', 'b1', 'w2', 'b2')}
            self.goal_budget = int(data['goal_budget'])
        expected = {'w1': (5, 16), 'b1': (16,), 'w2': (16, 1), 'b2': (1,)}
        for k, v in self.weights.items():
            if v.shape != expected[k] or not np.isfinite(v).all():
                raise ValueError('invalid RPN weights')
            v.setflags(write=False)
        self._initial = self.fingerprint()

    def fingerprint(self):
        return hashlib.sha256(b''.join(self.weights[k].tobytes() for k in sorted(self.weights))).hexdigest()

    def verify_frozen(self):
        if self.fingerprint() != self._initial:
            raise RuntimeError('RPN weights changed during evaluation')

    def predict(self, features):
        w = self.weights
        z = np.tanh(np.asarray(features) @ w['w1'] + w['b1']) @ w['w2'] + w['b2']
        return (1/(1+np.exp(-np.clip(z, -40, 40)))).ravel()
