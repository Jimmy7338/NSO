"""Bound per-process numerical thread pools on the shared robot controller."""
import os


def bounded_environment():
    environment=os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[name]='1'
    return environment
