"""Exact CPU nearest-support acceleration for the original float64 metric.

cKDTree is a candidate index, not the reported metric. Returned distances are
recomputed using the original NumPy norm expression. Nearly tied candidates
are jointly checked, and unsupported numeric regimes use the original routine.
No threshold, point, point ordering or geometry interpretation is changed.
"""
import numpy as np
from scipy.spatial import cKDTree

from nso.observed_instances_v41 import _distances as reference_distances


def exact_nearest_distances(points, support):
    """Match V41's NumPy minimum distances; accelerate normal float64 geometry.

    Queries use eps=0, p=2 and one worker. The first two candidates expose
    nearly equal minima; a 32-epsilon radius margin retrieves all competitors
    for those queries before recomputing the original norm. Degenerate values,
    non-float64 arrays and extreme floating-point ranges use the old function.
    """
    points, support = np.asarray(points), np.asarray(support)
    if (points.ndim != 2 or support.ndim != 2 or points.shape[1:] != (3,)
            or support.shape[1:] != (3,)):
        return reference_distances(points, support)
    if not len(points) or not len(support):
        return np.full(len(points), np.inf)
    if (points.dtype != np.float64 or support.dtype != np.float64 or
            not np.isfinite(points).all() or not np.isfinite(support).all()):
        return reference_distances(points, support)
    # Underflow/overflow can change how a mathematical Euclidean nearest
    # neighbor relates to the original floating-point NumPy metric.
    magnitude = max(float(np.max(np.abs(points))), float(np.max(np.abs(support))))
    if magnitude > 1e150:
        return reference_distances(points, support)
    tree = cKDTree(support)
    if len(support) == 1:
        return np.linalg.norm(points-support[0],axis=1)
    indexed, indices = tree.query(points,k=2,eps=0.,p=2,workers=1)
    if not np.isfinite(indexed).all():
        return reference_distances(points,support)
    pairwise = np.linalg.norm(points[:,None,:]-support[indices],axis=2)
    distances = pairwise.min(axis=1)
    extreme = (distances > 0.) & (distances < 1e-150)
    if extreme.any():
        distances[extreme] = reference_distances(points[extreme],support)
    margin = 32.*np.finfo(np.float64).eps*np.maximum(1.,indexed[:,1])
    ambiguous = (distances > 0.) & ~extreme & (indexed[:,1]-indexed[:,0] <= margin)
    if ambiguous.any():
        rows = np.flatnonzero(ambiguous)
        radii = np.nextafter(np.maximum(indexed[rows,1],distances[rows])+margin[rows],np.inf)
        neighborhoods = tree.query_ball_point(points[rows],r=radii,p=2,eps=0.,workers=1)
        for row, neighbors in zip(rows,neighborhoods):
            # This uses the exact NumPy subtraction, square/sum/sqrt convention
            # of the original implementation; index ordering cannot pick a
            # slightly larger floating-point norm at a mathematical tie.
            distances[row] = np.linalg.norm(points[row]-support[neighbors],axis=1).min()
    return distances
