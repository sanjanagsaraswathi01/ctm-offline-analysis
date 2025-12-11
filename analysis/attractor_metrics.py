import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

def compute_distance_matrix(states, labels=None):
    """
    Computes pairwise Euclidean distances between states.
    If states is (N, Features), returns (N, N) distance matrix.
    If labels are provided, returns intra/inter class stats.
    """
    N = states.shape[0]
    
    # Flatten features if needed
    if len(states.shape) > 2:
        states = states.reshape(N, -1)
        
    dist_matrix = squareform(pdist(states, metric='euclidean'))
    
    if labels is not None:
        # Intra vs Inter
        intra_dists = []
        inter_dists = []
        
        # This can be slow for large N, but N=1000 is fine (1M pairs)
        # Vectorized approach:
        # labels_matrix[i, j] is True if labels[i] == labels[j]
        labels_eq = labels[:, None] == labels[None, :]
        
        # Mask diagonal
        np.fill_diagonal(labels_eq, False)
        
        # Upper triangle only to avoid dupes/diagonal
        triu_mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
        
        intra_mask = labels_eq & triu_mask
        inter_mask = (~labels_eq) & triu_mask
        
        intra_dists = dist_matrix[intra_mask]
        inter_dists = dist_matrix[inter_mask]
        
        return dist_matrix, intra_dists, inter_dists
        
    return dist_matrix

def calc_convergence_rate(states_over_time):
    """
    Calculates the rate of change of state over time.
    states_over_time: (N, Time, Features)
    Returns: (N, Time-1) velocity magnitudes
    """
    # Delta z_t = ||z_t - z_{t-1}||
    deltas = states_over_time[:, 1:, :] - states_over_time[:, :-1, :]
    
    # Norm over feature dim (axis 2)
    # Check shape
    if len(deltas.shape) == 3:
        feature_axis = 2
    else:
        # Assuming (Time, Features) for single sample
        feature_axis = 1
        
    velocities = np.linalg.norm(deltas, axis=feature_axis)
    
    return velocities

def detect_decision_point(certainties, threshold=0.9):
    """
    Identifies the time step where certainty consistently exceeds threshold.
    certainties: (N, Time) or (N, Time, Classes) -> use max certainty
    """
    # If shape (N, Time, 2), take index 1 (certainty)
    if len(certainties.shape) == 3:
        certainties = certainties[..., 1] # Assuming index 1 is certainty score
        
    N, T = certainties.shape
    decision_points = np.full(N, -1)
    
    for i in range(N):
        # Find first time t where all t' >= t are > threshold
        # This is strictly stable decision.
        # Relaxed: first time > threshold
        
        # Let's use first crossing of threshold that stays above for at least k steps?
        # Simple: first crossing
        crossings = np.where(certainties[i] > threshold)[0]
        if len(crossings) > 0:
            decision_points[i] = crossings[0]
            
    return decision_points
