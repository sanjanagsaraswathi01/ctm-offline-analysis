import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import sqrtm

def compute_manifold_distance(offline_states, wake_states, subsample_wake=1000):
    """
    Computes the mean nearest-neighbor distance from each offline state to the 'wake' manifold.
    
    offline_states: (N_off, D) - Flattened offline states
    wake_states: (N_wake, D) - Flattened wake states defining the manifold
    
    Returns: distances (N_off,)
    """
    # Subsample wake states for efficiency if too large
    if wake_states.shape[0] > subsample_wake:
        indices = np.random.choice(wake_states.shape[0], subsample_wake, replace=False)
        target_states = wake_states[indices]
    else:
        target_states = wake_states
        
    # Build Tree
    tree = cKDTree(target_states)
    
    # Query
    dists, _ = tree.query(offline_states)
    return dists

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance:
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
             m = np.max(np.abs(covmean.imag))
             # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_sequence_matching_score(traj_a, traj_b):
    """
    Simple proxy for sequence matching: average correlation between trajectory dimensions.
    """
    # Normalize
    a_norm = (traj_a - traj_a.mean(axis=0)) / (traj_a.std(axis=0) + 1e-8)
    b_norm = (traj_b - traj_b.mean(axis=0)) / (traj_b.std(axis=0) + 1e-8)
    
    # Correlation
    corr = (a_norm * b_norm).mean()
    return corr
