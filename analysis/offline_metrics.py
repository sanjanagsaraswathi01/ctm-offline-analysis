import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA

def compute_trajectory_entropy(states, bins=10):
    """
    Computes Shannon entropy of the visited states.
    Uses PCA to reduce dims -> histogram -> entropy.
    states: (batch, time, features)
    """
    B, T, D = states.shape
    entropies = []
    
    # Fit PCA on all data to get common basis
    # Subsample if too large
    flat_states = states.reshape(-1, D)
    if flat_states.shape[0] > 10000:
        indices = np.random.choice(flat_states.shape[0], 10000, replace=False)
        pca_subset = flat_states[indices]
    else:
        pca_subset = flat_states
        
    pca = PCA(n_components=3)
    pca.fit(pca_subset)
    
    for i in range(B):
        sample_traj = states[i] # (T, D)
        proj = pca.transform(sample_traj)
        
        # Histogram entropy
        try:
            hist, _ = np.histogramdd(proj, bins=bins, density=True)
            # Filter zeros for log
            prob = hist[hist > 0] * (1.0 / hist.sum()) # Normalize just in case density is weird
            entropy = -np.sum(prob * np.log(prob + 1e-12))
            
            # Normalize by max entropy for these bins? or just raw.
            # Raw is fine for comparison.
            entropies.append(entropy)
        except Exception:
            entropies.append(0.0)
            
    return np.array(entropies)

def compute_clustering_coefficient(states, labels):
    """
    Ratio of (Mean Inter-Class Dist) / (Mean Intra-Class Dist).
    Higher is better clustering.
    states: (N, D) - usually final states
    """
    from scipy.spatial.distance import pdist, squareform
    
    dist = squareform(pdist(states))
    
    # Masks
    labels_eq = labels[:, None] == labels[None, :]
    np.fill_diagonal(labels_eq, False)
    
    mask_intra = labels_eq
    mask_inter = ~labels_eq
    np.fill_diagonal(mask_inter, False) # excl diag
    
    mean_intra = dist[mask_intra].mean()
    mean_inter = dist[mask_inter].mean()
    
    return mean_inter / (mean_intra + 1e-8)

def compute_mutual_information(states, labels, n_bins=10):
    """
    Estimates MI between State (Discretized) and Class Label.
    states: (N, D) - final states
    """
    # Simply discretize states via PCA to 1D scalar for crude MI
    pca = PCA(n_components=1)
    proj = pca.fit_transform(states).flatten()
    
    # Discretize
    digitized = np.digitize(proj, bins=np.linspace(proj.min(), proj.max(), n_bins))
    
    return mutual_info_score(digitized, labels)

def detect_basin_transitions(states, threshold=0.1):
    """
    Detects jumps between stable attractors.
    states: (time, features) or (batch, time, features)
    Returns: indices where velocity > threshold
    """
    # Calculate velocity
    if len(states.shape) == 3:
        velocities = np.linalg.norm(states[:, 1:] - states[:, :-1], axis=2)
    else:
        velocities = np.linalg.norm(states[1:] - states[:-1], axis=1)
        
    # Simple thresholding
    transitions = (velocities > threshold).astype(int)
    return transitions
