import h5py
import numpy as np
import sys
import os
from scipy.stats import ttest_ind

# Add root
sys.path.append(os.getcwd())
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from analysis.offline_metrics import compute_trajectory_entropy, compute_clustering_coefficient, compute_mutual_information

def main():
    print("Loading Data...")
    with h5py.File('data/offline_extended_trajectories.h5', 'r') as f:
        # Use Mask variant for primary "Offline" test
        states_mask = f['mask'][:].astype(np.float32)
        states_rw = f['random_walk'][:].astype(np.float32)
        labels = f['labels'][:]
        
    print(f"Stats Shape: {states_mask.shape}")
    
    # 1. Entropy Analysis
    print("\n--- 1. Entropy Analysis (Structure vs Random) ---")
    ent_mask = compute_trajectory_entropy(states_mask)
    ent_rw = compute_trajectory_entropy(states_rw)
    
    mean_mask = ent_mask.mean()
    mean_rw = ent_rw.mean()
    
    print(f"Mean Entropy (Mask): {mean_mask:.4f}")
    print(f"Mean Entropy (Random Walk): {mean_rw:.4f}")
    
    t_stat, p_val = ttest_ind(ent_mask, ent_rw, equal_var=False)
    print(f"T-Statistic: {t_stat:.2f}, P-Value: {p_val:.2e}")
    
    if mean_mask < mean_rw and p_val < 0.01:
        print("[PASS] Entropy < Random Walk (p < 0.01)")
    else:
        print("[FAIL] Entropy criteria not met.")

    # 2. Clustering Analysis
    print("\n--- 2. Clustering Analysis (Attractor Strength) ---")
    # Use final state
    final_states = states_mask[:, -1, :]
    
    clust_score = compute_clustering_coefficient(final_states, labels)
    print(f"Clustering Coefficient (Offline): {clust_score:.4f}")
    
    # Null Model (Shuffled Labels)
    shuffled_labels = np.random.permutation(labels)
    null_score = compute_clustering_coefficient(final_states, shuffled_labels)
    print(f"Clustering Coefficient (Null): {null_score:.4f}")
    
    if clust_score > null_score:
        print("[PASS] Clustering > Null Model")
    else:
        print("[FAIL] Clustering criteria not met.")
        
    # 3. Mutual Information
    print("\n--- 3. Mutual Information (Content Retention) ---")
    mi_score = compute_mutual_information(final_states, labels)
    print(f"Mutual Information (State <-> Label): {mi_score:.4f}")
    
    if mi_score > 0.3:
        print("[PASS] MI > 0.3")
    else:
        print("[FAIL] MI criteria not met.")

if __name__ == '__main__':
    main()
