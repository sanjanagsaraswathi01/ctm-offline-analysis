import h5py
import numpy as np
import sys
import os
from scipy.stats import ttest_ind

sys.path.append(os.getcwd())
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from analysis.offline_metrics import compute_trajectory_entropy, compute_clustering_coefficient, compute_mutual_information

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def analyze_group(name, states, labels):
    print(f"\nAnalyzing {name}...")
    ent = compute_trajectory_entropy(states)
    print(f"  Mean Entropy: {ent.mean():.4f}")
    
    final_states = states[:, -1, :]
    clust = compute_clustering_coefficient(final_states, labels)
    print(f"  Clustering Coeff: {clust:.4f}")
    
    mi = compute_mutual_information(final_states, labels)
    print(f"  Mutual Info: {mi:.4f}")
    
    return {'entropy': ent, 'clustering': clust, 'mi': mi}

def main():
    # Load Data
    with h5py.File('data/offline_extended_trajectories.h5', 'r') as f:
        trained_mask = f['mask'][:].astype(np.float32)
        labels = f['labels'][:] # Lengths
        
    with h5py.File('data/baseline_trajectories.h5', 'r') as f:
        untrained_ctm = f['ctm_untrained'][:].astype(np.float32)
        untrained_rnn = f['rnn_untrained'][:].astype(np.float32)
        
    # Analyze
    res_trained = analyze_group("Trained CTM (Mask)", trained_mask, labels)
    res_untrained = analyze_group("Untrained CTM", untrained_ctm, labels)
    res_rnn = analyze_group("Untrained RNN", untrained_rnn, labels)
    
    # Comparisons (Trained vs Untrained CTM)
    print("\n--- COMPARISON: Trained CTM vs Untrained CTM ---")
    
    # Entropy (Lower is better/more structured for Trained?)
    # Actually, Trained CTM should be Structured (Low Entropy vs Random), but Untrained CTM might be erratic (High Entropy) or Static (Zero Entropy)?
    # Random weights usually lead to chaotic or decaying activity.
    # We expect Trained to be significantly different.
    # Success Criteria: Trained > Untrained (p<0.001) for structure metrics?
    # Let's test the difference.
    
    t_stat, p_val = ttest_ind(res_trained['entropy'], res_untrained['entropy'], equal_var=False)
    d_val = cohen_d(res_trained['entropy'], res_untrained['entropy'])
    print(f"Entropy T-Test: t={t_stat:.2f}, p={p_val:.2e}, d={d_val:.2f}")
    
    if p_val < 0.001 and abs(d_val) > 0.8:
        print("[PASS] Significant difference in Entropy (p < 0.001, d > 0.8)")
    else:
        print("[FAIL] Entropy difference not significant enough.")
        
    # Clustering (Higher is better)
    diff_clust = res_trained['clustering'] - res_untrained['clustering']
    print(f"Clustering Diff: {diff_clust:.4f}")
    if res_trained['clustering'] > res_untrained['clustering']:
         print("[PASS] Trained Clustering > Untrained")
    else:
         print("[FAIL] Trained Clustering <= Untrained")

    # MI 
    diff_mi = res_trained['mi'] - res_untrained['mi']
    print(f"MI Diff: {diff_mi:.4f}")
    if res_trained['mi'] > res_untrained['mi']:
         print("[PASS] Trained MI > Untrained")
    else:
         print("[FAIL] Trained MI <= Untrained")
         
    # Comparison vs RNN
    print("\n--- COMPARISON: Trained CTM vs Untrained RNN ---")
    # Just checking clustering
    if res_trained['clustering'] > res_rnn['clustering']:
        print("[PASS] Trained Clustering > RNN")
    else:
        print("[FAIL] Trained Clustering <= RNN")

if __name__ == '__main__':
    main()
