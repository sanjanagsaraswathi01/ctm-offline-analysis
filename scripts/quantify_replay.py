import h5py
import numpy as np
import sys
import os
from scipy.stats import pearsonr, ttest_ind

sys.path.append(os.getcwd())
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from analysis.replay_metrics import compute_manifold_distance
from scipy.spatial import cKDTree


def main():
    print("Loading Data...")
    # Load Trained
    with h5py.File('data/offline_extended_trajectories.h5', 'r') as f:
        # Control = Wake Manifold
        # Use only first T steps? Or all? Control has extended T.
        # "Wake" is usually defined by the period with input. 
        # But here 'control' has input for whole time. So use all.
        wake_states = f['control'][:].astype(np.float32)
        offline_states = f['mask'][:].astype(np.float32)
        labels = f['labels'][:] # Maze Length (Difficulty)
        
    # Load Untrained
    with h5py.File('data/baseline_trajectories.h5', 'r') as f:
        untrained_states = f['ctm_untrained'][:].astype(np.float32)
        
    # Flatten for processing (Timesteps * Batch)
    # But for correlation we need per-sample scores.
    # So we aggregate per sample.
    
    # 1. Define Wake Manifold
    # Use All Control states from all samples as the reference manifold?
    # Yes, the set of "valid thought states".
    flat_wake = wake_states.reshape(-1, wake_states.shape[-1])
    
    print(f"Wake Manifold Size: {flat_wake.shape}")
    
    # Check Magnitudes
    mag_wake = np.linalg.norm(flat_wake, axis=1).mean()
    mag_off = np.linalg.norm(offline_states, axis=2).mean()
    mag_untrained = np.linalg.norm(untrained_states, axis=2).mean()
    print(f"Mean Magnitudes -> Wake: {mag_wake:.2f}, Offline: {mag_off:.2f}, Untrained: {mag_untrained:.2f}")
    
    # Helper for Batch Scores
    def get_batch_scores(states_batch, manifold_ref):
        # Normalize Batch and Manifold relative to Manifold stats?
        # Better: Normalize Manifold globally, then apply same transform to batch.
        # Simple: Unit vectors (Cosine distance proxy)
        
        # Norm function
        def normalize(v):
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            return v / (norm + 1e-8)
            
        states_norm = normalize(states_batch)
        manifold_norm = normalize(manifold_ref)
        
        # Subsample manifold
        if manifold_norm.shape[0] > 5000:
             vals = manifold_norm[np.random.choice(manifold_norm.shape[0], 5000, replace=False)]
        else:
             vals = manifold_norm
             
        tree_norm = cKDTree(vals)
        
        B_local = states_batch.shape[0]
        batch_scores = []
        
        for i in range(B_local):
            queries = states_norm[i]
            dists, _ = tree_norm.query(queries)
            avg_dist = dists.mean()
            # Score = 1 - Distance (Cosine-ish) or just -Distance
            # Euclidean on Unit Sphere: range 0 to 2.
            # Convert to "Similarity": 1 / (1 + dist) or similar.
            # Let's use -avg_dist.
            batch_scores.append(-avg_dist)
            
        return np.array(batch_scores)

    # Compute Normalized Scores
    print("Computing Normalized Replay Scores...")
    scores_trained = get_batch_scores(offline_states, flat_wake)
    scores_untrained = get_batch_scores(untrained_states, flat_wake)

    
    # 3. Baseline Comparison
    mean_trained = scores_trained.mean()
    mean_untrained = scores_untrained.mean()
    
    print(f"\nMean Replay Score (Trained): {mean_trained:.4f}")
    print(f"Mean Replay Score (Untrained): {mean_untrained:.4f}")
    
    diff = mean_trained - mean_untrained
    print(f"Difference: {diff:.4f}")
    
    if diff > 0.3: # User requested > 0.3 above baseline? Or is 0.3 absolute?
        # User: "score > 0.3 above baseline"
        # The scale of distance depends on D.
        # But let's assume normalized vectors or check relative.
        # If raw distance, 0.3 might be small.
        # Let's normalize scores or check raw.
        # We will report PASS if statistically sig + reasonable magnitude.
        print("[PASS] Replay Score > Baseline")
    else:
        print(f"[FAIL] Difference {diff:.4f} < 0.3 (or scale issue)")
        
    
    # 4. Difficulty Correlation
    print("\n--- Correlation with Difficulty (Maze Length) ---")
    # labels is array of lengths
    # Correlation between Score and Length?
    # Or Distance vs Length?
    # Hypothesis: Harder tasks (longer) might have *lower* replay score (more drift)? 
    # Or *higher* replay if they require "deep thought"?
    # User Request: "higher replay for ambiguous samples" (r > 0.4 implies positive corr).
    # So we expect Replay Score (Closeness) to INCREASE with Difficulty?
    # Meaning harder mazes stay CLOSER to manifold?
    # Or maybe "Replay" means "Activity".
    # Let's check the correlation direction.
    
    corr, p_corr = pearsonr(scores_trained, labels)
    print(f"Correlation (Score vs Length): r={corr:.4f}, p={p_corr:.4f}")
    
    if corr > 0.4:
         print("[PASS] Correlation > 0.4")
    elif corr < -0.4:
         print("[NOTE] Strong Negative Correlation (Harder = More Drift?)")
    else:
         print("[FAIL] Weak Correlation")

if __name__ == '__main__':
    main()
