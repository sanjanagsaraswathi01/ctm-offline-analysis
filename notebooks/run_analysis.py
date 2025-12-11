#!/usr/bin/env python
# coding: utf-8

# # CTM Attractor Analysis (Milestone 3)
# 
# This notebook analyzes the internal dynamics of the Continuous Thought Machine (CTM) to verify attractor behavior.

# In[ ]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.attractor_metrics import compute_distance_matrix, calc_convergence_rate


# In[ ]:


# Load Data
with h5py.File('../data/dynamics_trajectories.h5', 'r') as f:
    states = f['post_activations'][:]
    predictions = f['predictions'][:]
    lengths = f['maze_length'][:]

print(f"Loaded States: {states.shape}")


# ## 1. Convergence Analysis
# Do states stabilize over time?

# In[ ]:


velocities = calc_convergence_rate(states)
mean_velocity = velocities.mean(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(mean_velocity)
plt.title("Mean State Velocity (Convergence) over Time")
plt.xlabel("Internal Tick")
plt.ylabel("Delta State Norm")
plt.grid(True)
plt.savefig('convergence_plot.png')
plt.show()


# ## 2. Attractor Stability (Intra/Inter Distances)
# Are final states grouped by task difficulty (length)?

# In[ ]:


# Bin lengths for classification
labels = np.digitize(lengths, bins=[10, 20, 30, 40])

final_states = states[:, -1, :]
_, intra, inter = compute_distance_matrix(final_states, labels)

print(f"Mean Intra-Class Distance: {intra.mean():.4f}")
print(f"Mean Inter-Class Distance: {inter.mean():.4f}")

plt.figure(figsize=(8, 6))
sns.kdeplot(intra, label='Intra-Class', fill=True)
sns.kdeplot(inter, label='Inter-Class', fill=True)
plt.title("Attractor Separation (Final State)")
plt.legend()
plt.savefig('attractor_separation.png')
plt.show()


# ## 3. Offline Regime Evaluation
# Does the attractor persist when input is removed?

# In[ ]:


with h5py.File('../data/offline_simulation.h5', 'r') as f:
    states_std = f['states_std'][:]
    states_off = f['states_off'][:]
    cutoff = f.attrs['cutoff']

# Calculate Drift: Dist(Standard, Offline) at each tick
drift = np.linalg.norm(states_std - states_off, axis=2).mean(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(drift, label='State Drift (Std vs Offline)')
plt.axvline(cutoff, color='r', linestyle='--', label='Input Cutoff')
plt.title("Attractor Stability: Offline Drift")
plt.xlabel("Tick")
plt.ylabel("Distance to Standard Trajectory")
plt.legend()
plt.savefig('offline_drift.png')
plt.show()

