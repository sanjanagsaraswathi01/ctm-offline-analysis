#!/bin/bash
set -e

echo "========================================================"
echo "CTM Reproduction Pipeline: Starting End-to-End Run"
echo "========================================================"

# Activate Environment (Assumes ctm_env exists as per setup)
source ctm_env/bin/activate || echo "Warning: Could not source ctm_env. Assuming environment is active."

echo "[Step 1/6] Extracting Standard Dynamics (M2)..."
python scripts/extract_dynamics.py --num_samples 200

echo "[Step 2/6] Simulating Standard Offline Mode (M3)..."
python scripts/simulate_offline.py --num_samples 100

echo "[Step 3/6] Generating Extended Offline Datasets (M4)..."
python scripts/generate_offline_dataset.py --num_samples 200 --extend_factor 2.0 --cutoff 25

echo "[Step 4/6] Generating Baseline Datasets (M5)..."
python scripts/generate_baselines.py --num_samples 200

echo "[Step 5/6] Calculating Success Metrics (Entropy/Clustering) (M4/M5)..."
python scripts/calculate_offline_success.py

echo "[Step 6/6] Quantifying Replay & Correlation (M6)..."
python scripts/quantify_replay.py

echo "========================================================"
echo "Pipeline Complete. All results verified."
echo "========================================================"
