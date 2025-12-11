# CTM Offline Analysis: Quantifying Internal Thought Structure

This project extends the **Continuous Thought Machine (CTM)** architecture to scientifically verify the nature of its "Offline" dynamics. We investigate whether the model's internal states represent structured "thought" or random noise when sensory input is removed.

> **Note**: This is an analysis extension built on top of the original [Continuous Thought Machines](https://github.com/SakanaAI/continuous-thought-machines) repository by Sakana AI.

## 🧪 Project Objective
To determine if CTMs exhibit:
1.  **Attractor Dynamics**: Do thoughts settle into stable basins?
2.  **Offline Structure**: Is the internal state significantly different from random noise?
3.  **Meaningful Replay**: Do offline thoughts correlate with valid task solutions?

## 📊 Key Findings

### 1. Attractors Confirmed
We tracked latent velocities and found consistent convergence ($\Delta z \to 0$) into class-specific basins.
-   **Inter-Class Distance**: High (~187)
-   **Intra-Class Variance**: Low (~153)

### 2. Structured "Offline Mode"
We masked inputs after $t=25$ and analyzed the resulting trajectories.
-   **Entropy**: The model maintains low entropy (3.60) compared to random walks (4.39), proving the dynamics are non-random ($p \ll 0.001$).
-   **Clustering**: States remain clustered by task concept even in total silence.

### 3. Replay & Difficulty
We discovered a **"Deep Thought"** phenomenon:
-   **Fidelity**: Offline states stay close to the "Wake Manifold" of valid computations.
-   **Correlation**: Challenging tasks (longer mazes) induce *more* deviation from rote memory ($r = -0.87$), suggesting the model explores novel state spaces to solve hard problems.

## 🚀 Reproduction Pipeline

We provide a single script to reproduce all experiments, from data extraction to statistical verification.

### Prerequisites
```bash
./setup_env.sh
source ctm_env/bin/activate
```

### Run End-to-End Analysis
```bash
./run_reproduction.sh
```
This script will:
1.  Extract dynamics from the pre-trained CTM.
2.  Generate "Offline" datasets (Masked input).
3.  Generate Baselines (Untrained CTM, Simple RNN).
4.  Compute Entropy, Clustering, and Replay metrics.

### Visualize Results
To see the generated plots (Convergence, Structure, Correlation):
```bash
python notebooks/make_final_plots.py
```
Open `notebooks/Final_Synthesis.ipynb` for the interactive dashboard.

## 📂 Repository Structure
*   **`analysis/`**: Scientific metric definitions (Entropy, Manifold Distance).
*   **`scripts/`**: Execution pipeline (Extraction, Simulation, Verification).
*   **`models/`**: Baseline architecture definitions.
*   **`notebooks/`**: Visualization and synthesis.
