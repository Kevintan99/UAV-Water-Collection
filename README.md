# UAV Water Collection (MuJoCo + SMBPC)
## Project Overview
This project provides a **MuJoCo-based simulation** of a UAV tasked with collecting water from a river (or other body of water). The UAV uses **Sampling-Based Model Predictive Control (SBMPC)** to plan and execute trajectories that account for system dynamics, constraints, and uncertainties.  

The repository is designed as a research and experimentation framework for:
- Testing UAV–load dynamics in simulation
- Developing and tuning sampling-based control algorithms
- Comparing CPU vs GPU rollout performance (with JAX acceleration)
- Building towards real-world validation of UAV-based water collection

## Key features
- *Sampling-Based MPC*: Stochastic trajectory sampling with receding-horizon selection (tunable cost terms and constraints).  
- *High-fidelity physics*: MuJoCo contact & dynamics for realistic dip/collect maneuvers.  
- *JAX-accelerated rollouts* (optional): Significant speedups on GPU.  
- *Modular design*: Swap dynamics, costs, or sampler strategies without rewriting the loop.

## Repository Structure
```
UAV-Water-Collection/
├─ sbmpc/ # Core planning / control modules
├─ examples/ # Ready-to-run simulation scripts & configs
├─ crazyflie_mission.py # Example mission / waypoint runner
├─ crazyflie_mission_waypoints.txt
├─ environment.yml # Conda env (CPU/GPU variants via extras)
├─ requirements.txt # Pinned deps (pip)
├─ pyproject.toml # Package metadata / extras [cuda12, cuda12_local]
├─ LICENSE.txt # BSD-3-Clause
└─ README.md
```

# Installation
## Requirements
 - [Nvidia cuda toolkit](https://developer.nvidia.com/cuda-toolkit) installed system-wide (if you want to use a local CUDA version)
 - [miniforge](https://github.com/conda-forge/miniforge/releases)

## Clone Repository
Clone this repository and move into the project folder:

```bash
git clone https://github.com/Kevintan99/UAV-Water-Collection.git
cd UAV-Water-Collection
```
## Instructions
Create the conda environment with
```bash
mamba env create -f environment.yml
```

Activate the environment with
```bash
conda activate sbmpc
```

Depending on the CUDA settings of your machine, choose between
- CPU-only acceleration
```bash
pip install -e .
```
- GPU acceleration with pip-installed CUDA libraries
```bash
pip install -e ".[cuda12]"
```
- GPU acceleration with locally installed CUDA
```bash
pip install -e ".[cuda12_local]"
```

Refer to the Jax documentation for details.

# Quick Start
1. Activate the environment
```
conda activate sbmpc
```
2. Run an example
```
python examples/quadrotor_comparison.py
```

>>>>>>> 3274062 (Update readme file)


