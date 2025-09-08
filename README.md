# UAV Water Collection (MuJoCo + SMBPC)
MuJoCo-based quadrotor simulation for water-collection maneuvers using **Sampling-Based Model Predictive Control (SBMPC)**. It is built for research with CPU execution by default and optional JAX-accelerated GPU rollouts.

## Project Overview
This project offers a high-fidelity simulation of a quadrotor performing water-dip/collection maneuvers under a sampling-based MPC controller. The code is structured for rapid experimentation: (i) physics come from MuJoCo,  (ii) control is modular (dynamics, costs, and samplers are pluggable), and (iii) large rollout budgets can be accelerated on GPU via JAX. It’s intended for benchmarking and ablation studies as well as iterative design like tuning costs and constraints, exploring sampler variants, and assessing stability, precision, and robustness of the UAV–load interaction.

## What you can do with this repo
- Probe UAV–load dynamics in simulation
- Develop and tune sampling-based control algorithms
- Benchmark CPU vs GPU rollout performance
- Lay groundwork for real-world validation of UAV water collection

## Key Features

- **Sampling-Based MPC (SBMPC):** Stochastic trajectory sampling with receding-horizon selection; easily tune costs and constraints.
- **High-fidelity physics:** MuJoCo contact & dynamics to model the dip/collection interaction.
- **JAX acceleration (optional):** GPU-accelerated rollouts for large sampling budgets.
- **Modular design:** Swap samplers, costs, and dynamics without rewriting the control loop.

---

## Repository Structure
```
UAV-Water-Collection/
├─ sbmpc/ # Core planning/control modules for sampling-based MPC
├─ examples/ # Ready-to-run simulation scripts & configs
├─ crazyflie_mission.py # Example mission/waypoint runner (experimental)
├─ crazyflie_mission_waypoints.txt
├─ environment.yml # Conda env (CPU/GPU variants via extras)
├─ requirements.txt # Pinned pip dependencies
├─ pyproject.toml # Package metadata + extras [cuda12, cuda12_local]
├─ LICENSE.txt # BSD-3-Clause
└─ README.md
```

## Quick Start
### Prerequisites
 - [Nvidia cuda toolkit](https://developer.nvidia.com/cuda-toolkit) installed system-wide (if you want to use a local CUDA version)
 - [miniforge](https://github.com/conda-forge/miniforge/releases)


### Setup Instructions
1) Clone the repository
```bash
git clone https://github.com/Kevintan99/UAV-Water-Collection.git
cd UAV-Water-Collection
```

2) Create environment (first time)
```bash
mamba env create -f environment.yml
```

3) Activate the environment
```bash
conda activate sbmpc
```

4) Install the package (Choose CPU or GPU depending on the CUDA settings of your machine)
- CPU-only
```bash
pip install -e .
```
- GPU with pip-installed CUDA libraries
```bash
pip install -e ".[cuda12]"
```
- GPU with locally installed CUDA
```bash
pip install -e ".[cuda12_local]"
```

5) Run the simulation
```bash
python examples/quadrotor_comparison.py
```

Refer to the Jax documentation for more details.

## Notes & Next Steps
- Explore/modify controller settings, costs, and constraints inside the sbmpc/ modules to try different sampling strategies.
- The crazyflie_mission.py script and crazyflie_mission_waypoints.txt show a waypoint-style flow you can adapt for field robots or hardware-in-the-loop experiments

## License
This project is licensed under the **BSD-3-Clause License**.  
See the [LICENSE.txt](./LICENSE.txt) file for details.


