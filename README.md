<div align="center">

# DGPPO

[![Conference](https://img.shields.io/badge/ICLR-Accepted-success)](https://mit-realm.github.io/dgppo/)

Jax official implementation of ICLR2025 paper: [Songyuan Zhang](https://syzhang092218-source.github.io), [Oswin So](https://oswinso.xyz/), [Mitchell Black](https://www.blackmitchell.com/), and [Chuchu Fan](https://chuchu.mit.edu): "[Discrete GCBF Proximal Policy Optimization for Multi-agent Safe Optimal Control](https://mit-realm.github.io/dgppo/)". 

[Dependencies](#Dependencies) •
[Installation](#Installation) •
[Quickstart](#Quickstart) •
[Environments](#Environments) •
[Algorithms](#Algorithms) •
[Usage](#Usage)

</div>

<div align="center">
    <img src="./media/lidar_env/video/LidarSpread.gif" alt="LidarSpread" width="24.55%"/>
    <img src="./media/lidar_env/video/LidarLine.gif" alt="LidarLine" width="24.55%"/>
    <img src="./media/vmas/video/VMASReverseTransport.gif" alt="VMASReverseTransport" width="24.55%"/>
    <img src="./media/vmas/video/VMASWheel.gif" alt="VMASWheel" width="24.55%"/>
</div>

<div align="center">
    <img src="./media/dgppo.png" alt="DGPPO Framework" width="100%"/>
</div>

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n dgppo python=3.10
conda activate dgppo
```

Then install the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install DGPPO: 

```bash
pip install -e .
```

## Quickstart

To train a model on the `LidarSpread` environment, run:

```bash
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3
```

To evaluate a model, run:

```bash
python test.py --path ./logs/LidarSpread/dgppo/seed0_xxxxxxxxxx_XXXX
```

## Environments

We provide simulation environments across 3 different simulation engines. 

### MPE
We provide the following environments with the [MPE](https://github.com/openai/multiagent-particle-envs) (Multi-Agent Particle Environment) simulation engine: `MPETarget`, `MPESpread`, `MPEFormation`, `MPELine`, `MPECorridor`, `MPEConnectSpread`. In MPE, agents, goals/landmarks, and obstacles are represented as particles. Agents can observe other agents or obstacles when they are within their observation range. Agents follow the double integrator dynamics.

<div align="center">
    <img src="media/mpe/img/MPETarget.png" width="16.2%" alt="MPETarget">
    <img src="media/mpe/img/MPESpread.png" width="16.2%" alt="MPESpread">
    <img src="media/mpe/img/MPEFormation.png" width="16.2%" alt="MPEFormation">
    <img src="media/mpe/img/MPELine.png" width="16.2%" alt="MPELine">
    <img src="media/mpe/img/MPECorridor.png" width="16.2%" alt="MPECorridor">
    <img src="media/mpe/img/MPEConnectSpread.png" width="16.2%" alt="MPEConnectSpread">
    <img src="media/mpe/img/env_legend.png" alt="Legend" width="100%">
</div>

- `MPETarget`: The agents need to reach their pre-assigned goals.
- `MPESpread`: The agents need to collectively cover a set of goals without having access to an assignment.
- `MPEFormation`: The agents need to spread evenly around a given landmark.
- `MPELine`: The agents need to form a line between two given landmarks.
- `MPECorridor`: The agents need to navigate through a narrow corridor and cover a set of given goals.
- `MPEConnectSpread`: The agents need to cover a set of given goals while maintaining connectivity.

### LidarEnv
We provide the following environments with the [LidarEnv](https://github.com/MIT-REALM/gcbfplus/) simulation engine: `LidarTarget`, `LidarSpread`, `LidarLine`, `LidarBicycleTarget`. In these environments, agents use LiDAR sensors to observe the environments. Agents can observe other agents or obstacles when they are within their observation range. Agents follow the double integrator dynamics or the bicycle dynamics.

<div align="center">
    <img src="./media/lidar_env/img/LidarTarget.png" width="24.55%" alt="LidarTarget">
    <img src="./media/lidar_env/img/LidarSpread.png" width="24.55%" alt="LidarSpread">
    <img src="./media/lidar_env/img/LidarLine.png" width="24.55%" alt="LidarLine">
    <img src="./media/lidar_env/img/LidarBicycleTarget.png" width="24.55%" alt="LidarBicycleTarget">
</div>

- `LidarTarget`: The agents need to reach their pre-assigned goals.
- `LidarSpread`: The agents need to collectively cover a set of goals without having access to an assignment.
- `LidarLine`: The agents need to form a line between two given landmarks.
- `LidarBicycleTarget`: The agents (following the bicycle dynamics) need to reach their pre-assigned goals.

### VMAS
We provide the following environments with the [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) (VectorizedMultiAgentSimulator) simulation engine: `VMASReverseTransport`, `VMASWheel`. In these environments, contact dynamics are modeled. Agents follow the double integrator dynamics and have full observation.

<div align="center">
    <img src="./media/vmas/img/VMASReverseTransport.png" width="24.55%" alt="VMASReverseTransport">
    <img src="./media/vmas/img/VMASWheel.png" width="24.55%" alt="VMASWheel">
</div>

- `VMASReverseTransport`: The agents need to push a box from inside to its pre-assigned goals, while the center of the box must avoid the obstacles.
- `VMASWheel`: The agents need to push a wheel to its pre-assigned angle, while the wheel must avoid a range of dangerous angles.

### Custom Environments
It is very easy to create a custom environment by yourself! Choose one of the three engines, and inherit one of the existing environments. Define your reward function, graph connection, and dynamics, register the new environment in `env/__init__.py`, and you are good to go!

## Algorithms

We provide the following algorithms:

- `dgppo`: Discrete GCBF Proximal Policy Optimization.
- `informarl`: [MAPPO](https://github.com/marlbenchmark/on-policy) with GNN ([Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation](https://github.com/nsidn98/InforMARL/)).
- `informarl_lagr`: [MAPPO-Lagrangian](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) with GNN. Replaced the sum-over-time cost with max-over-time cost.
- `hcbfcrpo`: DGPPO but replace the learned GCBF with a hand-crafted CBF.

## Usage

### Train

To train the `<algo>` algorithm on the `<env>` environment with `<n_agent>` agents and `<n_obs>` obstacles, run:

```bash
python train.py --env <env> --algo <algo> -n <n_agent> --obs <n_obs>
```

The training logs will be saved in `logs/<env>/<algo>/seed<seed>_<timestamp>_<four random letters>`. We provide the following flags:

#### Required Flags

- `--env`: Environment. 
- `--algo`: Algorithm.
- `-n`: Number of agents.
- `--obs`: Number of obstacles.

#### For algorithms

- `--no-cbf-schedule`: [For dgppo and hcbfcrpo] Remove the CBF schedule, default False.
- `--cbf-weight`: [For dgppo and hcbrcrpo] Weight of the CBF loss, default 1.0.
- `--cbf-eps`: [For dgppo and hcbrcrpo] Epsilon of the CBF loss, default 0.01.
- `--alpha`: [For dgppo and hcbrcrpo] The class-$\kappa$ function, default 10.0.
- `--cost-weight`: [For informarl] Weight of the cost term in the reward, default 0.0.
- `--cost-schedule`: [For informarl] Use the cost schedule, default False.
- `--lagr-init`: [For informarl_lagr] Initial value of the Lagrangian multiplier, default 0.5.
- `--lr-lagr`: [For informarl_lagr] Learning rate of the Lagrangian multiplier, default 1e-7.
- `--clip-eps`: Clip epsilon, default 0.25.
- `--coef-ent`: Entropy coefficient, default 0.01.

#### For environments

- `--n-rays`: [For LidarEnv] Number of LiDAR rays, default 32.
- `--full-observation`: [For MPE and LidarEnv] Use full observation, default False.

#### Training options
- `--no-rnn`: Do not use RNN, default False. **Use this flag in the VMAS environments can accelerate the training**.
- `--n-env-train`: Number of environments for training, default 128. Decrease this number with `--batch-size` if the GPU memory is not enough.
- `--batch-size`: Batch size, default 16384.
- `--n-env-test`: Number of environments for testing (during training), default 32.
- `--log-dir`: Directory to save the logs, default `logs`.
- `--eval-interval`: Evaluation interval, default 50.
- `--eval-epi`: Number of episodes for evaluation, default 1.
- `--save-interval`: Save interval, default 50.
- `--seed`: Random seed, default 0.
- `--steps`: Number of training steps, default 200000.
- `--name`: Name of the experiment, default None.
- `--debug`: Debug mode, in which the logs will not be saved, default False.
- `--actor-gnn-layers`: Number of layers in the actor GNN, default 2.
- `--Vl-gnn-layers`: Number of layers in the $V^l$ GNN, default 2.
- `--Vh-gnn-layers`: Number of layers in the $V^h$ GNN, default 1.
- `--lr-actor`: Learning rate of the actor, default 3e-4. **Consider changing to 1e-5 for 1 agent.**
- `--lr-Vl`: Learning rate of $V^l$, default 1e-3. **Consider changing to 3e-4 for 1 agent.**
- `--lr-Vh`: Learning rate of $V^h$, default 1e-3. **Consider changing to 3e-4 for 1 agent.**
- `--rnn-layers`: Number of layers in the RNN, default 1.
- `--use-lstm`: Use LSTM, default False (use GRU).
- `--rnn-step`: Number of RNN steps in a chunk, default 16.

### Test

To test the learned model, use:

```bash
python test.py --path <path-to-log>
```

This evaluates the deterministic policy and reports the reward, min/max reward, cost, min/max cost, and safety rate. It restores the saved LiDAR ray count and observation mode, and generates videos in `<path-to-log>/videos`. Episodes start at offset zero; use `--seed` to select a different reproducible set of episodes. Use the following flags to customize the test:

#### Required Flags
`--path`: Path to the log.

#### Optional Flags
- `--epi`: Number of episodes for testing, default 5.
- `--no-video`: Do not generate videos, default False.
- `--step`: If this is given, evaluate the model at the given step, default None (evaluate the last model).
- `-n`: Number of agents, default as the same as training.
- `--obs`: Number of obstacles, default as the same as training.
- `--env`: Environment, default as the same as training.
- `--full-observation` / `--no-full-observation`: Override the observation mode; defaults to the saved training setting.
- `--cpu`: Use CPU only, default False.
- `--max-step`: Maximum number of steps for each episode, default None.
- `--log`: Log the results, default False.
- `--seed`: Random seed, default 1234.
- `--debug`: Debug mode.
- `--dpi`: DPI of the video, default 100.

## Citation

```
@inproceedings{zhang2025dgppo,
      title={Discrete {GCBF} Proximal Policy Optimization for Multi-agent Safe Optimal Control},
      author={Zhang, Songyuan and So, Oswin and Black, Mitchell and Fan, Chuchu},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
}
```
