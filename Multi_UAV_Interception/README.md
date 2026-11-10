# MultiUAV

A multi-agent reinforcement learning project in which UAVs cooperate to protect a leader drone and intercept an approaching dynamic obstacle in AirSim. The project uses MAPPO / RMAPPO policies and LiDAR observations, with experiments comparing raw LiDAR against motion- and latency-compensated LiDAR.

> This repository contains experimental code. The Unreal/AirSim simulation project, AirSim `settings.json`, and trained checkpoints are not included. Some entry points require import changes and checkpoint-path configuration before execution. These instructions are based on source inspection; execution with a running simulator has not been verified.

## Relationship to the original MAPPO implementation

This project builds on the publicly released **MAPPO/on-policy** implementation associated with:

Chao Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* ([paper](https://arxiv.org/abs/2103.01955); [original repository](https://github.com/marlbenchmark/on-policy)).

**The original MAPPO/on-policy code was written by the paper's authors, including Chao Yu, and the MAPPO contributors.** This repository extends and adapts their public implementation with AirSim multi-UAV environments, LiDAR observations and compensation, and UAV training and evaluation code. Credit and copyright for the original implementation belong to its original authors and contributors. The original MIT license and copyright notice are retained; see [LICENSE](on-policy/LICENSE).

## Features

- Cooperative continuous control with a shared policy for three follower drones by default
- MAPPO and recurrent-policy RMAPPO training
- Shared implementations for raw and motion-compensated LiDAR environments, plus experimental variants
- Statistics for interception success, leader hits, friendly collisions, and distance cutoffs
- Checkpoint saving, TensorBoard, and Weights & Biases logging
- Trained Actor evaluation over 100 episodes by default, with JSON output

## Repository structure

```text
MultiUAV/
|-- README.md
`-- on-policy/
    |-- train_UAV.py                   # General training; environment import needs updating
    |-- train_UAV_LiDAR_Raw.py         # Raw LiDAR training
    |-- eval_LiDAR_Raw.py              # Raw LiDAR evaluation; environment module is missing
    |-- eval_time_correction.py        # Latency-compensated evaluation
    |-- requirements.txt              # Legacy framework dependencies
    |-- environment.yaml              # Legacy Linux / Python 3.6 environment
    |-- setup.py
    |-- LICENSE
    `-- onpolicy/
        |-- config_UAV.py             # UAV training arguments
        |-- algorithms/               # RMAPPO, HAPPO, HATRPO, and MAT implementations
        |-- envs/                     # AirSim training and evaluation environments
        |-- runner/shared/uav_runner.py
        `-- utils/shared_buffer_uav.py
```

The UAV training entry points support only `rmappo` and `mappo`. Other algorithms and benchmark scripts are included, but are not supported by these UAV entry points.

## Environment setup

Run a separate AirSim multirotor simulator. The default connection settings and object names are:

| Item | Default |
| --- | --- |
| Server | `127.0.0.1:41451` |
| Leader | `Drone1` |
| Followers | `Follower0`, `Follower1`, `Follower2` |
| Dynamic obstacle | `DynamicObstacle` |
| Follower LiDAR sensors | `Follower0_LidarSensor`, `Follower1_LidarSensor`, `Follower2_LidarSensor` |
| Training control interval | `dt=0.15` seconds |
| Fixed Z coordinate | `-10.0` in AirSim's NED coordinate system |

Match the simulator's vehicle, object, and LiDAR sensor names to the code. Check the sensor configuration, coordinate frame, and mounting position as well. The shared training environment uses a default `lidar_offset` of `(0.0, 0.0, 0.3)`.

### Python dependencies

The main packages are `torch`, `numpy`, `gym`, `gymnasium`, `airsim`, `tensorboardX`, and `wandb`. The latency-compensated evaluation environment also uses `scikit-learn`. Training entry points optionally use `setproctitle`.

`requirements.txt` and `environment.yaml` describe the legacy base framework. They do not list all dependencies used by the current UAV code, including AirSim and Gymnasium, and are not a directly reproducible Windows installation specification. `setup.py` does not automatically install runtime dependencies either. Use a separate virtual environment and select versions compatible with your simulator and PyTorch installation.

The following checks required imports only; it does not validate training compatibility:

```powershell
cd on-policy
python -c "import torch, numpy, gym, gymnasium, airsim, tensorboardX, wandb; from sklearn.cluster import DBSCAN; print('imports OK')"
```

Running scripts directly from `on-policy/` makes the local `onpolicy` package available. For an editable installation, see the installation note below.

## Important notes before running

1. **General training environment import:** `train_UAV.py` references `onpolicy.envs.make_env`, which is missing. To use the compensated environment, you can change the import to `AirSimMultiDroneEnv` from `onpolicy.envs.make_env_fin`. Equivalence with the original missing environment must be checked.
2. **Missing raw evaluation environment:** `eval_LiDAR_Raw.py` references the missing `onpolicy.envs.make_env_no_time_correction` module. Restore that module or explicitly select an evaluation environment whose observation format matches the checkpoint.
3. **Training output path:** The default `base_dir` is `results/corrected` in `train_UAV.py` and `results/raw_lidar` in `train_UAV_LiDAR_Raw.py`, relative to the working directory. Edit `base_dir` in `main()` to change it. `config_UAV.py` does not register `--results_dir`, so the path cannot currently be set through that CLI argument.
4. **Checkpoint path:** Replace the `/your/directory/actor.pt` placeholder in `actor_model_path` near the top of each evaluation script with your own checkpoint path. Match the observation dimension, hidden size, and RNN configuration to those used during training.
5. **Package installation directory:** `setup.py` reads `on-policy/README.md`, which is included. Run `python -m pip install -e .` from `on-policy/` for an editable installation. Runtime dependencies must be installed separately.

Observation dimensions and operating conditions can differ between environment files. When connecting an existing model to an experimental environment such as `eval_env_control_raw12.py`, verify compatibility rather than relying on file names.

## Training

Start the simulator and check the output paths described above, then run these commands from `on-policy/`:

```powershell
# Raw LiDAR + RMAPPO, with local TensorBoard logging
python train_UAV_LiDAR_Raw.py --algorithm_name rmappo --experiment_name raw_lidar --num_agents 3 --num_env_steps 20000000 --use_wandb

# Raw LiDAR + MAPPO
python train_UAV_LiDAR_Raw.py --algorithm_name mappo --experiment_name raw_mappo --use_wandb
```

For compensated training, first update the environment import in `train_UAV.py`, then run:

```powershell
python train_UAV.py --algorithm_name rmappo --experiment_name corrected_lidar --use_wandb
```

**Argument behavior:** The parser defines `--use_wandb` and `--cuda` with `store_false`. W&B is enabled by default, and passing `--use_wandb` disables it. GPU use is enabled by default, and passing `--cuda` selects the CPU. Training also falls back to the CPU when no GPU is available.

| Argument | Default | Description |
| --- | --- | --- |
| `--algorithm_name` | `rmappo` | `rmappo` or `mappo` |
| `--experiment_name` | `drone_formation_v1` | Experiment name |
| `--seed` | `1` | Random seed |
| `--num_agents` | `3` | Follower count; must match the simulator setup |
| `--ip_address` | `127.0.0.1` | AirSim server IP |
| `--num_env_steps` | `20000000` | Total training steps |
| `--rollout_length` | `2048` | Steps collected per policy update |
| `--episode_length` | `2048` | Buffer-related length setting |
| `--hidden_size` | `64` | Policy hidden size |
| `--lr` / `--critic_lr` | `3e-4` / `5e-4` | Actor / Critic learning rates |
| `--model_dir` | `None` | Existing model directory |

The AirSim training entry points force the rollout thread count to one. Parameters such as `dt`, `port`, `fixed_z`, `leader_name`, and `dynamic_name` are not registered in the CLI parser; change the entry point or environment construction code to configure them. Although `step_length`, `leader_velocity`, and `optimal_distance` exist in the parser, the current training entry points do not pass them to the environment constructor.

## LiDAR experiment configuration

`make_env_baseline.py` and `make_env_fin.py` share `make_env_lidar_common.py`, setting `ENABLE_LIDAR_CORRECTION` to `False` and `True`, respectively.

With three followers, the shared training environment defines a 10-dimensional observation per agent: relative leader XY, relative ally XY, relative enemy XY, and the agent's own XY velocity. The shared Critic observation has 30 dimensions. The continuous action has two dimensions and is used for XY velocity commands.

The shared training environment uses sparse rewards: `+1` for successful interception and `-1` for a leader hit, friendly collision, or distance cutoff. The interception radius starts at 15 m and decreases by 0.1 m after every 20 consecutive successes at the current stage, down to a minimum of 1.5 m. Separate evaluation environments may use different radii or observation formats.

## Evaluation and outputs

Prepare the evaluation environment import and `actor_model_path`, then run:

```powershell
# After checking the checkpoint and compensated evaluation environment settings
python eval_time_correction.py

# After restoring the missing raw evaluation environment or selecting a compatible one
python eval_LiDAR_Raw.py
```

Both scripts evaluate a deterministic policy for 100 episodes by default and initialize W&B. Change the episode count and environment settings inside the scripts. They report defense success rate (DSR), friendly collision rate (CR), leader hit rate (LHR), distance cutoffs, and other statistics, and write `lidar_eval_results.json` to the current working directory. Both use the same output file name, so save results separately when comparing experiments.

Training outputs are stored under `<env_name>/<algorithm_name>/<experiment_name>/` within the configured `base_dir`.

- W&B enabled: `actor.pt` and `critic.pt` are saved in the W&B run directory.
- W&B disabled: checkpoints are saved in `models/`, and TensorBoard logs in `logs/`.

For experiment comparisons, record the environment file, observation dimension, sensor settings, compensation mode, seed, and evaluation conditions alongside the checkpoints. This repository does not include performance figures or completed reproduction results.

## License

The included on-policy framework is distributed under the [MIT License](on-policy/LICENSE). Base package metadata and copyright information are available in `on-policy/setup.py` and `on-policy/LICENSE`.
