# MultiUAV on-policy

MAPPO / RMAPPO experimental code for cooperative multi-UAV control in AirSim.

The original MAPPO/on-policy code was written by Chao Yu and the other authors of *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, together with the MAPPO contributors ([paper](https://arxiv.org/abs/2103.01955), [original repository](https://github.com/marlbenchmark/on-policy)). This project extends and adapts their public code for AirSim UAV experiments and retains the original MIT license and copyright notice.

See the [project README](../README.md) for the project overview, environment setup, training and evaluation commands, and required changes before execution.

Main entry points:

- `train_UAV_LiDAR_Raw.py`: raw LiDAR training
- `train_UAV.py`: general training; the environment import currently needs updating
- `eval_time_correction.py`: latency-compensated evaluation
- `eval_LiDAR_Raw.py`: raw LiDAR evaluation; the referenced environment module is currently missing

Run training commands from this directory. Prepare the Python dependencies and a separate AirSim simulator first.

See [LICENSE](LICENSE) for licensing information.
