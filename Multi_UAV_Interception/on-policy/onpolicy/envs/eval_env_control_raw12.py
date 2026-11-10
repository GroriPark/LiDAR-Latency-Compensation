"""Raw training observations on the same physical scenario as corrected14."""
import gymnasium as gym
import numpy as np

from onpolicy.envs.eval_env_control_common import AirSimMultiDroneEvalEnv as SharedEnv
from onpolicy.envs.eval_env_raw12_source import AirSimMultiDroneEnv as RawTrainingEnv


class AirSimMultiDroneEvalEnv(SharedEnv):
    EXPECTED_OBS_DIM = 12

    _get_lidar_measurement = RawTrainingEnv._get_lidar_measurement
    _calculate_lidar_dynamics = RawTrainingEnv._calculate_lidar_dynamics

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.leader_name != "Drone1":
            raise ValueError("Raw training observations require leader_name='Drone1'")
        self.MAX_SPEED = self.vmax_xy
        self.lidar_names = self.Lidar_names
        obs_dim = 2 + 2 * (len(self.possible_agents) - 1) + 4 + 2
        self.EXPECTED_OBS_DIM = obs_dim
        self.observation_spaces = {
            a: gym.spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.share_observation_spaces = gym.spaces.Box(
            -1.0, 1.0, shape=(obs_dim * len(self.possible_agents),), dtype=np.float32)
        self._reset_raw_memory()

    def _reset_raw_memory(self):
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        for agent in self.possible_agents:
            targets = [self.leader_name, self.dynamic_name] + [
                other for other in self.possible_agents if other != agent]
            self.lidar_memory[agent] = {target: [1.0, 1.0] for target in targets}
            self.prev_lidar_pos[agent] = {target: [100.0, 100.0] for target in targets}

    def reset(self, seed=None, options=None):
        self._reset_raw_memory()
        return super().reset(seed=seed, options=options)

    def _get_obs(self, agent):
        obs = RawTrainingEnv._get_obs(self, agent)
        # Training step updates previous enemy position after _get_obs.
        # Reset's initial observation leaves the default previous position.
        if self.step_count > 0:
            ex, ey = self.lidar_memory[agent][self.dynamic_name]
            self.prev_lidar_pos[agent][self.dynamic_name] = [ex * 100.0, ey * 100.0]
        return obs
