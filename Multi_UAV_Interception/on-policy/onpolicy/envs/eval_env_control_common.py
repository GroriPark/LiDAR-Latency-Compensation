"""Shared physical scenario for corrected14 and raw12 policy evaluation.

Both policies inherit initialization/enemy/termination/step from the same
supplied scenario. Only policy observations differ to match training.
"""
import math

import airsim
import numpy as np

from onpolicy.envs.eval_env_control_source import AirSimMultiDroneEvalEnv as SourceEnv


class AirSimMultiDroneEvalEnv(SourceEnv):
    EXPECTED_OBS_DIM = 14
    SCENARIO_VERSION = "shared_scenario_v2_norm10"

    def scenario_config(self):
        return {
            "version": self.SCENARIO_VERSION,
            "agents": self.possible_agents,
            "leader": self.leader_name, "enemy": self.dynamic_name,
            "dt_s": self.dt, "fixed_z_m": self.fixed_z,
            "max_speed_norm_m_s": self.vmax_xy,
            "action_limit": "component_clip_then_vector_norm_limit",
            "action_duration_s": self.action_duration,
            "enemy_duration_s": self.enemy_duration,
            "enemy_speed_m_s": 3.0,
            "follower_initial_offsets_xy_m": [[-5, 0], [-5, -5], [-5, 5]],
            "enemy_radius_range_m": [30, 35],
            "enemy_idle_steps_range": [2, 4],
            "hit_radius_m": self.hit_radius,
            "collision_radius_m": self.COLLISION_THRESHOLD,
            "leader_hit_radius_m": self.LEADER_HIT_RADIUS,
            "far_cutoff_xy_m": self.FAR_CUTOFF,
            "max_episode_steps": self.max_episode_steps,
        }

    def _do_action(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (len(self.possible_agents), 2) or not np.all(np.isfinite(actions)):
            raise ValueError("Expected finite (number of UAVs, 2) actions")
        clipped = np.clip(actions, -1.0, 1.0)
        commands = []
        for agent, action in zip(self.possible_agents, clipped):
            vx = float(action[0]) * self.vmax_xy
            vy = float(action[1]) * self.vmax_xy
            speed = math.hypot(vx, vy)
            if speed > self.vmax_xy:
                scale = self.vmax_xy / (speed + 1e-6)
                vx *= scale
                vy *= scale
            commands.append(self.client.moveByVelocityZAsync(
                vx=vx, vy=vy, z=self.fixed_z, duration=self.action_duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                vehicle_name=agent,
            ))
        for command in commands:
            command.join()
