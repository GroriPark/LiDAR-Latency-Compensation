"""Fixed-radius evaluation using the shared LiDAR dynamics and observations."""
from onpolicy.envs.make_env_lidar_common import AirSimMultiDroneEnv


class AirSimMultiDroneEvalEnv(AirSimMultiDroneEnv):
    def __init__(self, *, eval_success_radius=1.5, enable_lidar_correction=False, **kwargs):
        if eval_success_radius <= 0:
            raise ValueError("eval_success_radius must be positive")
        self.ENABLE_LIDAR_CORRECTION = bool(enable_lidar_correction)
        super().__init__(**kwargs)
        self.success_radius = float(eval_success_radius)

    def _update_success_curriculum(self, success=None):
    # 평가에서는 요격 반경을 고정합니다.
        pass
