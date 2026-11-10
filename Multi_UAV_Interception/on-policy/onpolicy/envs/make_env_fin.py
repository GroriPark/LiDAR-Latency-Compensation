"""Motion-compensated LiDAR entry point; all environment behavior is shared."""
from onpolicy.envs.make_env_lidar_common import AirSimMultiDroneEnv as _SharedEnv


class AirSimMultiDroneEnv(_SharedEnv):
    ENABLE_LIDAR_CORRECTION = True
