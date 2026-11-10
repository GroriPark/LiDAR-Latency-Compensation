"""Raw ablation of eval_env_control_source: identical pipeline except deskew.

This is derived from the supplied environment, not a recovered historical Raw
training environment. Verify observation compatibility with the Raw checkpoint.
"""
import numpy as np

from onpolicy.envs.eval_env_control_source import AirSimMultiDroneEvalEnv as CorrectedEnv


class AirSimMultiDroneEvalEnv(CorrectedEnv):
    def _get_gt_compensated_pointcloud(self, agent_name, enemy_name, gt_radius=4.0):
        """Keep the same range/Z/GT ROI and fallback, but use uncorrected returns."""
        xyz_raw, _, _ = self._get_PointCloud(agent_name)
        if xyz_raw is None or len(xyz_raw) == 0:
            return None, None, None, None

        dist = np.linalg.norm(xyz_raw, axis=1)
        mask_range = dist < self.lidar_config.get("max_range", 80.0)
        mask_z = (xyz_raw[:, 2] > -10.0) & (xyz_raw[:, 2] < 10.0)
        xyz_raw = xyz_raw[mask_range & mask_z]
        if len(xyz_raw) == 0:
            return None, None, None, None

        gt_rel = self._relative_position_world(agent_name, enemy_name).astype(np.float32)
        d2 = np.sum((xyz_raw - gt_rel.reshape(1, 3)) ** 2, axis=1)
        xyz_gt_raw = xyz_raw[d2 <= gt_radius ** 2]
        if len(xyz_gt_raw) < self.roi_min_points:
            return xyz_raw.astype(np.float32), None, gt_rel, gt_radius

        # Base estimator reads tuple item 2 as the selected cloud.
        # Supply the same selected raw cloud there, with no temporal correction.
        selected = xyz_gt_raw.astype(np.float32)
        return selected, selected.copy(), gt_rel, gt_radius
