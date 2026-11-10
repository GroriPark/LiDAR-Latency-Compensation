import gymnasium as gym
import numpy as np
from sklearn.cluster import DBSCAN
import airsim
import math
import time
import random


class AirSimMultiDroneEvalEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",
        follower_names=("Follower0", "Follower1", "Follower2"),
        port=41451,
        dt=0.15,
        fixed_z=-10.0,
        do_visualize=False,
        dynamic_name="DynamicObstacle",
        leader_name="Drone1",
        fixed_hit_radius=3.0,
        max_episode_steps=500,
    ):
        super().__init__()

        self.total_step_count = 0
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]
        self.leader_name = leader_name
        self.dynamic_name = dynamic_name

        self.fixed_z = float(fixed_z)
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)
        self.FAR_CUTOFF = 60.0
        self.max_episode_steps = int(max_episode_steps)

        self.step_count = 0
        self.episode_count = 0
        self._last_visualize_t = time.time()

        # action: [vx, vy] normalized in [-1, 1]
        self.vmax_xy = 10.0

        # faster control timings
        self.action_duration = self.dt
        self.enemy_duration = self.dt
        self.stabilize_duration = min(0.15, self.dt)
        self.visualize_period = 0.2

        # termination / reward thresholds
        self.COLLISION_THRESHOLD = 1.0
        self.LEADER_HIT_RADIUS = 1.5

        # ===== 평가용 고정 요격 반경 =====
        self.hit_radius = float(fixed_hit_radius)

        # ===== 평가 로그용 =====
        self.final_interceptor = None
        self.final_intercept_distance = None
        self.final_status = "RUNNING"

        # LiDAR configuration
        self.Lidar_names = {agent: f"{agent}_LidarSensor" for agent in self.possible_agents}
        self.lidar_config = {
            "rotations_per_second": 10.0,
            "max_range": 80.0,
        }
        self.dbscan_eps = 2.5
        self.dbscan_min_samples = 8
        self.poly_degree = 2

        # tracking memory
        self.enemy_history = {}
        self.enemy_memory = {}

        self.start_location = {}
        self.current_location = {}

        # ROI-based target tracking
        self.roi_base_radius = 6.0
        self.roi_vel_gain = 1.5
        self.roi_max_radius = 15.0
        self.roi_min_points = 3
        self.history_maxlen = 10
        self.bootstrap_global_steps = 2

        # normalization scales
        self.rel_pos_scale_xy = 80.0
        self.rel_pos_scale_z = 20.0
        self.closing_speed_scale = 20.0
        self.los_rate_scale = 1.0
        self.eps_norm = 1e-6

        # obs:
        # los_rate : 1
        # closing_speed : 1
        # curr_rel : 2
        # future1,2,3 : 2,2,2 => 6
        # ally_rel : 2
        # manned_rel : 2
        obs_dim = 14
        obs_low = -np.ones(obs_dim, dtype=np.float32)
        obs_high = np.ones(obs_dim, dtype=np.float32)
        share_obs_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=obs_low,
                high=obs_high,
                shape=(obs_dim,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.tile(obs_low, len(self.possible_agents)),
            high=np.tile(obs_high, len(self.possible_agents)),
            shape=(share_obs_dim,),
            dtype=np.float32,
        )

        self.client = airsim.MultirotorClient(ip=ip_address, port=port)
        self.client.confirmConnection()

    # ============================================================
    # spaces
    # ============================================================
    @property
    def observation_space(self):
        return [self.observation_spaces[a] for a in self.possible_agents]

    @property
    def action_space(self):
        return [self.action_spaces[a] for a in self.possible_agents]

    @property
    def share_observation_space(self):
        return [self.share_observation_spaces for _ in self.possible_agents]

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # ============================================================
    # normalization helpers
    # ============================================================
    def _clip_div(self, x, scale):
        return float(np.clip(x / (scale + self.eps_norm), -1.0, 1.0))

    def _normalize_vec3(self, v):
        v = np.asarray(v, dtype=np.float32)
        return np.array([
            self._clip_div(v[0], self.rel_pos_scale_xy),
            self._clip_div(v[1], self.rel_pos_scale_xy),
            self._clip_div(v[2], self.rel_pos_scale_z),
        ], dtype=np.float32)

    def _normalize_vec2(self, v):
        v = np.asarray(v, dtype=np.float32)
        return np.array([
            self._clip_div(v[0], self.rel_pos_scale_xy),
            self._clip_div(v[1], self.rel_pos_scale_xy),
        ], dtype=np.float32)

    def _normalize_scalar(self, x, scale):
        return np.array([self._clip_div(x, scale)], dtype=np.float32)

    # ============================================================
    # info helpers
    # ============================================================
    def _build_step_info(self, agent_name, status, reward=0.0):
        enemy_rel = self._relative_position_world(agent_name, self.dynamic_name)
        enemy_dist_xy = float(np.linalg.norm(enemy_rel[:2]))

        ally_rel = self._get_nearest_ally_relative_position(agent_name)
        nearest_ally_dist_xy = float(np.linalg.norm(ally_rel[:2]))

        leader_rel = self._relative_position_world(agent_name, self.leader_name)
        leader_dist_xy = float(np.linalg.norm(leader_rel[:2]))

        hist = self.enemy_history.get(agent_name, {}).get(self.dynamic_name, [])
        target_detected = len(hist) > 0

        episode_success = 1.0 if "SUCCESS" in status else 0.0
        episode_leader_hit = 1.0 if "LEADER_HIT" in status else 0.0
        episode_ally_collision = 1.0 if "ALLY_COLLISION" in status else 0.0
        episode_far_cutoff = 1.0 if "FAR_CUTOFF" in status else 0.0
        episode_timeout = 1.0 if "TIMEOUT" in status else 0.0

        return {
            "final_status": status,
            "reward": float(reward),
            "step_count": int(self.step_count),
            "total_step_count": int(self.total_step_count),

            "enemy_dist_xy": enemy_dist_xy,
            "nearest_ally_dist_xy": nearest_ally_dist_xy,
            "leader_dist_xy": leader_dist_xy,
            "target_detected": bool(target_detected),

            "fixed_hit_radius": float(self.hit_radius),
            "episode_success": episode_success,
            "episode_leader_hit": episode_leader_hit,
            "episode_ally_collision": episode_ally_collision,
            "episode_far_cutoff": episode_far_cutoff,
            "episode_timeout": episode_timeout,

            "episode_intercept_range": float(self.final_intercept_distance) if self.final_intercept_distance is not None else 0.0,
            "success_interceptor": self.final_interceptor if self.final_interceptor is not None else "",
        }

    # ============================================================
    # setup / reset
    # ============================================================
    def _stabilize_vehicle(self, name):
        self.client.moveByVelocityAsync(
            vx=0.0,
            vy=0.0,
            vz=0.0,
            duration=self.stabilize_duration,
            vehicle_name=name,
        ).join()

    def _setup_flight(self):
        names = [self.leader_name, self.dynamic_name] + self.possible_agents

        for name in names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)

        cmds = [self.client.takeoffAsync(vehicle_name=name) for name in names]
        for c in cmds:
            c.join()

        leader_pose = airsim.Pose(
            airsim.Vector3r(0.0, 0.0, self.fixed_z),
            airsim.Quaternionr(0, 0, 0, 1),
        )
        self.client.simSetVehiclePose(leader_pose, True, vehicle_name=self.leader_name)

        enemy_pose = airsim.Pose(
            airsim.Vector3r(60.0, 0.0, self.fixed_z),
            airsim.Quaternionr(0, 0, 0, 1),
        )
        self.client.simSetVehiclePose(enemy_pose, True, vehicle_name=self.dynamic_name)

        offsets = [(-5.0, 0.0), (-5.0, -5.0), (-5.0, 5.0)]
        for i, agent in enumerate(self.possible_agents):
            dx, dy = offsets[i % len(offsets)]
            pose = airsim.Pose(
                airsim.Vector3r(dx, dy, self.fixed_z),
                airsim.Quaternionr(0, 0, 0, 1),
            )
            self.client.simSetVehiclePose(pose, True, vehicle_name=agent)

        for name in names:
            self._stabilize_vehicle(name)

        self._get_current_location()

        self.start_location[self.leader_name] = self.current_location[self.leader_name]
        self.start_location[self.dynamic_name] = self.current_location[self.dynamic_name]
        for agent in self.possible_agents:
            self.start_location[agent] = self.current_location[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.episode_count += 1
        self.step_count = 0
        self.agents = self.possible_agents[:]

        self.final_interceptor = None
        self.final_intercept_distance = None
        self.final_status = "RUNNING"

        self.client.reset()
        self._setup_flight()
        self._teleport_obstacle_randomly()
        self._get_current_location()

        if self.do_visualize:
            self.client.simFlushPersistentMarkers()

        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(2, 4)

        self.enemy_history = {a: {self.dynamic_name: []} for a in self.possible_agents}
        self.enemy_memory = {a: {self.dynamic_name: np.zeros(3, dtype=np.float32)} for a in self.possible_agents}

        print(
            f"[RESET] episode={self.episode_count}, "
            f"fixed_hit_radius={self.hit_radius:.2f}"
        )

        return [self._get_obs(agent) for agent in self.possible_agents]

    # ============================================================
    # cached geometry helpers
    # ============================================================
    def _get_current_location(self):
        self.current_location = {}
        self.current_location[self.leader_name] = self.client.simGetObjectPose(self.leader_name)
        self.current_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)
        for agent in self.possible_agents:
            self.current_location[agent] = self.client.simGetObjectPose(agent)

    def _get_time_now_s(self, vehicle_name):
        return self.client.getMultirotorState(vehicle_name=vehicle_name).timestamp * 1e-9

    def _relative_position_world(self, src_name, tgt_name):
        src = self.current_location[src_name].position
        tgt = self.current_location[tgt_name].position
        return np.array([
            tgt.x_val - src.x_val,
            tgt.y_val - src.y_val,
            tgt.z_val - src.z_val,
        ], dtype=np.float32)

    def _get_relative_position_api(self, agent_name, target_name):
        return self._relative_position_world(agent_name, target_name)

    def _get_nearest_ally_relative_position(self, agent_name):
        others = [a for a in self.possible_agents if a != agent_name]
        if not others:
            return np.zeros(3, dtype=np.float32)

        rels = [self._relative_position_world(agent_name, other) for other in others]
        dists = [np.linalg.norm(r[:2]) for r in rels]
        return rels[int(np.argmin(dists))].astype(np.float32)

    def _get_intercept_radius(self):
        return self.hit_radius

    # ============================================================
    # visualization / enemy motion
    # ============================================================
    def _update_leader_movement(self):
        if not self.do_visualize:
            return

        now = time.time()
        if (now - self._last_visualize_t) >= self.visualize_period:
            self.client.simFlushPersistentMarkers()
            self._visualize_circles()
            self._last_visualize_t = now

    def _visualize_circles(self):
        try:
            leader_pos = self.current_location[self.leader_name].position
            center = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val], dtype=float)

            def ring_points(radius, n=36):
                pts = []
                for i in range(n + 1):
                    ang = (i / n) * 2 * np.pi
                    x = center[0] + radius * np.cos(ang)
                    y = center[1] + radius * np.sin(ang)
                    z = center[2]
                    pts.append(airsim.Vector3r(x, y, z))
                return pts

            self.client.simPlotLineStrip(
                ring_points(self._get_intercept_radius()),
                [1, 0, 0, 0.8],
                10.0,
                0.15,
                True,
            )
        except Exception:
            pass

    def _update_dynamic_obstacle(self):
        self._obs_step_timer += 1

        if self._obstacle_state == "IDLE":
            self.client.moveByVelocityAsync(
                0.0, 0.0, 0.0,
                duration=self.enemy_duration,
                vehicle_name=self.dynamic_name,
            )
            if self._obs_step_timer >= self._idle_wait_steps:
                self._obstacle_state = "ATTACK"
                self._obs_step_timer = 0
            return

        if self._obstacle_state == "ATTACK":
            try:
                leader_pos = self.current_location[self.leader_name].position
                obs_pos = self.current_location[self.dynamic_name].position
                diff = np.array([
                    leader_pos.x_val - obs_pos.x_val,
                    leader_pos.y_val - obs_pos.y_val,
                    leader_pos.z_val - obs_pos.z_val,
                ], dtype=np.float32)
                dist = np.linalg.norm(diff)
                if dist > 1e-6:
                    vel = (diff / dist) * 3.0
                    self.client.moveByVelocityZAsync(
                        vx=float(vel[0]),
                        vy=float(vel[1]),
                        z=self.fixed_z,
                        duration=self.enemy_duration,
                        vehicle_name=self.dynamic_name,
                    )
            except Exception:
                pass

            if self._obs_step_timer > int(500 / max(self.dt, 1e-3)):
                self._reset_obstacle_logic()

    def _teleport_obstacle_randomly(self):
        leader_pos = self.current_location[self.leader_name].position
        radius = random.uniform(30.0, 35.0)
        angle = random.uniform(0.0, 2.0 * math.pi)
        tx = leader_pos.x_val + radius * math.cos(angle)
        ty = leader_pos.y_val + radius * math.sin(angle)
        tz = self.fixed_z

        pose = airsim.Pose(
            airsim.Vector3r(tx, ty, tz),
            airsim.Quaternionr(0, 0, 0, 1),
        )
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.dynamic_name)
        self._stabilize_vehicle(self.dynamic_name)

    def _reset_obstacle_logic(self):
        self._teleport_obstacle_randomly()
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(2, 4)

    # ============================================================
    # LiDAR pipeline
    # ============================================================
    def _get_PointCloud(self, agent_name):
        lidar_name = self.Lidar_names[agent_name]
        lidar_data = self.client.getLidarData(lidar_name, vehicle_name=agent_name)
        if len(lidar_data.point_cloud) < 3:
            return None, None, None
        xyz = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        return xyz, lidar_data.time_stamp, len(xyz)

    def _estimate_PointTimes(self, xyz, time_end_ns):
        n = len(xyz)
        if n == 0:
            return np.array([], dtype=np.float64)

        scan_duration = 1.0 / self.lidar_config["rotations_per_second"]
        time_end_s = time_end_ns * 1e-9
        time_start_s = time_end_s - scan_duration

        if n == 1:
            return np.array([time_end_s], dtype=np.float64)

        return np.linspace(time_start_s, time_end_s, n, dtype=np.float64)

    def _estimate_sensor_positions(self, agent_name, point_times, ref_time_s):
        state = self.client.getMultirotorState(vehicle_name=agent_name)
        p_ref = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val,
        ], dtype=np.float64)
        v_ref = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val,
        ], dtype=np.float64)

        dt = (ref_time_s - point_times).reshape(-1, 1)
        sensor_positions = p_ref.reshape(1, 3) - v_ref.reshape(1, 3) * dt
        return sensor_positions, p_ref

    def _update_enemy_history(self, agent_name, enemy_name, rel_pos, t_now):
        self.enemy_history.setdefault(agent_name, {})
        self.enemy_history[agent_name].setdefault(enemy_name, [])

        self.enemy_history[agent_name][enemy_name].append({
            "t": float(t_now),
            "rel": np.array(rel_pos, dtype=np.float32),
        })

        if len(self.enemy_history[agent_name][enemy_name]) > self.history_maxlen:
            self.enemy_history[agent_name][enemy_name] = self.enemy_history[agent_name][enemy_name][-self.history_maxlen:]

        self.enemy_memory.setdefault(agent_name, {})
        self.enemy_memory[agent_name][enemy_name] = np.array(rel_pos, dtype=np.float32)

    def _get_enemy_history_array(self, agent_name, enemy_name):
        hist = self.enemy_history.get(agent_name, {}).get(enemy_name, [])
        if len(hist) == 0:
            return np.array([], dtype=np.float64), np.empty((0, 3), dtype=np.float32)

        times = np.array([h["t"] for h in hist], dtype=np.float64)
        rels = np.array([h["rel"] for h in hist], dtype=np.float32)
        return times, rels

    def _get_cv_fallback_enemy_position(self, agent_name, enemy_name):
        times, rels = self._get_enemy_history_array(agent_name, enemy_name)

        if len(times) >= 2:
            dt_hist = max(times[-1] - times[-2], 1e-3)
            v_rel = (rels[-1] - rels[-2]) / dt_hist
            pred = rels[-1] + v_rel * self.dt
            return pred.astype(np.float32)

        if len(rels) == 1:
            return rels[-1].astype(np.float32)

        return np.zeros(3, dtype=np.float32)

    def _get_gt_compensated_pointcloud(self, agent_name, enemy_name, gt_radius=4.0):
        xyz_raw, time_end_ns, _ = self._get_PointCloud(agent_name)
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
        xyz_gt_raw = xyz_raw[d2 <= (gt_radius ** 2)]

        if len(xyz_gt_raw) < self.roi_min_points:
            return xyz_raw.astype(np.float32), None, gt_rel.astype(np.float32), gt_radius

        ref_time_s = time_end_ns * 1e-9
        point_times = self._estimate_PointTimes(xyz_gt_raw, time_end_ns)
        sensor_positions, p_ref = self._estimate_sensor_positions(agent_name, point_times, ref_time_s)
        xyz_gt_comp = xyz_gt_raw.astype(np.float64) + (p_ref.reshape(1, 3) - sensor_positions)

        return (
            xyz_gt_raw.astype(np.float32),
            xyz_gt_comp.astype(np.float32),
            gt_rel.astype(np.float32),
            gt_radius,
        )

    def _estimate_enemy_from_gt_points(self, agent_name, enemy_name):
        _, xyz_gt_comp, _, _ = self._get_gt_compensated_pointcloud(
            agent_name, enemy_name, gt_radius=4.0
        )

        fallback = self._get_cv_fallback_enemy_position(agent_name, enemy_name)

        if xyz_gt_comp is None or len(xyz_gt_comp) < self.roi_min_points:
            return fallback.astype(np.float32), False

        center = np.mean(xyz_gt_comp, axis=0)
        return center.astype(np.float32), True

    def _get_current_enemy_relative_position(self, agent_name, enemy_name):
        rel_pos, detected = self._estimate_enemy_from_gt_points(agent_name, enemy_name)
        return rel_pos.astype(np.float32), detected

    def _predict_future_enemy_positions(self, agent_name, enemy_name, future_steps=(0.15, 0.3, 0.45)):
        times, rels = self._get_enemy_history_array(agent_name, enemy_name)
        if len(times) == 0:
            return [np.zeros(3, dtype=np.float32) for _ in future_steps]
        if len(times) < 3:
            last_rel = rels[-1]
            return [last_rel.copy() for _ in future_steps]

        times = times[-10:]
        rels = rels[-10:]

        t0 = times[0]
        t_rel = times - t0
        t_now = t_rel[-1]
        deg = min(self.poly_degree, len(times) - 1)

        try:
            poly_x = np.poly1d(np.polyfit(t_rel, rels[:, 0], deg=deg))
            poly_y = np.poly1d(np.polyfit(t_rel, rels[:, 1], deg=deg))
            poly_z = np.poly1d(np.polyfit(t_rel, rels[:, 2], deg=deg))
            preds = []
            for dt_future in future_steps:
                tq = t_now + dt_future
                preds.append(np.array([poly_x(tq), poly_y(tq), poly_z(tq)], dtype=np.float32))
            return preds
        except Exception:
            last_rel = rels[-1]
            return [last_rel.copy() for _ in future_steps]

    def _calculate_enemy_dynamics(self, agent_name, enemy_name):
        times, rels = self._get_enemy_history_array(agent_name, enemy_name)
        if len(times) < 2:
            return 0.0, 0.0

        curr = rels[-1]
        prev = rels[-2]
        dt = max(times[-1] - times[-2], 1e-3)
        v_rel = (curr - prev) / dt

        r = curr[:2].astype(np.float64)
        v = v_rel[:2].astype(np.float64)
        dist = np.linalg.norm(r) + 1e-6

        closing_speed = -float(np.dot(r, v)) / dist
        los_rate = float(r[0] * v[1] - r[1] * v[0]) / (dist ** 2)
        return los_rate, closing_speed

    def _build_enemy_observation(self, agent_name, enemy_name, t_now):
        curr_rel, detected = self._get_current_enemy_relative_position(agent_name, enemy_name)
        self._update_enemy_history(agent_name, enemy_name, curr_rel, t_now)

        _, rels = self._get_enemy_history_array(agent_name, enemy_name)

        if len(rels) >= 1:
            curr = rels[-1]
        else:
            curr = np.zeros(3, dtype=np.float32)

        fut1, fut2, fut3 = self._predict_future_enemy_positions(
            agent_name, enemy_name, future_steps=(0.15, 0.3, 0.45)
        )
        los_rate, closing_speed = self._calculate_enemy_dynamics(agent_name, enemy_name)

        return {
            "detected": bool(detected),
            "los_rate": float(los_rate),
            "closing_speed": float(closing_speed),
            "curr_rel": curr[:2].astype(np.float32),
            "future_rel_1": fut1[:2].astype(np.float32),
            "future_rel_2": fut2[:2].astype(np.float32),
            "future_rel_3": fut3[:2].astype(np.float32),
        }

    def _build_full_observation(self, agent_name, enemy_name, t_now):
        enemy_obs = self._build_enemy_observation(agent_name, enemy_name, t_now)
        ally_rel = self._get_nearest_ally_relative_position(agent_name)
        manned_rel = self._relative_position_world(agent_name, self.leader_name)

        return {
            "enemy": enemy_obs,
            "ally_rel": ally_rel.astype(np.float32),
            "manned_rel": manned_rel.astype(np.float32),
        }

    def _flatten_observation(self, obs_dict):
        e = obs_dict["enemy"]

        los_rate = self._normalize_scalar(e["los_rate"], self.los_rate_scale)
        closing_speed = self._normalize_scalar(e["closing_speed"], self.closing_speed_scale)

        curr_rel = self._normalize_vec2(e["curr_rel"])
        future_rel_1 = self._normalize_vec2(e["future_rel_1"])
        future_rel_2 = self._normalize_vec2(e["future_rel_2"])
        future_rel_3 = self._normalize_vec2(e["future_rel_3"])

        ally_rel = self._normalize_vec2(obs_dict["ally_rel"][:2])
        manned_rel = self._normalize_vec2(obs_dict["manned_rel"][:2])

        obs = np.concatenate([
            los_rate,
            closing_speed,
            curr_rel,
            future_rel_1,
            future_rel_2,
            future_rel_3,
            ally_rel,
            manned_rel,
        ], axis=0).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _get_obs(self, agent_name):
        t_now = self._get_time_now_s(agent_name)
        obs_dict = self._build_full_observation(agent_name, self.dynamic_name, t_now)
        return self._flatten_observation(obs_dict)

    # ============================================================
    # action / reward / termination
    # ============================================================
    def _do_action(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, -1.0, 1.0)
        if actions.ndim == 1:
            actions = actions.reshape(len(self.possible_agents), 2)

        cmds = []
        for i, agent in enumerate(self.possible_agents):
            vx = float(actions[i][0] * self.vmax_xy)
            vy = float(actions[i][1] * self.vmax_xy)

            cmds.append(
                self.client.moveByVelocityZAsync(
                    vx=vx,
                    vy=vy,
                    z=self.fixed_z,
                    duration=self.action_duration,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                    vehicle_name=agent,
                )
            )

        for c in cmds:
            c.join()

    def _check_done_event(self):
        intercept_radius = self._get_intercept_radius()

        # 1) 요격 성공 (고정 3m)
        for agent in self.possible_agents:
            d_enemy = np.linalg.norm(self._relative_position_world(agent, self.dynamic_name))
            if d_enemy <= intercept_radius:
                self.final_interceptor = agent
                self.final_intercept_distance = float(d_enemy)
                self.final_status = "SUCCESS_INTERCEPT_FIXED_R3.0"
                return True, 1.0, self.final_status

        # 2) far cutoff 실패
        for agent in self.possible_agents:
            d_leader_xy = np.linalg.norm(
                self._relative_position_world(agent, self.leader_name)[:2]
            )
            if d_leader_xy > self.FAR_CUTOFF:
                self.final_status = f"FAIL_FAR_CUTOFF_{agent}"
                return True, -1.0, self.final_status

        # 3) 리더 피격
        d_leader_enemy = np.linalg.norm(
            self._relative_position_world(self.leader_name, self.dynamic_name)
        )
        if d_leader_enemy <= self.LEADER_HIT_RADIUS:
            self.final_status = "FAIL_LEADER_HIT"
            return True, -1.0, self.final_status

        # 4) 에이전트-리더 충돌
        for agent in self.possible_agents:
            d = np.linalg.norm(self._relative_position_world(agent, self.leader_name))
            if d <= self.COLLISION_THRESHOLD:
                self.final_status = f"FAIL_AGENT_LEADER_COLLISION_{agent}"
                return True, -1.0, self.final_status

        # 5) 아군끼리 충돌
        for i in range(len(self.possible_agents)):
            for j in range(i + 1, len(self.possible_agents)):
                a_i = self.possible_agents[i]
                a_j = self.possible_agents[j]
                d = np.linalg.norm(self._relative_position_world(a_i, a_j))
                if d <= self.COLLISION_THRESHOLD:
                    self.final_status = f"FAIL_ALLY_COLLISION_{a_i}_{a_j}"
                    return True, -1.0, self.final_status

        return False, 0.0, "RUNNING"

    def _end_episode(self, reward, status):
        obs_list = [self._get_obs(agent) for agent in self.possible_agents]
        rewards = [float(reward) for _ in self.possible_agents]
        terminations = [True for _ in self.possible_agents]
        infos = [self._build_step_info(agent, status, reward) for agent in self.possible_agents]
        return obs_list, rewards, terminations, infos

    def step(self, actions):
        self.step_count += 1
        self.total_step_count += 1

        if self.step_count >= self.max_episode_steps:
            self.final_status = "FAIL_TIMEOUT"
            return self._end_episode(-1.0, self.final_status)

        self._do_action(actions)
        self._update_dynamic_obstacle()
        self._get_current_location()
        self._update_leader_movement()

        done, reward, status = self._check_done_event()
        if done:
            return self._end_episode(reward, status)

        obs_list = [self._get_obs(agent) for agent in self.possible_agents]
        rewards = [0.0 for _ in self.possible_agents]
        terminations = [False for _ in self.possible_agents]
        infos = [self._build_step_info(agent, status, 0.0) for agent in self.possible_agents]
        return obs_list, rewards, terminations, infos