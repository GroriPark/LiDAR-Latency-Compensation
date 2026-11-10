"""Shared environment for raw and motion-compensated LiDAR experiments.

The entry-point subclasses differ only in ENABLE_LIDAR_CORRECTION.
All dynamics, ROI selection, observation order, rewards and statistics are shared.
"""
import gymnasium as gym
import numpy as np
import airsim
import math
import time
import random
from collections import deque


class AirSimMultiDroneEnv:
    ENABLE_LIDAR_CORRECTION = True

    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",
        follower_names=("Follower0", "Follower1", "Follower2"),
        port=41451,
        step_length=0.01,
        leader_velocity=1.0,
        optimal_distance=10.0,
        far_cutoff=60.0,
        too_close=0.5,
        dt=0.15,
        fixed_z=-10.0,
        dynamic_name="DynamicObstacle",
        leader_name="Drone1",
        do_visualize=True,
        correction_latency=None,
        lidar_offset=(0.0, 0.0, 0.3),
        total_observation_latency=0.181,
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 1.5
        self.STOP_DISTANCE_LEADER_OBSTACLE = 1.0

        # ===== 희소보상 + 커리큘럼 =====
        self.success_radius = 15.0              # 처음 요격 인정 반경
        self.success_radius_min = 1.5           # 너무 작아지는 것 방지용(원하면 0.5 등으로 바꿔도 됨)
        self.success_radius_decay = 0.1         # 연속 20번 성공 시 0.1m 감소
        self.curriculum_success_streak = 0  # Consecutive successes at current stage
        self.success_count_for_decay = 20       # 몇 번 성공할 때 줄일지
        self.curriculum_success_count = 0       # 누적 성공 횟수

        # 희소보상 값
        self.REWARD_HIT_ENEMY = 1.0             # 요격 성공
        self.REWARD_LEADER_HIT = -1.0           # 리더 피격
        self.REWARD_AGENT_CRASH = -1.0          # 아군 충돌(리더-에이전트, 에이전트-에이전트)
        self.REWARD_FAR_CUTOFF = -1.0           # 필요하면 이탈도 -1 유지

        # 속도/액션 버퍼
        self.vmax_self = 2.0
        self._timestep = float(dt)

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = float(fixed_z)
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)
        self.max_cmd_speed = self.step_length / self.dt
        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)
        self.dynamic_name = dynamic_name
        self.leader_name = leader_name
        # Lidar 메모리: { "AgentName": { "TargetName": [dx, dy] } }
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        # [추가] Lidar 센서 이름 매핑 (settings.json의 이름과 일치해야 함)
        self.lidar_names = {
            agent: f"{agent}_LidarSensor" for agent in self.possible_agents
        }

        # Total horizon is measured from scan START, not from scan end.
        # None selects the total-horizon model; an explicit correction_latency
        # retains the old API's scan-END-to-control meaning as an override.
        self.total_observation_latency = float(total_observation_latency)
        if not np.isfinite(self.total_observation_latency) or self.total_observation_latency < 0:
            raise ValueError("total_observation_latency must be finite and nonnegative")
        self.correction_latency = None if correction_latency is None else float(correction_latency)
        if self.correction_latency is not None and (
                not np.isfinite(self.correction_latency) or self.correction_latency < 0):
            raise ValueError("correction_latency must be finite and nonnegative")
        self.lidar_offset = np.asarray(lidar_offset, dtype=np.float64)
        if self.lidar_offset.shape != (3,):
            raise ValueError("lidar_offset must have three components")
        self.poly_degree = 2
        self.motion_history_samples = 10
        self.motion_history = {a: deque(maxlen=256) for a in self.possible_agents}
        self.target_memory = {a: {} for a in self.possible_agents}
        self.target_detected = {a: {} for a in self.possible_agents}
        self.Lidar_names = self.lidar_names
        self.lidar_config = {"rotations_per_second": 10.0, "max_range": 80.0}
        self.roi_min_points = 3
        self._measurement_cache = {}

        # States
        self.step_count = 0
        self.episode_count = 0

        # 통계 계산을 위한 Deque
        self.stats_history = {
            "win": deque(maxlen=20),
            "coll_leader": deque(maxlen=20),
            "coll_drone": deque(maxlen=20),
            "coll_obj": deque(maxlen=20)
        }

        # 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}

        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1
        self.K_enemy = 1
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy

        # ---- [수정] 관측 공간 범위 정의 (상대 좌표 도입) ----
        # 기존: Bearing(-1~1), Dist(0~200)
        # 변경: Rel_X(-1~1), Rel_Y(-1~1) 로 변경 (100m 기준 정규화)
        
        low_rel_pos = -1.0
        high_rel_pos = 1.0
        
        low_self_state = -1.0
        high_self_state = 1.0

        # [리더(2)] + [아군(2)*K] + [적(2)*K] + [self_state(2)]
        # 적(Enemy)의 경우: [rel_x, rel_y]
        per_agent_low = (
            [low_rel_pos, low_rel_pos] +                          # Leader (dx, dy)
            [low_rel_pos, low_rel_pos] * self.num_ally +          # Allies (dx, dy)
            [low_rel_pos, low_rel_pos] * self.num_enemy + # Enemy
            [low_self_state] * 2                                  # Self (vx, vy)
        )
        per_agent_high = (
            [high_rel_pos, high_rel_pos] +
            [high_rel_pos, high_rel_pos] * self.num_ally +
            [high_rel_pos, high_rel_pos] * self.num_enemy +
            [high_self_state] * 2
        )

        obs_dim = len(per_agent_low)
        share_obs_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.MAX_YAW = math.radians(90)
        self.MAX_SPEED = 10

        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float64,
            )
            for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,),
            dtype=np.float32,
        )

        # PN 보상용 버퍼들
        self._prev_d_leader_enemy = None
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._prev_los_angle = {a: None for a in self.possible_agents}

        # 위치 캐시
        self.start_location = {}
        self.current_location = {}

        # 클라이언트 셋업
        self.client = airsim.MultirotorClient(ip=ip_address, port=port)
        self.client.confirmConnection()

        self._last_visualize_t = time.time()

    def _relative_position_world(self, src_name, tgt_name):
        src = self.current_location[src_name].position
        tgt = self.current_location[tgt_name].position
        return np.array([
            tgt.x_val - src.x_val,
            tgt.y_val - src.y_val,
            tgt.z_val - src.z_val,
        ], dtype=np.float32)

    @staticmethod
    def _rotate_xy(values, yaw):
        result = np.array(values, dtype=np.float64, copy=True)
        c, s = np.cos(yaw), np.sin(yaw)
        result[..., 0] = c * values[..., 0] - s * values[..., 1]
        result[..., 1] = s * values[..., 0] + c * values[..., 1]
        return result

    def _record_motion(self, agent):
        state = self.client.getMultirotorState(vehicle_name=agent)
        kin = state.kinematics_estimated
        _, _, yaw = airsim.to_eularian_angles(kin.orientation)
        velocity = np.array([kin.linear_velocity.x_val,
                             kin.linear_velocity.y_val, 0.0])
        body_velocity = self._rotate_xy(velocity, -yaw)
        history = self.motion_history[agent]
        timestamp = int(state.timestamp)
        if history and timestamp < history[-1][0]:
            history.clear()
        if not history or timestamp > history[-1][0]:
            history.append((timestamp, body_velocity[0], body_velocity[1],
                            kin.angular_velocity.z_val))
        return state, float(yaw), velocity

    def fit_motion(self, history, observation_ns):
        # LiDAR 관측 timestamp보다 이전인 샘플만 선택한다(미래 데이터 제외).
        samples = [sample for sample in history if sample[0] < observation_ns][-self.motion_history_samples:]
        if len(samples) < self.motion_history_samples:
            return None
        # 큰 절대 timestamp 대신 스캔 관측 시점을 0초로 둔다.
        times = np.array([(sample[0] - observation_ns) * 1e-9 for sample in samples])
        values = np.array([sample[1:] for sample in samples], dtype=float)
        if np.any(np.diff(times) <= 0) or not np.isfinite(values).all():
            raise ValueError('Motion samples must have increasing timestamps and finite velocities')
        scale = max(float(np.max(np.abs(times))), 1e-9)
        design = np.polynomial.polynomial.polyvander(times / scale, self.poly_degree)
        coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
        if rank != self.poly_degree + 1:
            raise ValueError('Motion regression has insufficient independent timestamps')
        # 행: vx, vy, wz / 열: 상수항, 1차항, 2차항 (시간 단위는 초).
        return coefficients.T / scale ** np.arange(self.poly_degree + 1)

    def integrate_motion(self, coefficients, ti, tc):
        if tc < ti:
            raise ValueError('Control time must not precede point measurement time')
        integration_time = np.linspace(ti, tc, 101)
        poly = np.polynomial.polynomial
        vx = poly.polyval(integration_time, coefficients[0])
        vy = poly.polyval(integration_time, coefficients[1])

        # 식 (7): 점의 측정 시점부터 각 시점까지 누적된 상대 요 회전각
        yaw_integral = poly.polyint(coefficients[2])
        phi = poly.polyval(integration_time, yaw_integral) - poly.polyval(ti, yaw_integral)

        # 식 (8), (9)의 적분항
        velocity_x = (
            vx * np.cos(phi)
            - vy * np.sin(phi)
        )

        velocity_y = (
            vx * np.sin(phi)
            + vy * np.cos(phi)
        )

        # 식 (8), (9): ti 시점의 UAV 좌표계에서의 이동량
        delta_x = np.trapz(velocity_x, integration_time)
        delta_y = np.trapz(velocity_y, integration_time)

        # 식 (11): 전체 요 회전각
        delta_psi = phi[-1]

        return delta_x, delta_y, delta_psi

    def _control_time_from_scan_end(self, scan_time):
        """Return the control horizon on the scan-end-relative time axis.

        Default: ti in [-0.100, 0], tc=0.081 -> intervals [0.081, 0.181].
        Command duration dt=0.15 is independent and is not added here.
        """
        if not np.isfinite(scan_time) or scan_time <= 0:
            raise ValueError("scan_time must be finite and positive")
        if self.correction_latency is not None:
            return self.correction_latency
        control_time = self.total_observation_latency - scan_time
        if control_time < 0:
            raise ValueError("total_observation_latency must cover the complete scan")
        return control_time

    def _measure_targets(self, agent, targets, state, yaw):
        """GT selects object points only; observations use corrected returns.

        By default, compensate ego motion to scan_start + 0.181 seconds.
        At 10 Hz this equals scan_end + 0.081 seconds. Scan-end timestamps
        and uniformly ordered point times are assumptions of this model.
        This does not estimate the target's own motion during that delay.
        Missing returns/history hold the last measurement (initially 100m).
        Both variants use identical scan acquisition, GT ROI and readiness rules.
        """
        lidar = self.client.getLidarData(self.Lidar_names[agent], vehicle_name=agent)
        results = {t: self.target_memory[agent].get(
            t, np.array([100.0, 100.0, 0.0], dtype=np.float32)).copy() for t in targets}
        self.target_detected[agent] = {t: False for t in targets}
        if not lidar.point_cloud:
            return results
        if len(lidar.point_cloud) % 3:
            raise ValueError("LiDAR point_cloud length is not a multiple of 3")
        observation_ns = int(lidar.time_stamp)
        coefficients = self.fit_motion(self.motion_history[agent], observation_ns)
        if coefficients is None:
            return results
        xyz = np.asarray(lidar.point_cloud, dtype=np.float64).reshape(-1, 3)
        n = len(xyz)
        scan_time = 1.0 / self.lidar_config["rotations_per_second"]
        control_time = self._control_time_from_scan_end(scan_time)
        # Assign times BEFORE any range/target filtering.
        point_times = np.linspace(-scan_time, 0.0, n) if n > 1 else np.zeros(1)
        state_time = (int(state.timestamp) - observation_ns) * 1e-9
        yaw_poly = np.polynomial.polynomial.polyint(coefficients[2])
        polyval = np.polynomial.polynomial.polyval
        scan_yaw = yaw - (polyval(state_time, yaw_poly) - polyval(0.0, yaw_poly))
        control_yaw = scan_yaw + polyval(control_time, yaw_poly) - polyval(0.0, yaw_poly)
        # GT ROI is expressed in the same body frame as the raw scan.
        body_points = xyz + self.lidar_offset
        valid = (np.isfinite(xyz).all(axis=1)
                 & (np.linalg.norm(xyz, axis=1) < self.lidar_config["max_range"])
                 & (xyz[:, 2] > -10.0) & (xyz[:, 2] < 10.0))
        centers = np.array([self._rotate_xy(
            self._relative_position_world(agent, target), -scan_yaw)
            for target in targets])
        distances = np.linalg.norm(body_points[:, None, :] - centers[None, :, :], axis=2)
        # A return can belong to only one target, even when ROI spheres overlap.
        labels = np.argmin(distances, axis=1)
        for target_index, target in enumerate(targets):
            indices = np.flatnonzero(valid & (labels == target_index)
                                     & (distances[:, target_index] <= 4.0))
            if len(indices) < self.roi_min_points:
                continue
            if self.ENABLE_LIDAR_CORRECTION:
                corrected = []
                for index in indices:
                    dx, dy, angle = self.integrate_motion(
                        coefficients, point_times[index], control_time)
                    shifted = body_points[index] - np.array([dx, dy, 0.0])
                    corrected.append(self._rotate_xy(shifted, -angle))
                center = self._rotate_xy(np.mean(corrected, axis=0), control_yaw)
            else:
                # Same selected returns and output axes, without time correction.
                center = self._rotate_xy(np.mean(body_points[indices], axis=0), scan_yaw)
            results[target] = center.astype(np.float32)
            self.target_memory[agent][target] = results[target].copy()
            self.target_detected[agent][target] = True
        return results

    def _get_lidar_measurement(self, agent_name, target_name):
        rel = self._measurement_cache[agent_name][target_name]
        normalized = np.clip(rel[:2] / 100.0, -1.0, 1.0)
        self.lidar_memory[agent_name][target_name] = normalized.tolist()
        return float(normalized[0]), float(normalized[1]), self.target_detected[agent_name][target_name]

    def _get_current_location(self):
        self.current_location = {}
        self.current_location[self.leader_name] = self.client.simGetObjectPose(self.leader_name)
        for agent in self.agents:
            self.current_location[agent] = self.client.simGetObjectPose(agent)
        self.current_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

    # ======================================================================
    # 초기화/이동/시각화 관련
    # ======================================================================
    def _hover(self, name):
        self.client.moveByVelocityZAsync(
            vx=0.0, vy=0.0,
            z=self.fixed_z,
            duration=0.3,
            vehicle_name=name
        ).join()

        try:
            self.client.hoverAsync(vehicle_name=name).join()
        except:
            pass

    def _setup_flight(self):
        self.client.enableApiControl(True, vehicle_name=self.leader_name)
        self.client.armDisarm(True, vehicle_name=self.leader_name)

        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
            self.client.armDisarm(True, vehicle_name=agent)

        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.armDisarm(True, vehicle_name=self.dynamic_name)

        # 1. 이륙 (Takeoff) - 시동 걸기용
        cmds = []
        cmds.append(self.client.takeoffAsync(vehicle_name=self.leader_name))
        for agent in self.possible_agents:
            cmds.append(self.client.takeoffAsync(vehicle_name=agent))
        cmds.append(self.client.takeoffAsync(vehicle_name=self.dynamic_name))

        for c in cmds:
            c.join()

        # 2. [수정] 이동(MoveTo) 대신 텔레포트(SetPose) 사용
        # 물리적 이동 없이 좌표를 강제로 찍어버려서 관성을 0으로 만듭니다.
        
        # (1) Leader 위치 고정
        pose_leader = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose_leader, True, vehicle_name=self.leader_name)
        
        # (2) Followers 위치 고정 (겹치지 않게 약간 분산시켜도 됨, 여기선 0,0으로 모음)
        # 만약 에이전트끼리 겹쳐서 튕겨나간다면 x, y에 약간의 오프셋을 주어야 합니다.
        for i, agent in enumerate(self.possible_agents):
            # 예: 에이전트들을 리더 주변에 약간 띄워서 배치하고 싶다면 아래 주석 해제
            # offset_x = (i % 2) * 2.0 - 1.0  
            # offset_y = (i // 2) * 2.0 - 1.0
            # pose_agent = airsim.Pose(airsim.Vector3r(offset_x, offset_y, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
            
            # 현재는 모두 0,0에 겹쳐도 물리 충돌 무시 설정이 되어있다면 괜찮지만, 보통은 띄우는게 좋습니다.
            # 일단 기존 코드 의도대로 0,0에 둡니다.
            pose_agent = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
            self.client.simSetVehiclePose(pose_agent, True, vehicle_name=agent)

        # (3) 적(DynamicObstacle) 위치 고정 (초기화는 나중에 _reset_obstacle_logic에서 하겠지만 안전상 고정)
        pose_enemy = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose_enemy, True, vehicle_name=self.dynamic_name)

        # 3. [중요] 속도 0으로 강제 초기화 (Momentum Kill)
        # 위치를 옮겨도 이전 속도 벡터가 남아있을 수 있으므로 0으로 덮어씁니다.
        cmds = []
        cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.leader_name))
        for agent in self.possible_agents:
            cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=agent))
        cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name))
        
        for c in cmds:
            c.join()
            
        time.sleep(0.5) # 물리 엔진 안정화 대기

        # 4. 초기 위치 저장
        self.start_location[self.leader_name] = self.client.simGetObjectPose(self.leader_name)
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        self.start_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

        # Hover 함수 호출 (최종 안정화)
        self._hover(self.leader_name)
        for agent in self.possible_agents:
            self._hover(agent)
        self._hover(self.dynamic_name)

    def _update_leader_movement(self):
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

    def _visualize_circles(self):
        try:
            leader_pos = self.client.simGetObjectPose(self.leader_name).position
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

            line_thickness = 20.0
            self.client.simPlotLineStrip(
                ring_points(self.optimal_distance),
                [1, 1, 0, 0.8],
                line_thickness, 0.15, True
            )
            self.client.simPlotLineStrip(
                ring_points(self.far_cutoff),
                [0, 1, 0, 0.8],
                line_thickness, 0.15, True
            )
        except Exception:
            pass

    # ======================================================================
    # 보상/종료 관련
    # ======================================================================
    def _compute_reward(self, agent):
        return 0.0
    def _update_success_curriculum(self, success):
        if not success:
            self.curriculum_success_streak = 0
            return
        self.curriculum_success_count += 1
        self.curriculum_success_streak += 1

        if self.curriculum_success_streak >= self.success_count_for_decay:
            self.curriculum_success_streak = 0
            old_radius = self.success_radius
            self.success_radius = max(
                self.success_radius_min,
                self.success_radius - self.success_radius_decay
            )
            print(
                f"[Curriculum] success_count={self.curriculum_success_count} | "
                f"success_radius: {old_radius:.2f} -> {self.success_radius:.2f}"
            )
    def _end_episode(self, reward, status):
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        # [수정 1] 성공/실패 판정 로직을 TTG와 동일하게 상세화
        is_success = 1.0 if status == "SUCCESS_DISTANCE_AGENT_DYNAMIC" else 0.0
        self._update_success_curriculum(bool(is_success))
        
        # 리더 피격 (적 -> 리더)
        is_leader_hit = 1.0 if "LEADER_AND_DYNAMIC" in status else 0.0
        
        # 아군 충돌 (에이전트끼리 OR 에이전트가 리더와 충돌)
        # PN 기존 로직은 "AGENT_AGENT"만 체크했으나, 리더와의 충돌도 아군 충돌로 보는 것이 맞음
        if "AGENT_AGENT" in status or "AGENT_LEADER" in status:
            is_ally_collision = 1.0 
        else:
            is_ally_collision = 0.0
            
        # 이탈 (Far Cutoff)
        is_obj_collision = 1.0 if "FAR_CUTOFF" in status else 0.0

        self.stats_history["win"].append(is_success)
        self.stats_history["coll_leader"].append(is_leader_hit)
        self.stats_history["coll_drone"].append(is_ally_collision)
        self.stats_history["coll_obj"].append(is_obj_collision)
        

        def get_rate(key):
            if len(self.stats_history[key]) > 0:
                return sum(self.stats_history[key]) / len(self.stats_history[key])
            return 0.0

        win_rate = get_rate("win")

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(reward)
            _terminations_list.append(True)

            # [수정 2] info 구조를 { "AgentName": { ... } } 형태로 중첩 (TTG와 통일)
            # Runner가 이 구조를 인식하여 wandb 로그를 수집합니다.
            _infos_list.append({
            agent: {
                "final_status": status,
                "episode_success": is_success,
                "episode_leader_hit": is_leader_hit,
                "episode_ally_collision": is_ally_collision,
                "win_rate": win_rate,
                "cur_episode_steps": self.step_count,
                "current_hit_radius": self.success_radius,
                "total_successes": self.curriculum_success_count,
                "curriculum_success_streak": self.curriculum_success_streak,
                }
            })
        
        print(f"[{self.episode_count} Ep] WinRate: {win_rate:.2f} | Status: {status}")
        return _obs_list, _rewards_list, _terminations_list, _infos_list
    
    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self):
        self._obs_step_timer += 1
        target_speed = 3.0 

        if self._obstacle_state == "IDLE":
            self.client.moveByVelocityAsync(0, 0, 0, duration=self.dt, vehicle_name=self.dynamic_name)
            if self._obs_step_timer >= self._idle_wait_steps:
                self._obstacle_state = "ATTACK"
                self._obs_step_timer = 0

        elif self._obstacle_state == "ATTACK":
            try:
                leader_pos = self.current_location[self.leader_name].position
                obs_pos = self.current_location[self.dynamic_name].position

                l_vec = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val])
                o_vec = np.array([obs_pos.x_val, obs_pos.y_val, obs_pos.z_val])

                diff = l_vec - o_vec
                dist = np.linalg.norm(diff)

                if dist > 0.5:
                    direction = diff / dist
                    vel = direction * target_speed

                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                        duration=self.dt,
                        vehicle_name=self.dynamic_name
                    )
            except Exception as e:
                print(f"Attack Logic Error: {e}")

    def _teleport_obstacle_randomly(self):
        leader_pos = self.client.simGetVehiclePose(self.leader_name).position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        radius = 40.0
        angle = random.uniform(0, 2 * math.pi)

        tx = lx - 20 + radius * math.cos(angle)
        ty = ly - 20 + radius * math.sin(angle)
        tz = self.fixed_z

        pose = airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.dynamic_name)
        self.client.moveByVelocityAsync(0, 0, 0, duration=self.dt, vehicle_name=self.dynamic_name).join()

    def _reset_obstacle_logic(self):
        self._teleport_obstacle_randomly()
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(2,4)
        print(f"[DynamicObstacle] Reset to IDLE. Waiting for {self._idle_wait_steps} steps.")

    def _check_distance_collision(self, name_a, name_b, threshold):
        pa = self.current_location[name_a].position
        pb = self.current_location[name_b].position

        dx = pa.x_val - pb.x_val
        dy = pa.y_val - pb.y_val

        dist = math.sqrt(dx * dx + dy * dy)
        return dist < threshold, dist

    # ======================================================================
    # RL/PettingZoo API
    # ======================================================================
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

    def _get_obs(self, agent):
        state, yaw, _ = self._record_motion(agent)
        targets = [self.leader_name] + [a for a in self.possible_agents if a != agent] + [self.dynamic_name]
        self._measurement_cache[agent] = self._measure_targets(agent, targets, state, yaw)
        # 1. 내 상태 (속도) - IMU/GPS는 자신의 정보이므로 그대로 사용
        try:
            my_state = state
            vx = my_state.kinematics_estimated.linear_velocity.x_val
            vy = my_state.kinematics_estimated.linear_velocity.y_val
            norm_vx = np.clip(vx / self.MAX_SPEED, -1.0, 1.0)
            norm_vy = np.clip(vy / self.MAX_SPEED, -1.0, 1.0)
        except:
            norm_vx, norm_vy = 0.0, 0.0

        # 2. 리더 위치 (Lidar)
        lx, ly, _ = self._get_lidar_measurement(agent, self.leader_name)
        _leader_feats = [lx, ly]

        # 3. 아군 위치 (Lidar) - 고정 반복문
        _ally_feats = []
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            ox, oy, _ = self._get_lidar_measurement(agent, other)
            _ally_feats.append(ox)
            _ally_feats.append(oy)

        # 4. 적군 상대 위치만 관측 (접근 속도 및 LOS 변화율 제외)
        ex, ey, _ = self._get_lidar_measurement(agent, self.dynamic_name)
        _dynamic_feats = [ex, ey]

        # 5. 최종 결합
        obs = np.concatenate([
            np.array(_leader_feats, dtype=np.float32).flatten(),
            np.array(_ally_feats, dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
            np.array([norm_vx, norm_vy], dtype=np.float32) # 내 속도
        ]).astype(np.float32)

        return obs
    

    def _do_action(self, actions):
        actions = np.clip(actions, -1, 1)
        dt = self.dt
        commands = []

        for i, agent in enumerate(self.possible_agents):
            a = actions[i]
            vx = float(a[0]) * self.MAX_SPEED
            vy = float(a[1]) * self.MAX_SPEED

            speed = math.hypot(vx, vy)
            if speed > self.MAX_SPEED:
                scale = self.MAX_SPEED / (speed + 1e-6)
                vx *= scale
                vy *= scale

            commands.append(
                self.client.moveByVelocityZAsync(
                    vx=vx,
                    vy=vy,
                    z=self.fixed_z,
                    duration=dt,
                    yaw_mode=airsim.YawMode(
                        is_rate=False, yaw_or_rate=0.0
                    ),
                    vehicle_name=agent,
                )
            )

        for command in commands:
            command.join()

    def _get_rewards(self, per_agent_results):
        return [np.mean(per_agent_results) for _ in self.possible_agents]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.episode_count += 1
        print(f"에피소드: {self.episode_count} | 소비한 스텝 수: {self.step_count}")

        self.step_count = 0
        self.agents = self.possible_agents[:]

        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        self._reset_obstacle_logic()
        self._get_current_location()

        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._last_pose.clear()
        self._last_time.clear()

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

        # 메모리 및 이전 위치 초기화
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        
        for agent in self.possible_agents:
            self.lidar_memory[agent] = {}
            self.prev_lidar_pos[agent] = {}
            
            # 초기엔 타겟 위치 모름 (100m 끝지점으로 가정)
            # 여기서는 정규화 전인 '실제 미터' 단위로 저장하는 것이 계산에 편합니다.
            default_real_pos = [100.0, 100.0] 
            
            self.lidar_memory[agent][self.leader_name] = [1.0, 1.0] # Obs용 정규화 값
            self.prev_lidar_pos[agent][self.leader_name] = default_real_pos # 속도계산용 실제 값

            self.lidar_memory[agent][self.dynamic_name] = [1.0, 1.0]
            self.prev_lidar_pos[agent][self.dynamic_name] = default_real_pos
            
            # 아군들
            for other in self.possible_agents:
                if agent != other:
                    self.lidar_memory[agent][other] = [1.0, 1.0]
                    self.prev_lidar_pos[agent][other] = default_real_pos

        self.motion_history = {a: deque(maxlen=256) for a in self.possible_agents}
        self.target_memory = {a: {} for a in self.possible_agents}
        self.target_detected = {a: {} for a in self.possible_agents}
        self._measurement_cache = {}
        # Identical initial history collection in both variants.
        for _ in range(self.motion_history_samples + 2):
            for agent in self.possible_agents:
                self._record_motion(agent)
            time.sleep(0.02)

        obs_list = [self._get_obs(a) for a in self.agents]
        return obs_list

    def step(self, actions):
        self.step_count += 1
        
        per_agent_obs, per_agent_results, per_agent_infos = [], [], []

        self._do_action(actions)
        self._update_leader_movement()
        self._update_dynamic_obstacle()
        self._get_current_location() # 충돌 판정용

        for agent in self.possible_agents:
            # 이탈 체크
            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location[self.leader_name].position.x_val,
                self.current_location[agent].position.y_val - self.current_location[self.leader_name].position.y_val,
                self.current_location[agent].position.z_val - self.current_location[self.leader_name].position.z_val
            ])

            if _distance_leader > self.far_cutoff:
                print(f"[이탈] {agent} 리더 거리 초과! → 전체 실패")
                return self._end_episode(self.REWARD_FAR_CUTOFF, "FAIL_AGENT_FAR_CUTOFF")

            # 리더와 충돌
            hit, d = self._check_distance_collision(agent, self.leader_name, threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"⚠️💔[충돌] {agent} ↔ Leader")
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_LEADER")

            # 아군끼리 충돌
            other_agents = [a for a in self.possible_agents if a != agent]
            for other in other_agents:
                hit, d = self._check_distance_collision(agent, other, threshold=self.COLLISION_THRESHOLD)
                if hit:
                    print(f"💥🤖[충돌] {agent} ↔ {other}")
                    return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_AGENT")

            # 리더 피격 (Enemy -> Leader)
            collisionInfo = self.client.simGetCollisionInfo(self.leader_name)
            if collisionInfo.has_collided and collisionInfo.object_name == self.dynamic_name:
                print(f"💥[피격] 리더가 적에게 피격됨!")
                return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")
            # 리더 피격 (Enemy -> Leader)
            hit, d = self._check_distance_collision(
                self.leader_name,
                self.dynamic_name,
                threshold=2.0
            )
            if hit:
                print(f"💥[피격] 리더가 적에게 피격됨! | distance={d:.2f}m")
                return self._end_episode(
                    self.REWARD_LEADER_HIT,
                    "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION"
                )
            
            # 요격 성공 (Agent -> Enemy)
            hit, d = self._check_distance_collision(
                agent,
                self.dynamic_name,
                threshold=self.success_radius
            )
            if hit:
                print(f"🎯🔥[요격] {agent}가 적을 요격 성공! | success_radius={self.success_radius:.2f}m")
                return self._end_episode(self.REWARD_HIT_ENEMY, "SUCCESS_DISTANCE_AGENT_DYNAMIC")

           # --- [관측 및 보상] ---
            
            # 1. 관측 (여기서 self.lidar_memory가 갱신됨!)
            obs = self._get_obs(agent)
            per_agent_obs.append(obs)

            # 2. 보상 (갱신된 메모리와 이전 위치를 비교하여 계산)
            _reward = self._compute_reward(agent) 
            per_agent_results.append(_reward)
            per_agent_infos.append({"step_reward": _reward})
            
            # 3. [중요] 다음 스텝을 위해 현재 Lidar 위치를 '이전 위치'로 저장
            #    _get_obs에서 갱신된 lidar_memory(정규화됨)를 다시 미터로 바꿔 저장
            norm_curr = self.lidar_memory[agent][self.dynamic_name]
            real_curr = [norm_curr[0] * 100.0, norm_curr[1] * 100.0]
            self.prev_lidar_pos[agent][self.dynamic_name] = real_curr

        termination_list = [False for _ in self.possible_agents]
        rewards_list = self._get_rewards(per_agent_results)
        obs_list = per_agent_obs
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list