"""Static Cylinder_2 experiment: one circle per trial with tangent yaw.

Before running, manually place/spawn Cylinder_2 in the simulator. This script
never spawns, moves, scales or destroys it. The object must survive reset().
GT = (Cylinder_2 world position - control-time UAV world position), rotated
into the control-time UAV yaw frame. Raw and Ours centroids use this same GT.
As in the original experiment, all returned LiDAR points form the centroid:
configure the scene/sensor so those points represent Cylinder_2. Center-based
GT can include a surface-centroid offset for a finite-radius cylinder.
Assumes LiDAR horizontal axes/origin match the UAV; planar yaw compensation.
Run: python make_circle_static.py
"""
import csv
import math
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import airsim
import numpy as np

TRIAL_COUNT = 100
CIRCLE_DIAMETER_M = 100.0
PATH_ACCELERATION_MPS2 = 2.0  # target-speed startup/stop ramp
ARRIVAL_RADIUS_M = 1.0
ARRIVAL_SPEED_MPS = 0.5
PLOT_EVERY_TRIALS = 10  # refresh graphs every 10 trials and at exit
SPEED = 5.0  # requested cruise speed; change to 10 or 15 for another run
ALTITUDE = -20.0  # o_v.py
LATENCY = 0.2  # retain current experiment setting
SCAN_TIME = 0.1
HISTORY_SAMPLES = 10  # N: 관측 이전 운동 샘플 수
POLY_DEGREE = 2       # m: 다항식 차수
HISTORY_INTERVAL = 0.02  # 초기 이력 수집/이력 부족 시 대기 간격(초)
VEHICLE_NAME = 'Follower0'
LIDAR_NAME = 'Follower0_LidarSensor'
OBJECT_NAME = 'Cylinder_2'  # Simulator object name; manually place/spawn before running
MAX_YAW_RATE_DEG_S = 60.0  # feedback command ceiling, NOT a constant yaw-rate target
CONTROL_DT = 0.05
OUTPUT_ROOT = Path(__file__).resolve().parent / 'experiment_results' / 'make_circle_static'

ERROR_FIELDS = ['raw_x_mae_m', 'raw_y_mae_m', 'corrected_x_mae_m', 'corrected_y_mae_m',
                'raw_bearing_mae_deg', 'ours_bearing_mae_deg']


class CircleEpisode:
    def __init__(self, origin, diameter, speed, yaw_limit=60., acceleration=2.):
        if not all(math.isfinite(v) and v > 0 for v in (diameter, speed, yaw_limit, acceleration)):
            raise ValueError('Circle dimensions and motion limits must be finite and positive')
        self.origin = np.asarray(origin, dtype=float)
        self.radius = diameter/2
        self.speed = self.cruise = speed
        self.yaw_limit = math.radians(yaw_limit)
        if speed/self.radius > .85*self.yaw_limit:
            raise ValueError('Requested speed needs more yaw rate: enlarge diameter or raise yaw limit')
        self.acceleration = acceleration
        self.command_acceleration = max(acceleration*2, 1.5*speed**2/self.radius)
        self.length = 2*math.pi*self.radius
        self.distance = self.reference_speed = self.yaw_command = 0.
        self.command = np.zeros(2)
        self.velocity_integral = np.zeros(2)
        self.phase = 'accelerating'
        self.reference_yaw_rate_deg_s = 0.
        self.reference_progress = 0.

    @property
    def progress(self):
        return min(1., self.distance/self.length)

    def geometry(self, q):
        point = self.origin + [self.radius*(1-math.cos(q)), -self.radius*math.sin(q)]
        tangent = np.array([math.sin(q), -math.cos(q)])
        return point, tangent, 1/self.radius

    def update(self, position, velocity, yaw, dt, arrival_radius=1., arrival_speed=.5):
        dt = max(0., min(float(dt), .1))
        reference, tangent, curvature = self.geometry(self.distance/self.radius)
        remaining = max(0., self.length-self.distance)
        at_end = remaining < .03
        target = min(self.speed, math.sqrt(2*self.acceleration*remaining))
        self.reference_speed += float(np.clip(target-self.reference_speed,
                                               -self.acceleration*dt, self.acceleration*dt))
        self.phase = ('cruise' if abs(self.reference_speed-self.speed) < .01 else
                      'braking' if remaining < self.speed**2/(2*self.acceleration) else 'accelerating')
        if at_end:
            self.distance = self.length
            reference, tangent, curvature = self.geometry(2*math.pi)
            self.reference_speed = 0.
            self.phase = 'settling'
        self.reference_progress = self.progress
        error = reference-position
        if np.linalg.norm(error) > 15.:
            raise RuntimeError('Circle tracking error exceeds 15 m; trial cannot represent the requested path')
        desired_velocity = tangent*self.reference_speed
        velocity_error = desired_velocity-velocity
        self.velocity_integral = np.clip(self.velocity_integral+velocity_error*dt, -3., 3.)
        requested = desired_velocity + .6*error + .6*velocity_error + .15*self.velocity_integral
        if at_end:
            self.velocity_integral[:] = 0.
            requested = .6*error-.4*velocity
        # Headroom corrects tracking lag instead of silently lowering SPEED.
        norm = np.linalg.norm(requested)
        limit = 1.5*self.speed
        if norm > limit:
            requested *= limit/norm
        delta = requested-self.command
        change = np.linalg.norm(delta)
        if change > self.command_acceleration*dt:
            delta *= self.command_acceleration*dt/change
        self.command += delta
        heading = math.atan2(tangent[1], tangent[0])
        heading_error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
        self.reference_yaw_rate_deg_s = math.degrees(curvature*self.reference_speed)
        target_rate = float(np.clip(curvature*self.reference_speed+1.5*heading_error,
                                    -self.yaw_limit, self.yaw_limit))
        self.yaw_command += float(np.clip(target_rate-self.yaw_command,
                                          -self.yaw_limit*dt, self.yaw_limit*dt))
        done = (at_end and np.linalg.norm(position-self.origin) <= arrival_radius
                and np.linalg.norm(velocity) <= arrival_speed)
        if not at_end:
            self.distance = min(self.length, self.distance+self.reference_speed*dt)
        return self.command.copy(), math.degrees(self.yaw_command), reference, done


def read_motion(output):
    output = Path(output)
    statuses = {}
    trial_file = output / 'trials.csv'
    if trial_file.exists():
        with trial_file.open(encoding='utf-8-sig', newline='') as f:
            statuses = {int(r['trial']): r['status'] for r in csv.DictReader(f)}
    records = []
    for path in sorted(output.glob('trajectory_trial_*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        # Legacy figure-eight telemetry lacks some of these measurements.
        if 'actual_yaw_rate_deg_s' not in rows[0]:
            continue
        trial = int(path.stem.rsplit('_', 1)[1])
        fields = ['elapsed_sim_s', 'speed_mps', 'target_speed_mps', 'command_speed_mps',
                  'actual_yaw_rate_deg_s', 'target_yaw_rate_deg_s', 'yaw_rate_command_deg_s',
                  'tracking_error_m']
        data = {k: np.array([float(r[k]) for r in rows]) for k in fields}
        data.update(trial=trial, status=statuses.get(trial, 'partial'),
                    cruise=np.array([r['phase'] == 'cruise' for r in rows]))
        records.append(data)
    return records


def motion_statistics(records):
    results = []
    for r in records:
        result = {'trial': r['trial'], 'status': r['status'],
                  'samples': len(r['speed_mps']), 'cruise_samples': int(r['cruise'].sum())}
        for label, actual, target in [
                ('speed_mps', 'speed_mps', 'target_speed_mps'),
                ('yaw_rate_deg_s', 'actual_yaw_rate_deg_s', 'target_yaw_rate_deg_s')]:
            mask = r['cruise'] & np.isfinite(r[actual]) & np.isfinite(r[target])
            values = r[actual][mask]
            result['cruise_'+label+'_mean'] = float(values.mean()) if values.size else None
            result['cruise_'+label+'_std'] = float(values.std()) if values.size else None
            result['cruise_'+label+'_target'] = float(r[target][mask].mean()) if values.size else None
            result['cruise_'+label+'_mae'] = float(np.abs(values-r[target][mask]).mean()) if values.size else None
        result['max_tracking_error_m'] = float(np.nanmax(r['tracking_error_m']))
        results.append(result)
    return results


def plot_motion_history(output):
    output = Path(output)
    records = read_motion(output)
    if not records:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stats = motion_statistics(records)
    with (output/'motion_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    complete = sum(r['status'] == 'completed' for r in records)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    offset = 0.
    for i, r in enumerate(records):
        t = r['elapsed_sim_s']-r['elapsed_sim_s'][0]+offset
        for ax, actual, target, command in [
                (axes[0], 'speed_mps', 'target_speed_mps', 'command_speed_mps'),
                (axes[1], 'actual_yaw_rate_deg_s', 'target_yaw_rate_deg_s', 'yaw_rate_command_deg_s')]:
            ax.plot(t, r[actual], color='#2563eb', lw=.8, label='Measured' if i == 0 else None)
            ax.plot(t, r[target], color='#f97316', ls='--', lw=1, label='Reference' if i == 0 else None)
            ax.plot(t, r[command], color='#64748b', alpha=.5, lw=.6, label='Command' if i == 0 else None)
            ax.axvline(offset, color='gray', alpha=.15, lw=.5)
        offset = t[-1]+1.  # visual separator; reset/takeoff times are not concatenated
    axes[0].set_ylabel('Horizontal speed (m/s)')
    axes[1].set_ylabel('Yaw rate (deg/s)')
    axes[1].set_xlabel('Concatenated flight time (s); 1 s separators between trials')
    axes[0].set_title(f'Measured motion: {len(records)} recorded trials, {complete} completed')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc='upper right')
    fig.tight_layout()
    trace = output/'motion_history.png'
    fig.savefig(trace, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # Keep incomplete/error trials out of the completed-trial comparison.
    valid = [r for r in stats if r['status'] == 'completed' and r['cruise_samples'] > 0]
    for ax, label, unit in [(axes[0], 'speed_mps', 'm/s'), (axes[1], 'yaw_rate_deg_s', 'deg/s')]:
        ids = [r['trial'] for r in valid]
        means = np.array([r['cruise_'+label+'_mean'] for r in valid], dtype=float)
        stds = np.array([r['cruise_'+label+'_std'] for r in valid], dtype=float)
        ax.plot(ids, means, 'o-', ms=3, label='Cruise mean (measured)')
        ax.fill_between(ids, means-stds, means+stds, alpha=.2, label='Within-trial ±1 SD')
        ax.plot(ids, [r['cruise_'+label+'_target'] for r in valid], '--', label='Cruise reference')
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
        ax.legend(loc='best')
    axes[0].set_title('Completed trials: cruise only (startup / braking / settling excluded)')
    axes[1].set_xlabel('Trial number')
    fig.tight_layout()
    comparison = output/'motion_by_trial.png'
    fig.savefig(comparison, dpi=150)
    plt.close(fig)
    return [trace, comparison, output/'motion_summary.csv']




class LidarTest:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.obj_name = OBJECT_NAME
        self.stop_event = threading.Event()
        self.control_thread = None
        self.trial_origin = None
        self.control_error = None
        self.episode_done = threading.Event()

    def start_trial(self, vehicle_name):
        # 매 회차 동일 시작 상태로 초기화한 뒤 기존 코드와 같은 순서로 비행한다.
        self.stop_circle()
        self.client.reset()
        self.vehicle_name = vehicle_name
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        self.client.takeoffAsync(vehicle_name=vehicle_name).join()
        self.client.moveToZAsync(ALTITUDE, 1.0, vehicle_name=vehicle_name).join()
        pose = self.client.simGetVehiclePose(vehicle_name=vehicle_name)
        if self.trial_origin is None:
            self.trial_origin = (pose.position.x_val, pose.position.y_val)
        self.path = CircleEpisode(
            self.trial_origin, CIRCLE_DIAMETER_M,
            SPEED, MAX_YAW_RATE_DEG_S, PATH_ACCELERATION_MPS2)
        self.episode_done.clear()
        self.validate_static_object()
        # Align to the initial tangent while stationary, before measurement.
        self.client.rotateToYawAsync(-90., vehicle_name=vehicle_name).join()
        self.client.hoverAsync(vehicle_name=vehicle_name).join()
        # Reuse the first trial's origin, rather than takeoff drift on each reset.
        self.client.simSetVehiclePose(airsim.Pose(
            airsim.Vector3r(*self.trial_origin, ALTITUDE),
            airsim.to_quaternion(0., 0., -math.pi/2)), True, vehicle_name=vehicle_name)
        print(f'Circle diameter={CIRCLE_DIAMETER_M:.1f} m, target speed={SPEED:.2f} m/s, '
              f'nominal yaw rate={math.degrees(SPEED/self.path.radius):.2f} deg/s', flush=True)
        self.move(vehicle_name)
        self.check_control()

    def validate_static_object(self):
        """Read the user-owned object without spawning or changing it."""
        pose = self.client.simGetObjectPose(self.obj_name)
        xyz = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        if not np.isfinite(xyz).all():
            raise RuntimeError(
                f'Static object {self.obj_name!r} was not found or has an invalid pose. '
                'Place/spawn it in the simulator before running; verify its AirSim object name. '
                'It must also remain available after client.reset().')
        print(f'Static object: {self.obj_name}, world NED position={xyz}', flush=True)
        return pose

    def move(self, vehicle_name):
        self.vehicle_name = vehicle_name
        self.stop_event.clear()
        self.control_error = None
        self.control_thread = threading.Thread(target=self.circle_control_loop, daemon=True)
        self.control_thread.start()

    def circle_control_loop(self):
        client = None
        try:
            client = airsim.MultirotorClient(timeout_value=3)
            client.confirmConnection()
            client.enableApiControl(True, vehicle_name=self.vehicle_name)
            previous_ns = first_ns = previous_yaw = None
            previous_angle = None
            actual_swept_angle = 0.
            with (self.output / f'trajectory_trial_{self.trial:03d}.csv').open(
                    'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_ns', 'progress', 'reference_x', 'reference_y',
                                 'actual_x', 'actual_y', 'actual_z', 'speed_mps',
                                 'yaw_rate_command_deg_s',
                                 'elapsed_sim_s', 'target_speed_mps', 'command_speed_mps',
                                 'actual_yaw_rate_deg_s', 'body_wz_deg_s',
                                 'target_yaw_rate_deg_s', 'phase', 'tracking_error_m',
                                 'actual_circle_progress', 'arrival_distance_m', 'episode_done'])
                while not self.stop_event.is_set():
                    state = client.getMultirotorState(vehicle_name=self.vehicle_name)
                    kin = state.kinematics_estimated
                    dt = 0. if previous_ns is None else (state.timestamp-previous_ns)*1e-9
                    if dt < 0:
                        raise RuntimeError('Simulator timestamp moved backwards')
                    position = np.array([kin.position.x_val, kin.position.y_val])
                    velocity = np.array([kin.linear_velocity.x_val, kin.linear_velocity.y_val])
                    _, _, yaw = airsim.to_eularian_angles(kin.orientation)
                    if first_ns is None:
                        first_ns = state.timestamp
                    actual_yaw_rate = (math.degrees(math.atan2(math.sin(yaw-previous_yaw),
                                       math.cos(yaw-previous_yaw)))/dt
                                       if previous_yaw is not None and dt > 0 else float('nan'))
                    previous_ns, previous_yaw = state.timestamp, yaw
                    elapsed_sim_s = (state.timestamp-first_ns)*1e-9
                    command, rate, reference, done = self.path.update(
                        position, velocity, yaw, dt, ARRIVAL_RADIUS_M, ARRIVAL_SPEED_MPS)
                    # update() advances distance for the NEXT step; log progress
                    # corresponding to the reference returned for THIS step.
                    progress = self.path.reference_progress
                    angle = math.atan2(-(position[1]-self.path.origin[1]),
                                       self.path.radius-(position[0]-self.path.origin[0]))
                    if previous_angle is not None:
                        actual_swept_angle += math.atan2(math.sin(angle-previous_angle),
                                                        math.cos(angle-previous_angle))
                    previous_angle = angle
                    writer.writerow([state.timestamp, progress, *reference, *position,
                                     kin.position.z_val, np.linalg.norm(velocity), rate,
                                     elapsed_sim_s, self.path.reference_speed,
                                     np.linalg.norm(command), actual_yaw_rate,
                                     math.degrees(kin.angular_velocity.z_val),
                                     self.path.reference_yaw_rate_deg_s,
                                     self.path.phase, np.linalg.norm(reference-position),
                                     actual_swept_angle/(2*math.pi),
                                     np.linalg.norm(position-self.path.origin), done])
                    f.flush()
                    if done:
                        client.moveByVelocityZAsync(0, 0, ALTITUDE, duration=1.,
                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                            vehicle_name=self.vehicle_name)
                        self.episode_done.set()
                        self.stop_event.set()
                        return
                    client.moveByVelocityZAsync(float(command[0]), float(command[1]), ALTITUDE,
                        duration=max(.15, 3*CONTROL_DT),
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rate),
                        vehicle_name=self.vehicle_name)
                    self.stop_event.wait(CONTROL_DT)
        except Exception as exc:
            self.control_error = exc
            self.stop_event.set()
            if client is not None:
                try:
                    client.hoverAsync(vehicle_name=self.vehicle_name)
                except Exception:
                    pass


    def check_control(self):
        if self.control_error is not None:
            raise RuntimeError('Circle control failed') from self.control_error

    def stop_circle(self):
        self.stop_event.set()
        for name in ('control_thread',):
            worker = getattr(self, name, None)
            if worker is not None:
                worker.join(timeout=10.0)
                if worker.is_alive():
                    raise RuntimeError(f'{name} did not stop')
                setattr(self, name, None)

    # 쿼터니언 곱셈 함수 
    def quaternion_multiply(self,q1, q2): 
        w1, x1, y1, z1 = q1 
        w2, x2, y2, z2 = q2 

        return np.array([ 
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, 
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, 
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, 
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2 
        ]) 

    def rotate(self,corxyz,radian): 
        axis = np.array([0.0,0.0,1.0]) 
        axis = axis/np.linalg.norm(axis) #축 정규화 

        '''회전 쿼터니언 계산''' 
        seta = radian/2  
        w = np.cos(seta) 
        qx,qy,qz = axis*np.sin(seta) 
        q = np.array([w,qx,qy,qz]) 

        #순수 쿼터니언(벡터 부분만 남기기) 
        qv = np.array([0]+list(corxyz)) 

        #회전 : q*qv*qconjugate 
        qconjugate = np.array([w,-qx,-qy,-qz]) 
        rotated = self.quaternion_multiply(self.quaternion_multiply(q, qv), qconjugate) 


        return rotated[1:] 
    
    @staticmethod
    def bearing_metrics(raw_x, raw_y, corrected_x, corrected_y, true_x, true_y):
        """make__rotate-style wrapped errors; all inputs already in UAV yaw frame."""
        pairs = ((raw_x, raw_y), (corrected_x, corrected_y), (true_x, true_y))
        if any(not np.isfinite(pair).all() or math.hypot(*pair) < 1e-9 for pair in pairs):
            raise ValueError('Relative bearing undefined: zero-length or nonfinite position')
        raw, ours, gt = (math.atan2(y, x) for x, y in pairs)
        # GT has already been rotated into the control-time body frame. Do NOT
        # subtract control_yaw a second time as the world-GT rotate script does.
        raw_error = abs(math.degrees(math.atan2(math.sin(raw-gt), math.cos(raw-gt))))
        ours_error = abs(math.degrees(math.atan2(math.sin(ours-gt), math.cos(ours-gt))))
        return dict(raw_bearing_deg=math.degrees(raw), ours_bearing_deg=math.degrees(ours),
                    gt_bearing_deg=math.degrees(gt), raw_error_deg=raw_error,
                    ours_error_deg=ours_error)

    def fit_motion(self, history, observation_ns):
        # LiDAR 관측 timestamp보다 이전인 샘플만 선택한다(미래 데이터 제외).
        samples = [sample for sample in history if sample[0] < observation_ns][-HISTORY_SAMPLES:]
        if len(samples) < HISTORY_SAMPLES:
            return None
        # 큰 절대 timestamp 대신 스캔 관측 시점을 0초로 둔다.
        times = np.array([(sample[0] - observation_ns) * 1e-9 for sample in samples])
        values = np.array([sample[1:] for sample in samples], dtype=float)
        if np.any(np.diff(times) <= 0) or not np.isfinite(values).all():
            raise ValueError('Motion samples must have increasing timestamps and finite velocities')
        scale = max(float(np.max(np.abs(times))), 1e-9)
        design = np.polynomial.polynomial.polyvander(times / scale, POLY_DEGREE)
        coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
        if rank != POLY_DEGREE + 1:
            raise ValueError('Motion regression has insufficient independent timestamps')
        # 행: vx, vy, wz / 열: 상수항, 1차항, 2차항 (시간 단위는 초).
        return coefficients.T / scale ** np.arange(POLY_DEGREE + 1)

    @staticmethod
    def world_vector_to_body(x_world, y_world, yaw):
        """Express a world displacement in the yaw frame used by compensation."""
        c, s = math.cos(yaw), math.sin(yaw)
        return c*x_world+s*y_world, -s*x_world+c*y_world

    # 식 (7)~(9): 회귀한 속도와 각속도를 각 점의 ti부터 tc까지 적분한다.
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
        

    def run_trial(self, trial, vehicle_name, lidar_name, scan_writer, scan_file, result):
        count = 0
        empty_scans = 0
        sums = np.zeros(len(ERROR_FIELDS))
        started = None
        # 회차마다 새 이력. 최신 상태가 LiDAR보다 늦을 수 있어 N보다 여유 있게 보관한다.
        motion_history = deque(maxlen=HISTORY_SAMPLES * 2)
        result.update(trial=trial, status='running', error='')
        try:
            print(f"Trial {trial}/{TRIAL_COUNT}: reset / takeoff", flush=True)
            self.trial = trial
            self.start_trial(vehicle_name)
            started = time.monotonic()
            print(f"Trial {trial}/{TRIAL_COUNT}: one full circle, returning to start", flush=True)
            while not self.episode_done.is_set():
                self.check_control()
                latency = LATENCY # 관측->제어명령 적용까지 걸리는 지연시간 합치면 0.2초로 추정 
                scantime = SCAN_TIME # 라이다 스캔 과정에서 걸리는 시간 
                state = self.client.getMultirotorState(vehicle_name=vehicle_name) 
                vx_world = state.kinematics_estimated.linear_velocity.x_val
                vy_world = state.kinematics_estimated.linear_velocity.y_val
                # 2차원 운동 가정: NED 선속도를 yaw 기준 기체 좌표계로 변환한다.
                _, _, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
                vx = vx_world * np.cos(yaw) + vy_world * np.sin(yaw)
                vy = -vx_world * np.sin(yaw) + vy_world * np.cos(yaw)
                wz = state.kinematics_estimated.angular_velocity.z_val 
                state_ns = int(state.timestamp)
                if motion_history and state_ns < motion_history[-1][0]:
                    raise ValueError('Motion timestamp moved backwards')
                if not motion_history or state_ns > motion_history[-1][0]:
                    motion_history.append((state_ns, vx, vy, wz))
                ''' 
                N=i 번째 스캔 받기 
                ''' 
                lidar_data = self.client.getLidarData(lidar_name=lidar_name,vehicle_name=vehicle_name) 
                ''' 
                보정식 시작 
                ''' 
                if not lidar_data.point_cloud:
                    empty_scans += 1
                    print(f"Trial {trial}: empty LiDAR scan (vx={vx:.3f}, vy={vy:.3f})", flush=True)
                    time.sleep(latency)
                    continue
                if len(lidar_data.point_cloud) % 3:
                    raise ValueError("LiDAR point_cloud length is not a multiple of 3")
                observation_ns = int(lidar_data.time_stamp)
                coefficients = self.fit_motion(motion_history, observation_ns)
                if coefficients is None:
                    available = sum(sample[0] < observation_ns for sample in motion_history)
                    print(f"Trial {trial}: collecting past motion samples "
                          f"{min(available, HISTORY_SAMPLES)}/{HISTORY_SAMPLES}", flush=True)
                    time.sleep(HISTORY_INTERVAL)
                    continue
                k = 1 # K 번째 점을 의미 
                tempx=0 
                tempy=0 
                tempz=0 
                nottempx =0 
                nottempy =0 
                nottempz =0 
                size = 0 
                for i in range(0, len(lidar_data.point_cloud), 3): 
                    size = len(lidar_data.point_cloud)/3 #N 번째 스캔에서 점의 개수 
                    inter = scantime / (size - 1) if size > 1 else 0.0
                    # 스캔 종료 후 latency만큼 대기: 첫 점 0.3초, 마지막 점 0.2초.
                    # 한 점만 있으면 스캔 종료 시점에 측정된 것으로 간주한다.
                    t_correct = latency + scantime - (k-1)*inter if size > 1 else latency
                    k = k+1 # 다음 점의 순서를 가리키는 인덱스  
                    xyz = lidar_data.point_cloud[i:i+3] 

                    #이동 거리 추정 
                    # 식 (7)~(11)을 이용한 이동량 및 회전각 계산
                    px, py, radian = self.integrate_motion(
                        coefficients,
                        latency - t_correct,  # ti: 스캔 종료 기준 -0.1~0초
                        latency               # tc: 같은 시간 원점 기준 +0.2초
                    )

                    # 점의 이동 보정
                    corx = xyz[0] - px
                    cory = xyz[1] - py

                    # 쿼터니언 행렬 변환 
                    corxyz = np.array([corx,cory,xyz[2]]) 
                    # 측정 시점 기체 상대 좌표를 제어 시점으로 변환: R(-delta_yaw)(p-d).
                    tempxyz= self.rotate(corxyz,-radian) #최종 점의 보정 좌표
                    tempx = tempx + tempxyz[0] 
                    tempy = tempy + tempxyz[1] 
                    tempz = tempz + tempxyz[2] 
                    #보정안한거 
                    nottempx = nottempx + xyz[0] 
                    nottempy = nottempy + xyz[1] 
                    nottempz = nottempz + xyz[2] 
                #최종 보정된 점 
                finx = tempx / size  
                finy = tempy / size  
                finz = tempz / size 
                print("finx : %f \n" %finx) 
                print("finy : %f \n" %finy) 

                notfinx = nottempx / size  
                notfiny = nottempy / size  
                notfinz = nottempz / size 

                
                # 스캔 + 의사결정 지연 시간동안 드론이 움직임 
                time.sleep(latency) 

                pose = self.client.simGetVehiclePose(vehicle_name=vehicle_name) 
                object_pose = self.validate_static_object() 
                world_relx = object_pose.position.x_val - pose.position.x_val
                world_rely = object_pose.position.y_val - pose.position.y_val
                _, _, control_yaw = airsim.to_eularian_angles(pose.orientation)
                relx, rely = self.world_vector_to_body(world_relx, world_rely, control_yaw)
                
                #print("==========================================================================\n") 
                print("relx : %f \n" %(relx)) 
                print("rely : %f \n" %(rely)) 
                #print("drone pose : x = %f y = %f z = %f \n" %(pose.position.x_val,pose.position.y_val,pose.position.z_val)) 
                #print("object pose : x = %f y = %f z = %f \n" %(object_pose.position.x_val,object_pose.position.y_val,object_pose.position.z_val)) 
                errx = finx-relx 
                erry = finy-rely 
                print(" N번째 보정점의 오차 결과 : | x = %f | \n" %errx) 
                print(" N번째 보정점의 오차 결과 : | y = %f | \n" %erry) 

                noterrx = notfinx-relx 
                noterry = notfiny-rely 
                print(" N번째 보정안한점의 오차 결과 : | x = %f | \n" %noterrx) 
                print(" N번째 보정안한점의 오차 결과 : | y = %f | \n" %noterry) 
                # 기존 계산 결과를 그대로 기록한다. 평균에는 절대 오차를 사용한다.
                values = [finx, finy, notfinx, notfiny, relx, rely, errx, erry, noterrx, noterry]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite position/error; check LiDAR data and Cylinder_2 pose")
                bearings = self.bearing_metrics(notfinx, notfiny, finx, finy, relx, rely)
                count += 1
                elapsed = time.monotonic() - started
                sums += [abs(noterrx), abs(noterry), abs(errx), abs(erry),
                         bearings['raw_error_deg'], bearings['ours_error_deg']]
                scan_writer.writerow({
                    'trial': trial, 'scan': count, 'elapsed_s': elapsed,
                    'lidar_timestamp': lidar_data.time_stamp, 'points': int(size),
                    'vx': vx, 'vy': vy, 'wz': wz,
                    'raw_x': notfinx, 'raw_y': notfiny,
                    'corrected_x': finx, 'corrected_y': finy,
                    'true_x': relx, 'true_y': rely,
                    'true_world_x': world_relx, 'true_world_y': world_rely,
                    'control_yaw_deg': math.degrees(control_yaw),
                    'raw_error_x': noterrx, 'raw_error_y': noterry,
                    'corrected_error_x': errx, 'corrected_error_y': erry, **bearings})
                scan_file.flush()
                print(f"Trial {trial}/{TRIAL_COUNT}, scan {count}, {elapsed:.1f}s, "
                      f"vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}", flush=True)
            self.check_control()
            result['status'] = 'completed' if count else 'no_data'
        except KeyboardInterrupt:
            result.update(status='interrupted', error='KeyboardInterrupt')
            raise  # 원래 traceback을 유지하고 반복 실험을 중단한다.
        except Exception as exc:
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise  # 오류를 한 줄 메시지로 숨기거나 다음 회차로 넘어가지 않는다.
        finally:
            try:
                self.stop_circle()
            except Exception:
                traceback.print_exc()
            result.update(scans=count, empty_scans=empty_scans,
                          elapsed_s=time.monotonic()-started if started is not None else 0)
            result.update(dict(zip(ERROR_FIELDS, (sums/count).tolist() if count else [None]*len(ERROR_FIELDS))))

    @staticmethod
    def save_motion_plots(output):
        try:
            plot_motion_history(output)
        except Exception as exc:
            print(f'Motion plot not generated: {exc}; trajectory CSV files are retained', flush=True)

    def execute(self, vehicle_name=VEHICLE_NAME, lidar_name=LIDAR_NAME):
        if not np.isfinite([SPEED, MAX_YAW_RATE_DEG_S]).all() or min(SPEED, MAX_YAW_RATE_DEG_S) <= 0:
            raise ValueError('SPEED and MAX_YAW_RATE_DEG_S must be positive and finite')
        if TRIAL_COUNT < 1:
            raise ValueError('Trial count must be positive')
        planned = CircleEpisode((0, 0), CIRCLE_DIAMETER_M, SPEED,
                                MAX_YAW_RATE_DEG_S, PATH_ACCELERATION_MPS2)
        if not all(math.isfinite(v) and v > 0 for v in
                   (ARRIVAL_RADIUS_M, ARRIVAL_SPEED_MPS, CONTROL_DT)):
            raise ValueError('Arrival tolerances and control interval must be positive and finite')
        if PLOT_EVERY_TRIALS < 1:
            raise ValueError('PLOT_EVERY_TRIALS must be positive')
        if not 0 <= POLY_DEGREE < HISTORY_SAMPLES:
            raise ValueError('Require 0 <= POLY_DEGREE < HISTORY_SAMPLES')
        output = OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        print(f'Results: {output}', flush=True)
        scan_fields = ['trial', 'scan', 'elapsed_s', 'lidar_timestamp', 'points',
                       'true_world_x', 'true_world_y', 'control_yaw_deg',
                       'vx', 'vy', 'wz', 'raw_x', 'raw_y', 'corrected_x', 'corrected_y',
                       'true_x', 'true_y', 'raw_error_x', 'raw_error_y',
                       'corrected_error_x', 'corrected_error_y',
                       'raw_bearing_deg', 'ours_bearing_deg', 'gt_bearing_deg',
                       'raw_error_deg', 'ours_error_deg']
        trial_fields = ['trial', 'status', 'error', 'scans', 'empty_scans', 'elapsed_s', *ERROR_FIELDS]
        completed = []
        attempted = 0
        # 실행 설정도 CSV로 저장한다.
        with (output / 'settings.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'value'])
            writer.writerows([
                ['trial_count', TRIAL_COUNT],
                ['episode_completion', 'one full path then within arrival radius and speed'],
                ['arrival_radius_m', ARRIVAL_RADIUS_M], ['arrival_speed_mps', ARRIVAL_SPEED_MPS],
                ['speed_mps', SPEED], ['altitude_ned_m', ALTITUDE], ['latency_s', LATENCY],
                ['scan_time_s', SCAN_TIME], ['vehicle', vehicle_name], ['lidar', lidar_name],
                ['history_samples_N', HISTORY_SAMPLES], ['polynomial_degree_m', POLY_DEGREE],
                ['history_wait_interval_s', HISTORY_INTERVAL],
                ['object', OBJECT_NAME], ['metric', 'LiDAR point centroid minus static Cylinder_2 center, both in control-time UAV yaw frame'],
                ['evaluation_frame', 'UAV yaw frame; assumes LiDAR horizontal axes/origin match UAV'],
                ['yaw_rate_command_limit_deg_s', MAX_YAW_RATE_DEG_S],
                ['circle_path', 'x=x0+R*(1-cos(q)); y=y0-R*sin(q); q=0..2pi'],
                ['circle_diameter_m', CIRCLE_DIAMETER_M],
                ['nominal_cruise_yaw_rate_deg_s', math.degrees(SPEED/(CIRCLE_DIAMETER_M/2))],
                ['reference_path_length_m', planned.length], ['requested_cruise_speed_mps', planned.cruise],
                ['target_tangential_acceleration_mps2', PATH_ACCELERATION_MPS2],
                ['velocity_command_acceleration_limit_mps2', planned.command_acceleration],
                ['speed_semantics', 'requested cruise speed with feedback; startup and arrival ramps'],
                ['control_dt_s', CONTROL_DT],
                ['object_management', 'User places/spawns static object; script only reads its pose'],
                ['object_motion', 'Static in world NED frame'],
                ['point_selection', 'All returned LiDAR points; scan must represent Cylinder_2'],
                ['bearing_metric', 'Absolute wrapped atan2 difference in control-time UAV yaw frame (degrees)'],
                ['aggregation', 'Mean absolute error per scan; equal weight per completed trial']])
        try:
            with (output / 'scans.csv').open('w', newline='', encoding='utf-8-sig') as sf, \
                    (output / 'trials.csv').open('w', newline='', encoding='utf-8-sig') as tf:
                scan_writer = csv.DictWriter(sf, fieldnames=scan_fields)
                trial_writer = csv.DictWriter(tf, fieldnames=trial_fields)
                scan_writer.writeheader()
                trial_writer.writeheader()
                for trial in range(1, TRIAL_COUNT + 1):
                    result = {}
                    attempted += 1
                    try:
                        self.run_trial(trial, vehicle_name, lidar_name, scan_writer, sf, result)
                    finally:
                        # 오류나 Ctrl+C 발생 시에도 현재 회차의 부분 결과를 저장한다.
                        trial_writer.writerow(result)
                        tf.flush()
                        # A full flight counts even if no usable LiDAR scans arrived.
                        # Keep no_data explicit and exclude its missing error metrics.
                        if result.get('status') in ('completed', 'no_data'):
                            completed.append(result)
                        summary = {'requested_trials': TRIAL_COUNT, 'attempted_trials': attempted,
                                   'completed_trials': len(completed),
                                   'complete': len(completed) == TRIAL_COUNT}
                        summary.update({key: float(np.mean([r[key] for r in completed if r['scans']]))
                                        if any(r['scans'] for r in completed) else None for key in ERROR_FIELDS})
                        with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
                            writer = csv.DictWriter(f, fieldnames=list(summary))
                            writer.writeheader()
                            writer.writerow(summary)
                    print(f"Trial {trial}/{TRIAL_COUNT}: {result['status']}, scans={result['scans']}", flush=True)
                    if trial % PLOT_EVERY_TRIALS == 0:
                        self.save_motion_plots(output)
        finally:
            try:
                self.stop_circle()
            except Exception:
                traceback.print_exc()
            self.save_motion_plots(output)
            # 종료 때 진행 중인 비행 명령을 hover로 교체한다. 원래 예외는 가리지 않는다.
            try:
                self.client.hoverAsync(vehicle_name=vehicle_name)
            except Exception:
                traceback.print_exc()
        print(f'Completed trials: {len(completed)}/{TRIAL_COUNT}; results: {output}', flush=True)


if __name__ == '__main__':
    lidarTest = LidarTest()
    lidarTest.execute(VEHICLE_NAME, LIDAR_NAME)
