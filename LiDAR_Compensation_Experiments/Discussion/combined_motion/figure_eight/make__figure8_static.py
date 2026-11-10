"""90 m by 45 m figure-eight flight evaluating a static Cylinder_2.
Run: python make__figure8_static.py
Manually place the object before running; it must survive simulator reset.
OBJECT_NAME must exactly match the AirSim object name.
GT is object center minus control-time UAV position, rotated into UAV yaw frame.
Raw and Ours centroids are compared to this same GT for XY and bearing errors.
All returned LiDAR points are averaged: configure scans to represent this object.
A visible surface centroid may differ from the object's center even without delay.
Assumes LiDAR horizontal axes/origin match the UAV; planar yaw compensation.
The script only reads the object pose. Flight, timing CSVs and plots are retained.
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
PLOT_EVERY_TRIALS = 10
EPISODE_TIMEOUT_SECONDS = 300.0  # watchdog only; timeout is an error, not completion
FIGURE8_HEIGHT_M = 90.0  # total extent in world x (screen vertical)
FIGURE8_WIDTH_M = 45.0  # total extent in world y (screen horizontal)
PATH_ACCELERATION_MPS2 = 2.0
ARRIVAL_RADIUS_M = 1.0
ARRIVAL_SPEED_MPS = 0.5
SPEED = 15.0  # upper target; path curvature/acceleration may lower cruise speed
ALTITUDE = -10.0  # o_v.py
LATENCY = 0.2  # retain current experiment setting
SCAN_TIME = 0.1
HISTORY_SAMPLES = 10  # N: 관측 이전 운동 샘플 수
POLY_DEGREE = 2       # m: 다항식 차수
HISTORY_INTERVAL = 0.02  # 초기 이력 수집/이력 부족 시 대기 간격(초)
VEHICLE_NAME = 'Follower0'
LIDAR_NAME = 'Follower0_LidarSensor'
OBJECT_NAME = 'Cylinder_2'  # Exact AirSim name of the user-placed static object
MAX_YAW_RATE_DEG_S = 45.0  # command limit, never a constant target rate
CONTROL_DT = 0.05
OUTPUT_ROOT = Path(__file__).resolve().parent / 'experiment_results' / 'make__figure8_static'

ERROR_FIELDS = ['raw_x_mae_m', 'raw_y_mae_m', 'corrected_x_mae_m', 'corrected_y_mae_m',
                'raw_bearing_mae_deg', 'ours_bearing_mae_deg']


class FigureEightEpisode:
    def __init__(self, origin, height, width, speed, yaw_limit, acceleration=2.0):
        values = (height, width, speed, yaw_limit, acceleration)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Path dimensions and motion limits must be finite and positive')
        self.origin = np.array(origin, dtype=float)
        self.a, self.b = height / 2, width / 2
        self.speed, self.yaw_limit, self.acceleration = speed, math.radians(yaw_limit), acceleration
        self.phases = np.linspace(0, 2 * math.pi, 8001)
        points = np.array([self.geometry(q)[0] for q in self.phases])
        self.lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        self.length = float(self.lengths[-1])
        # Conservative global curvature limit keeps turns smooth without a
        # sudden local speed reduction; translational acceleration is also bounded.
        curvature = max(abs(self.geometry(q)[2]) for q in self.phases)
        self.cruise = min(speed, .7 * self.yaw_limit / curvature,
                          math.sqrt(acceleration / curvature))
        self.distance = 0.
        self.reference_speed = 0.
        self.command = np.zeros(2)
        self.yaw_command = 0.
        self.phase = 'accelerating'
        self.reference_yaw_rate_deg_s = 0.

    def geometry(self, q):
        # Screen up = world +x, screen right = world +y. Bottom start/finish.
        point = self.origin + [self.a * (1 - math.cos(q)), -self.b * math.sin(2*q)]
        d = np.array([self.a * math.sin(q), -2*self.b * math.cos(2*q)])
        dd = np.array([self.a * math.cos(q), 4*self.b * math.sin(2*q)])
        norm = np.linalg.norm(d)
        curvature = (d[0]*dd[1]-d[1]*dd[0]) / norm**3
        return point, d / norm, float(curvature)

    @property
    def progress(self):
        return min(1., self.distance / self.length)

    def update(self, position, velocity, yaw, dt, arrival_radius=1., arrival_speed=.5):
        dt = max(0., min(float(dt), .1))
        phase = float(np.interp(self.distance, self.lengths, self.phases))
        reference, tangent, curvature = self.geometry(phase)
        error = reference - position
        remaining = max(0., self.length-self.distance)
        # Stop advancing the reference when the aircraft falls behind; a nearby
        # crossing cannot skip either lobe because progress is strictly ordered.
        desired_speed = min(self.cruise, math.sqrt(2*self.acceleration*remaining))
        desired_speed *= max(0., 1-np.linalg.norm(error)/5.)
        self.reference_speed += float(np.clip(desired_speed-self.reference_speed,
                                              -self.acceleration*dt, self.acceleration*dt))
        at_end = remaining <= 0.
        if at_end:
            self.distance = self.length
            reference, tangent, curvature = self.geometry(2*math.pi)
            error = reference-position
            self.reference_speed = 0.
        self.phase = ('settling' if at_end else 'cruise'
                      if abs(self.reference_speed-self.cruise) < .05 else 'changing_speed')
        requested = tangent*self.reference_speed + .6*error
        norm = np.linalg.norm(requested)
        if norm > self.speed:
            requested *= self.speed/norm
        delta = requested-self.command
        change = np.linalg.norm(delta)
        if change > self.acceleration*dt:
            delta *= self.acceleration*dt/change
        self.command += delta
        heading = math.atan2(tangent[1], tangent[0])
        self.reference_heading_deg = math.degrees(heading)
        self.path_curvature = curvature
        self.reference_progress = self.progress
        heading_error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
        self.reference_yaw_rate_deg_s = math.degrees(curvature*self.reference_speed)
        desired_yaw = float(np.clip(curvature*self.reference_speed + 1.5*heading_error,
                                    -self.yaw_limit, self.yaw_limit))
        # Bound angular acceleration as well as yaw-rate magnitude.
        self.yaw_command += float(np.clip(desired_yaw-self.yaw_command,
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
        self.control_error = None
        self.episode_done = threading.Event()
        self.path_lock = threading.Lock()

    def start_trial(self, vehicle_name):
        # 매 회차 동일 시작 상태로 초기화한 뒤 기존 코드와 같은 순서로 비행한다.
        self.stop_flight()
        self.client.reset()
        self.vehicle_name = vehicle_name
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        self.client.takeoffAsync(vehicle_name=vehicle_name).join()
        self.client.moveToZAsync(ALTITUDE, 1.0, vehicle_name=vehicle_name).join()
        pose = self.client.simGetVehiclePose(vehicle_name=vehicle_name)
        if not hasattr(self, 'path_origin'):
            self.path_origin = (pose.position.x_val, pose.position.y_val)
        # Fix geometry across trials, including small takeoff/reset drift.
        self.client.moveToPositionAsync(*self.path_origin, ALTITUDE, 1.0,
                                        vehicle_name=vehicle_name).join()
        self.client.hoverAsync(vehicle_name=vehicle_name).join()
        self.path = FigureEightEpisode(
            self.path_origin, FIGURE8_HEIGHT_M,
            FIGURE8_WIDTH_M, SPEED, MAX_YAW_RATE_DEG_S, PATH_ACCELERATION_MPS2)
        self.episode_done.clear()
        self.read_static_object_pose()
        # Align to the initial tangent while stationary, before measurement.
        self.client.rotateToYawAsync(-90., vehicle_name=vehicle_name).join()
        time.sleep(1.0)
        print(f'Path length={self.path.length:.1f} m, cruise cap={self.path.cruise:.2f} m/s', flush=True)
        self.move(vehicle_name)
        self.check_control()

    def read_static_object_pose(self):
        """Validate the user-owned object; never create, move or delete it."""
        pose = self.client.simGetObjectPose(self.obj_name)
        xyz = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        if not np.isfinite(xyz).all():
            raise RuntimeError(
                f'Static object {self.obj_name!r} not found or has an invalid pose. '
                'Place it before running and verify OBJECT_NAME matches AirSim. '
                'It must remain present after client.reset().')
        return pose

    def move(self, vehicle_name):
        self.vehicle_name = vehicle_name
        self.stop_event.clear()
        self.control_error = None
        self.episode_start_ns = self.client.getMultirotorState(vehicle_name=vehicle_name).timestamp
        self.control_thread = threading.Thread(target=self.figure8_control_loop, daemon=True)
        self.control_thread.start()

    def figure8_control_loop(self):
        client = None
        try:
            client = airsim.MultirotorClient(timeout_value=3)
            client.confirmConnection()
            client.enableApiControl(True, vehicle_name=self.vehicle_name)
            previous_ns = previous_yaw = None
            with (self.output / f'trajectory_trial_{self.trial:03d}.csv').open(
                    'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_ns', 'progress', 'reference_x', 'reference_y',
                                 'actual_x', 'actual_y', 'actual_z', 'speed_mps',
                                 'yaw_rate_command_deg_s',
                                 'elapsed_sim_s', 'target_speed_mps', 'command_speed_mps',
                                 'actual_yaw_rate_deg_s', 'body_wz_deg_s',
                                 'target_yaw_rate_deg_s', 'phase', 'tracking_error_m',
                                 'actual_yaw_deg', 'reference_heading_deg', 'path_curvature_1_m'])
                while not self.stop_event.is_set():
                    state = client.getMultirotorState(vehicle_name=self.vehicle_name)
                    kin = state.kinematics_estimated
                    dt = 0. if previous_ns is None else (state.timestamp-previous_ns)*1e-9
                    if dt < 0:
                        raise RuntimeError('Simulator timestamp moved backwards')
                    position = np.array([kin.position.x_val, kin.position.y_val])
                    velocity = np.array([kin.linear_velocity.x_val, kin.linear_velocity.y_val])
                    _, _, yaw = airsim.to_eularian_angles(kin.orientation)
                    actual_yaw_rate = (math.degrees(math.atan2(math.sin(yaw-previous_yaw),
                                       math.cos(yaw-previous_yaw)))/dt
                                       if previous_yaw is not None and dt > 0 else float('nan'))
                    previous_ns, previous_yaw = state.timestamp, yaw
                    with self.path_lock:
                        command, rate, reference, done = self.path.update(
                            position, velocity, yaw, dt, ARRIVAL_RADIUS_M, ARRIVAL_SPEED_MPS)
                        progress = self.path.reference_progress
                    elapsed = (state.timestamp-self.episode_start_ns)*1e-9
                    writer.writerow([state.timestamp, progress, *reference, *position,
                                     kin.position.z_val, np.linalg.norm(velocity), rate,
                                     elapsed, self.path.reference_speed,
                                     np.linalg.norm(command), actual_yaw_rate,
                                     math.degrees(kin.angular_velocity.z_val),
                                     self.path.reference_yaw_rate_deg_s,
                                     self.path.phase, np.linalg.norm(reference-position),
                                     math.degrees(yaw), self.path.reference_heading_deg,
                                     self.path.path_curvature])
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
            raise RuntimeError('Figure-eight control failed') from self.control_error

    def stop_flight(self):
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
            print(f"Trial {trial}/{TRIAL_COUNT}: one full figure eight, returning to start", flush=True)
            while not self.episode_done.is_set():
                if time.monotonic() - started > EPISODE_TIMEOUT_SECONDS:
                    raise RuntimeError('Episode timeout: full path and arrival were not completed')
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
                timing_observation_received_ns = time.perf_counter_ns()
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
                timing_ours_start_ns = time.perf_counter_ns()
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
                timing_ours_processing_ms = (time.perf_counter_ns() - timing_ours_start_ns) * 1e-6
                print("finx : %f \n" %finx) 
                print("finy : %f \n" %finy) 

                notfinx = nottempx / size  
                notfiny = nottempy / size  
                notfinz = nottempz / size 

                
                # 스캔 + 의사결정 지연 시간동안 드론이 움직임 
                timing_wait_start_ns = time.perf_counter_ns()
                time.sleep(latency) 
                timing_wait_ms = (time.perf_counter_ns() - timing_wait_start_ns) * 1e-6
                timing_received_to_evaluation_ms = (time.perf_counter_ns() - timing_observation_received_ns) * 1e-6

                pose = self.client.simGetVehiclePose(vehicle_name=vehicle_name) 
                object_pose = self.read_static_object_pose() 
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
                    raise ValueError("Nonfinite position/error; check LiDAR data and static object pose")
                bearings = self.bearing_metrics(notfinx, notfiny, finx, finy, relx, rely)
                # 시간 계측만 추가: 기존 보정/Raw/GT 결과에는 사용하지 않는다.
                # 원래 점별 합산과 나눗셈을 Raw 단독으로 재현하여 시간을 측정한다.
                # GT 조회 이후에 수행하므로 이 추가 연산은 기존 평가 시점을 늦추지 않는다.
                timing_raw_start_ns = time.perf_counter_ns()
                timing_raw_x = timing_raw_y = timing_raw_z = 0.0
                for timing_i in range(0, len(lidar_data.point_cloud), 3):
                    timing_xyz = lidar_data.point_cloud[timing_i:timing_i + 3]
                    timing_raw_x += timing_xyz[0]
                    timing_raw_y += timing_xyz[1]
                    timing_raw_z += timing_xyz[2]
                timing_raw_center = (timing_raw_x / size, timing_raw_y / size, timing_raw_z / size)
                timing_raw_processing_ms = (time.perf_counter_ns() - timing_raw_start_ns) * 1e-6
                # Raw는 별도 비행/대기를 수행하지 않는다. 아래 전체 시간은 같은
                # 스캔시간과 실측 대기시간을 더한 추정값이며 실제 명령 적용시간이 아니다.
                timing_scan_ms = scantime * 1000.0 if size > 1 else 0.0
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
                    'timing_history_samples_N': HISTORY_SAMPLES,
                    'raw_processing_ms': timing_raw_processing_ms,
                    'ours_processing_ms': timing_ours_processing_ms,
                    'extra_processing_vs_raw_ms': timing_ours_processing_ms - timing_raw_processing_ms,
                    'measured_latency_wait_ms': timing_wait_ms,
                    'raw_observation_to_control_estimated_ms': timing_scan_ms + timing_raw_processing_ms + timing_wait_ms,
                    'ours_observation_to_control_estimated_ms': timing_scan_ms + timing_ours_processing_ms + timing_wait_ms,
                    'ours_received_to_evaluation_ms': timing_received_to_evaluation_ms,
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
                self.stop_flight()
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
        if TRIAL_COUNT < 1 or not math.isfinite(EPISODE_TIMEOUT_SECONDS) or EPISODE_TIMEOUT_SECONDS <= 0:
            raise ValueError('Trial count and episode timeout must be positive')
        planned = FigureEightEpisode((0, 0), FIGURE8_HEIGHT_M, FIGURE8_WIDTH_M,
                                     SPEED, MAX_YAW_RATE_DEG_S, PATH_ACCELERATION_MPS2)
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
                       'raw_error_deg', 'ours_error_deg',
                       'timing_history_samples_N', 'raw_processing_ms', 'ours_processing_ms',
                       'extra_processing_vs_raw_ms', 'measured_latency_wait_ms',
                       'raw_observation_to_control_estimated_ms',
                       'ours_observation_to_control_estimated_ms', 'ours_received_to_evaluation_ms']
        trial_fields = ['trial', 'status', 'error', 'scans', 'empty_scans', 'elapsed_s', *ERROR_FIELDS]
        completed = []
        attempted = 0
        # 실행 설정도 CSV로 저장한다.
        with (output / 'settings.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'value'])
            writer.writerows([
                ['trial_count', TRIAL_COUNT], ['episode_timeout_wall_s', EPISODE_TIMEOUT_SECONDS],
                ['episode_completion', 'one full path then within arrival radius and speed'],
                ['arrival_radius_m', ARRIVAL_RADIUS_M], ['arrival_speed_mps', ARRIVAL_SPEED_MPS],
                ['speed_mps', SPEED], ['altitude_ned_m', ALTITUDE], ['latency_s', LATENCY],
                ['scan_time_s', SCAN_TIME], ['vehicle', vehicle_name], ['lidar', lidar_name],
                ['history_samples_N', HISTORY_SAMPLES], ['polynomial_degree_m', POLY_DEGREE],
                ['history_wait_interval_s', HISTORY_INTERVAL],
                ['object', OBJECT_NAME], ['metric', 'LiDAR point centroid minus static object center, both in control-time UAV yaw frame'],
                ['evaluation_frame', 'UAV yaw frame; assumes LiDAR horizontal axes/origin match UAV'],
                ['max_yaw_rate_deg_s', MAX_YAW_RATE_DEG_S],
                ['figure8_path', 'x=x0+H/2*(1-cos(q)); y=y0-W/2*sin(2q); q=0..2pi'],
                ['figure8_height_m', FIGURE8_HEIGHT_M], ['figure8_width_m', FIGURE8_WIDTH_M],
                ['reference_path_length_m', planned.length], ['planned_cruise_cap_mps', planned.cruise],
                ['acceleration_limit_mps2', PATH_ACCELERATION_MPS2],
                ['speed_semantics', 'upper target; curvature and acceleration constrained'],
                ['control_dt_s', CONTROL_DT],
                ['object_management', 'User-placed static object; pose read only'],
                ['point_selection', 'All returned LiDAR points; scan must represent the static object'],
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
                        if result.get('status') == 'completed':
                            completed.append(result)
                        summary = {'requested_trials': TRIAL_COUNT, 'attempted_trials': attempted,
                                   'completed_trials': len(completed),
                                   'complete': len(completed) == TRIAL_COUNT}
                        summary.update({key: float(np.mean([r[key] for r in completed]))
                                        if completed else None for key in ERROR_FIELDS})
                        with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
                            writer = csv.DictWriter(f, fieldnames=list(summary))
                            writer.writeheader()
                            writer.writerow(summary)
                    print(f"Trial {trial}/{TRIAL_COUNT}: {result['status']}, scans={result['scans']}", flush=True)
                    if trial % PLOT_EVERY_TRIALS == 0:
                        self.save_motion_plots(output)
        finally:
            try:
                self.stop_flight()
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
