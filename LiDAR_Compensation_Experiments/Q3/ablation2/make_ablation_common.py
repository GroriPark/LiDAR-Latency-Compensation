"""Paired AirSim ablations based on make__.py's planar motion equations.

All variants share a scan, frozen pre-observation model and application-time truth.
Static-scene uniform scans are reconstructed with independent ground-truth poses;
these are not native rolling LiDAR measurements. No simulator connection on import.
"""
import argparse
import csv
import json
import math
import shutil
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from fixed_s_path import FixedSPath, tracking_velocity


@dataclass
class Config:
    trials: int = 100
    duration: float = 15.0
    scan_time: float = 0.1
    generation_delay: float = 0.2  # tM -> command generated (NOT paper's l1)
    application_delay: float = 0.1  # command generated -> applied
    history_samples: int = 10
    degree: int = 2
    sample_interval: float = 0.02
    speed_min: float = 5.0
    speed_max: float = 10.0
    speed_period: float = 6.0
    turn_period: float = 8.0
    heading_amplitude: float = 30.0  # deg: alternates left/right, always forward
    warmup: float = 3.0
    altitude: float = -20.0
    vehicle: str = 'Follower0'
    lidar: str = 'Follower0_LidarSensor'
    object: str = 'Cube_2'
    output: str = ''
    path_tolerance: float = 3.0
    endpoint_tolerance: float = 2.0
    tracking_gain: float = 1.5
    tracking_response_time: float = 3.0
    tracking_damping: float = 2.0

    def validate(self):
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError('All numeric settings must be finite')
        if min(self.path_tolerance, self.endpoint_tolerance, self.tracking_gain) <= 0:
            raise ValueError('Path tolerances and tracking gain must be positive')
        if min(self.tracking_response_time, self.tracking_damping) < 0:
            raise ValueError('Tracking response time and damping must be nonnegative')
        if self.trials < 1 or not 0 <= self.degree < self.history_samples:
            raise ValueError('Require trials >= 1 and 0 <= degree < history_samples')
        if min(self.duration, self.scan_time, self.sample_interval,
               self.speed_period, self.turn_period, self.speed_min) <= 0:
            raise ValueError('Durations, sampling interval and speed_min must be positive')
        if min(self.generation_delay, self.application_delay, self.warmup) < 0:
            raise ValueError('Delays and warmup must be nonnegative')
        if not self.speed_min <= self.speed_max or not 0 < self.heading_amplitude < 80:
            raise ValueError('Require speed_min <= speed_max and 0 < heading_amplitude < 80')
        if self.sample_interval > self.scan_time / 2:
            raise ValueError('sample_interval must be <= scan_time/2 for pose interpolation')


METHODS = {
    1: ('scan_only', 'scan_generation', 'full'),
    2: ('Full', 'w/o T1', 'w/o T1,T3', 'w/o T1,T2,T3', 'Raw'),
}
METRICS = ('raw_x_mae_m', 'raw_y_mae_m', 'corrected_x_mae_m', 'corrected_y_mae_m',
           'raw_position_mae_m', 'corrected_position_mae_m',
           'raw_bearing_mae_deg', 'corrected_bearing_mae_deg',
           'position_error_m', 'bearing_error_deg')


def rotate(points, angle):
    """Yaw rotation, supporting one angle per point."""
    points = np.asarray(points, dtype=float)
    out = points.copy()
    c, s = np.cos(angle), np.sin(angle)
    out[..., 0] = c * points[..., 0] - s * points[..., 1]
    out[..., 1] = s * points[..., 0] + c * points[..., 1]
    return out


def motion_command(t, cfg):
    # Start at 5 m/s; smooth acceleration/deceleration between 5 and 10.
    speed = cfg.speed_min + (cfg.speed_max-cfg.speed_min) * (1-np.cos(2*np.pi*t/cfg.speed_period))/2
    heading = np.deg2rad(cfg.heading_amplitude) * np.sin(2*np.pi*t/cfg.turn_period)
    return float(speed), float(heading)


def fit_models(history, observation_ns, cfg):
    """Freeze models using strictly earlier estimated states, never future truth."""
    samples = [s for s in history if s[0] < observation_ns][-cfg.history_samples:]
    if len(samples) < cfg.history_samples:
        return None
    times = np.array([(s[0]-observation_ns)*1e-9 for s in samples])
    values = np.array([s[1:4] for s in samples])
    if np.any(np.diff(times) <= 0) or not np.isfinite(values).all():
        raise ValueError('Invalid motion history')
    scale = max(float(np.max(np.abs(times))), 1e-9)
    design = np.polynomial.polynomial.polyvander(times/scale, cfg.degree)
    coeff, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != cfg.degree+1:
        raise ValueError('Rank-deficient motion regression')
    polynomial = coeff.T / scale**np.arange(cfg.degree+1)
    # Last observed vx, vy, wz, held constant over the ENTIRE interval.
    held = values[-1, :, None].copy()
    return polynomial, held, samples[-1][0]


def correct(points, ti, target, coefficients, apply_rotation=True):
    """make__.py equations (7)-(11), vectorized in bounded chunks.

    Same 101-node trapezoidal integration. Translation is p-d; only T3
    enables R(-delta_yaw)(p-d). Integrated body velocity uses yaw in both cases.
    Constant body velocities still produce a curved path when wz != 0.
    """
    points = np.asarray(points, dtype=float)
    ti = np.broadcast_to(np.asarray(ti, dtype=float), (len(points),))
    if np.any(ti > target):
        raise ValueError('Compensation endpoint precedes a measurement')
    result = np.empty_like(points)
    poly = np.polynomial.polynomial
    yaw_integral = poly.polyint(coefficients[2])
    trap = getattr(np, 'trapezoid', None) or np.trapz
    for start in range(0, len(points), 1024):
        stop = min(start+1024, len(points))
        lower = ti[start:stop, None]
        grid = lower + (target-lower)*np.linspace(0, 1, 101)
        phi = poly.polyval(grid, yaw_integral)-poly.polyval(lower, yaw_integral)
        vx, vy = poly.polyval(grid, coefficients[0]), poly.polyval(grid, coefficients[1])
        delta = np.zeros((stop-start, 3))
        delta[:, 0] = trap(vx*np.cos(phi)-vy*np.sin(phi), grid, axis=1)
        delta[:, 1] = trap(vx*np.sin(phi)+vy*np.cos(phi), grid, axis=1)
        translated = points[start:stop]-delta
        result[start:stop] = rotate(translated, -phi[:, -1]) if apply_rotation else translated
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite correction')
    return result


def variants(experiment, raw, ti, polynomial, held, generation, application):
    if experiment == 1:
        return {name: correct(raw, ti, endpoint, polynomial) for name, endpoint in
                zip(METHODS[1], (0.0, generation, application))}
    if experiment == 2:
        # T1: point-specific acquisition times. Without T1 all points are
        # assumed measured at tM (0), while the reconstructed raw is unchanged.
        # T2: polynomial prediction versus last pre-observation sample held.
        # T3: final coordinate rotation; yaw inside translation is preserved.
        simultaneous = np.zeros(len(raw), dtype=float)
        return {'Full': correct(raw, ti, application, polynomial),
                'w/o T1': correct(raw, simultaneous, application, polynomial),
                'w/o T1,T3': correct(raw, simultaneous, application, polynomial, apply_rotation=False),
                'w/o T1,T2,T3': correct(raw, simultaneous, application, held, apply_rotation=False),
                'Raw': raw.copy()}
    raise ValueError('Unknown experiment')


def vec(v):
    return np.array([v.x_val, v.y_val, v.z_val], dtype=float)


def yaw_of(q):
    return math.atan2(2*(q.w_val*q.z_val+q.x_val*q.y_val),
                      1-2*(q.y_val*q.y_val+q.z_val*q.z_val))


def sensor_to_world(points, pose):
    q = pose.orientation
    w, x, y, z = np.array([q.w_val, q.x_val, q.y_val, q.z_val], dtype=float)
    norm = np.linalg.norm([w, x, y, z])
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid LiDAR orientation')
    w, x, y, z = np.array([w, x, y, z])/norm
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return points @ rotation.T + vec(pose.position)


def reconstruct_scan(world, history, observation_ns, scan_time):
    """Static world points -> independent GT poses at uniform point times."""
    ti = np.linspace(-scan_time, 0, len(world)) if len(world) > 1 else np.zeros(1)
    times = np.array([(s[0]-observation_ns)*1e-9 for s in history])
    if ti[0] < times[0] or ti[-1] > times[-1]:
        return None
    positions = np.array([s[4] for s in history])
    yaws = np.unwrap([s[5] for s in history])
    position = np.column_stack([np.interp(ti, times, positions[:, i]) for i in range(3)])
    yaw = np.interp(ti, times, yaws)
    return rotate(world-position, -yaw), ti


def errors(raw, corrected, truth):
    """Same static surface points at tc; equal scan weighting in summaries."""
    result = {}
    for label, points in (('raw', raw), ('corrected', corrected)):
        diff = points[:, :2]-truth[:, :2]
        result[label+'_x_mae_m'] = float(np.mean(np.abs(diff[:, 0])))
        result[label+'_y_mae_m'] = float(np.mean(np.abs(diff[:, 1])))
        result[label+'_position_mae_m'] = float(np.mean(np.linalg.norm(diff, axis=1)))
        valid = (np.linalg.norm(points[:, :2], axis=1) > 1e-9) & (np.linalg.norm(truth[:, :2], axis=1) > 1e-9)
        angle = np.arctan2(points[valid, 1], points[valid, 0])-np.arctan2(truth[valid, 1], truth[valid, 0])
        result[label+'_bearing_mae_deg'] = (float(np.mean(np.abs(np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle))))))
                                            if valid.any() else None)
    return result


class FlushingWriter:
    def __init__(self, writer, file):
        self.writer, self.file = writer, file

    def writerow(self, row):
        self.writer.writerow(row)
        self.file.flush()


def sample_std(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def average(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_excel(output):
    """CSV is flushed during flight; XLSX is rebuilt even after interruption."""
    from openpyxl import Workbook
    wb = Workbook(write_only=True)
    for name in ('settings', 'scans', 'trials', 'summary', 'trajectory'):
        path = output / (name+'.csv')
        if not path.exists():
            continue
        ws = wb.create_sheet(name)
        with path.open(encoding='utf-8-sig', newline='') as f:
            for i, row in enumerate(csv.reader(f)):
                if i >= 1048576:
                    raise ValueError('Excel row limit exceeded; complete data remains in CSV')
                values = []
                for v in row:
                    try:
                        # Retain nanosecond timestamps as text: Excel has only 15 digits.
                        value = float(v) if v and len(v.lstrip('-')) < 16 else v
                    except ValueError:
                        value = v
                    values.append(value)
                ws.append(values)
    wb.save(output / 'results.xlsx')


class Experiment:
    def __init__(self, cfg, experiment, client=None):
        import airsim
        self.airsim = airsim
        self.cfg, self.experiment = cfg, experiment
        self.client = client if client is not None else airsim.MultirotorClient()
        self.client.confirmConnection()
        self.history = deque()
        self.current = None
        self.origin_ns = None
        self.collision_monitor_start_ns = None
        self.reported_stale_collision_ns = None
        self.path = FixedSPath(cfg.speed_min, cfg.speed_max) if experiment == 2 else None
        self.path_active = False
        self.path_origin = None
        self.telemetry = {}
        self.actual_path_length = 0.0
        self.max_tracking_error = 0.0

    def read_state(self):
        cfg, client = self.cfg, self.client
        state = client.getMultirotorState(vehicle_name=cfg.vehicle)
        truth = client.simGetGroundTruthKinematics(vehicle_name=cfg.vehicle)
        yaw = yaw_of(state.kinematics_estimated.orientation)
        self.estimated_yaw = yaw
        body = rotate(vec(state.kinematics_estimated.linear_velocity), -yaw)
        sample = (int(state.timestamp), body[0], body[1],
                  state.kinematics_estimated.angular_velocity.z_val,
                  vec(truth.position), yaw_of(truth.orientation))
        if not np.isfinite(np.r_[sample[1:4], sample[4], sample[5]]).all():
            raise ValueError('Nonfinite motion state')
        if self.history and sample[0] < self.history[-1][0]:
            raise ValueError('Simulator timestamp moved backwards')
        if not self.history or sample[0] > self.history[-1][0]:
            self.history.append(sample)
        horizon = max(5.0, cfg.scan_time+cfg.history_samples*cfg.sample_interval*4,
                      cfg.generation_delay+cfg.application_delay+1)
        while len(self.history) > cfg.history_samples+2 and (sample[0]-self.history[1][0])*1e-9 > horizon:
            self.history.popleft()
        self.current = sample
        return sample

    def step(self, seconds, trajectory_writer, trial):
        cfg, client = self.cfg, self.client
        before = self.current[0]
        before_position = self.current[4][:2].copy()
        elapsed = max(0.0, (before-self.origin_ns)*1e-9)
        speed, heading = motion_command(elapsed, cfg)
        velocity = speed*np.array([np.cos(heading), np.sin(heading)])
        progress, reference, yaw_rate = 0., self.current[4][:2].copy(), 0.
        if self.path is not None:
            if self.path_active:
                distance = self.path.project(self.current[4][:2]-self.path_origin)
                progress, offset, heading, speed, yaw_rate = self.path.at_distance(distance)
                reference = self.path_origin+offset
                actual_velocity = rotate(np.array([self.current[1], self.current[2], 0.]),
                                         self.estimated_yaw)[:2]
                velocity = tracking_velocity(
                    heading, speed, yaw_rate/speed, reference-self.current[4][:2],
                    actual_velocity, cfg.tracking_gain, cfg.tracking_response_time,
                    cfg.tracking_damping)
            else:
                heading, speed = float(self.path.heading[0]), 0.
                velocity = np.zeros(2)
                reference = self.path_origin.copy()
        client.moveByVelocityZAsync(
            float(velocity[0]), float(velocity[1]), cfg.altitude,
            max(1.0, 4*seconds), self.airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=self.airsim.YawMode(False, float(np.rad2deg(heading))), vehicle_name=cfg.vehicle)
        # A remainder shorter than a physics/state update tick can return the
        # same timestamp. Retry with a full sampling interval (at least 20 ms).
        # Never invent timestamps or treat a permanently stalled clock as success.
        for retry in range(3):
            requested = float(seconds) if retry == 0 else max(cfg.sample_interval, 0.02)
            client.simContinueForTime(requested)
            deadline = time.monotonic()+15.0
            while not client.simIsPause():
                if time.monotonic() > deadline:
                    raise TimeoutError('Simulator did not pause after simContinueForTime')
                time.sleep(0.001)
            s = self.read_state()
            if s[0] > before:
                break
        else:
            raise RuntimeError(
                f'Simulator time did not advance after 3 attempts; '
                f'timestamp={before}, initial_request_s={seconds:.9g}, '
                f'retry_request_s={requested:.9g}')
        if self.path is not None and self.path_active:
            distance = self.path.project(s[4][:2]-self.path_origin)
            progress, offset, _, _, _ = self.path.at_distance(distance)
            reference = self.path_origin+offset
        tracking_error = float(np.linalg.norm(s[4][:2]-reference))
        if self.path is not None and self.path_active:
            self.actual_path_length += float(np.linalg.norm(s[4][:2]-before_position))
            self.max_tracking_error = max(self.max_tracking_error, tracking_error)
        self.telemetry = dict(path_progress=progress, commanded_speed_mps=speed,
                              actual_speed_mps=float(np.hypot(s[1], s[2])),
                              commanded_yaw_rate_deg_s=float(np.rad2deg(yaw_rate)),
                              actual_yaw_rate_deg_s=float(np.rad2deg(s[3])))
        trajectory_writer.writerow([trial, s[0], (s[0]-self.origin_ns)*1e-9,
                                    speed, np.rad2deg(heading), np.hypot(s[1], s[2]),
                                    s[1], s[2], s[3], *s[4], np.rad2deg(s[5]),
                                    seconds, (s[0]-before)*1e-9, retry,
                                    progress, *reference, tracking_error, np.rad2deg(yaw_rate),
                                    np.rad2deg(s[3]), *velocity, int(self.path_active)])
        if self.path is not None and self.path_active and tracking_error > cfg.path_tolerance:
            raise RuntimeError(f'Fixed-path tracking error {tracking_error:.3f} m exceeds tolerance')
        self.check_collision()

    def check_collision(self):
        """Keep the rejected state in trajectory.csv and report the collision evidence.

        Ignore dated contact records from before post-takeoff monitoring began.
        Keep this cutoff fixed through warmup and the path clock reset. Missing or
        invalid timestamps cannot establish stale contact and still cause failure.
        """
        collision = self.client.simGetCollisionInfo(vehicle_name=self.cfg.vehicle)
        if not collision.has_collided:
            return
        timestamp = getattr(collision, 'time_stamp', None)
        if (isinstance(timestamp, (int, np.integer)) and timestamp > 0
                and self.collision_monitor_start_ns is not None
                and timestamp < self.collision_monitor_start_ns):
            if timestamp != self.reported_stale_collision_ns:
                print(f'Ignoring pre-monitor collision record: '
                      f'object={getattr(collision, "object_name", "")!r}, '
                      f'collision_timestamp={timestamp}, '
                      f'monitor_start_timestamp={self.collision_monitor_start_ns}', flush=True)
                self.reported_stale_collision_ns = timestamp
            return
        name = getattr(collision, 'object_name', '') or '<unknown>'
        object_id = getattr(collision, 'object_id', None)
        impact = getattr(collision, 'impact_point', None)
        impact_xyz = vec(impact).tolist() if impact is not None else None
        raise RuntimeError(
            f'Collision: object={name!r}, object_id={object_id}, '
            f'collision_timestamp={timestamp}, state_timestamp={self.current[0]}, '
            f'motion_start_timestamp={self.origin_ns}, '
            f'collision_monitor_start_timestamp={self.collision_monitor_start_ns}, '
            f'uav_position_ned={self.current[4].tolist()}, impact_point={impact_xyz}; '
            'this trial is excluded from completed-trial summary')

    def advance_to(self, target_ns, trajectory_writer, trial):
        while self.current[0] < target_ns:
            seconds = min(self.cfg.sample_interval, (target_ns-self.current[0])*1e-9)
            self.step(seconds, trajectory_writer, trial)

    def setup(self):
        cfg, client = self.cfg, self.client
        client.simPause(False)
        client.reset()
        client.enableApiControl(True, vehicle_name=cfg.vehicle)
        client.armDisarm(True, vehicle_name=cfg.vehicle)
        client.takeoffAsync(vehicle_name=cfg.vehicle).join()
        client.moveToZAsync(cfg.altitude, 1.0, vehicle_name=cfg.vehicle).join()
        client.simPause(True)
        self.path_active = False
        self.actual_path_length = 0.0
        self.max_tracking_error = 0.0
        self.history.clear()
        self.read_state()
        self.origin_ns = self.current[0]
        self.collision_monitor_start_ns = self.current[0]
        self.reported_stale_collision_ns = None
        if self.path is not None:
            start = self.current[4][:2].copy()
            if self.path_origin is None:
                self.path_origin = start
            elif np.linalg.norm(start-self.path_origin) > .1:
                raise RuntimeError('Trial start differs from fixed path origin by > 0.1 m')

    def run_trial(self, trial, scan_writer, scan_file, trajectory_writer, result):
        cfg, client = self.cfg, self.client
        rows = {name: [] for name in METHODS[self.experiment]}
        result.update(status='running', error='', scans=0, empty_scans=0,
                      skipped_history=0, duplicate_scans=0, stale_scans=0, elapsed_s=0.0,
                      endpoint_error_m=None, start_x=None, start_y=None,
                      reference_path_length_m=self.path.length if self.path else None)
        started_ns = None
        try:
            self.setup()
            warmup = max(cfg.warmup, cfg.scan_time+(cfg.history_samples+2)*cfg.sample_interval)
            self.advance_to(self.origin_ns+int(warmup*1e9), trajectory_writer, trial)
            started_ns = self.current[0]
            # Keep the command waveform continuous through warmup.
            if self.path is not None:
                if np.linalg.norm(self.current[4][:2]-self.path_origin) > .1:
                    raise RuntimeError('Warmup drifted > 0.1 m from fixed path start')
                result.update(start_x=float(self.current[4][0]), start_y=float(self.current[4][1]))
                self.origin_ns = started_ns
                self.path_active = True
            # Spatial completion, with a bounded watchdog for stalled vehicles.
            end_ns = started_ns+round((3*self.path.length/cfg.speed_min if self.path else cfg.duration)*1e9)
            previous_scan = None
            while self.current[0] < end_ns:
                if self.path is not None and self.path.project(self.current[4][:2]-self.path_origin) >= self.path.length-.1:
                    break
                lidar = client.getLidarData(lidar_name=cfg.lidar, vehicle_name=cfg.vehicle)
                observation_ns = int(lidar.time_stamp)
                if observation_ns == previous_scan:
                    result['duplicate_scans'] += 1
                    self.step(cfg.sample_interval, trajectory_writer, trial)
                    continue
                previous_scan = observation_ns
                if not lidar.point_cloud:
                    result['empty_scans'] += 1
                    self.step(cfg.sample_interval, trajectory_writer, trial)
                    continue
                if len(lidar.point_cloud) % 3:
                    raise ValueError('LiDAR point_cloud length must be a multiple of 3')
                points = np.asarray(lidar.point_cloud, dtype=float).reshape(-1, 3)
                if not np.isfinite(points).all():
                    raise ValueError('Nonfinite LiDAR points')
                if observation_ns > self.current[0]:
                    raise ValueError('LiDAR timestamp is ahead of simulator state')
                generation_ns = observation_ns+round(cfg.generation_delay*1e9)
                target_ns = generation_ns+round(cfg.application_delay*1e9)
                if self.current[0] > target_ns:
                    result['stale_scans'] += 1
                    self.step(cfg.sample_interval, trajectory_writer, trial)
                    continue
                if target_ns > end_ns:
                    self.advance_to(end_ns, trajectory_writer, trial)
                    break
                models = fit_models(self.history, observation_ns, cfg)
                world = sensor_to_world(points, lidar.pose) if self.frame == 'SensorLocalFrame' else points
                reconstructed = reconstruct_scan(world, self.history, observation_ns, cfg.scan_time)
                if models is None or reconstructed is None:
                    result['skipped_history'] += 1
                    self.step(cfg.sample_interval, trajectory_writer, trial)
                    continue
                raw, ti = reconstructed
                polynomial, held, last_sample_ns = models
                # No ground truth or post-observation states enter either frozen model.
                self.advance_to(target_ns, trajectory_writer, trial)
                actual_ns, vx, vy, wz, position, yaw = self.current
                actual_delay = (actual_ns-observation_ns)*1e-9
                corrected = variants(self.experiment, raw, ti, polynomial, held,
                                     cfg.generation_delay, actual_delay)
                truth = rotate(world-position, -yaw)
                object_position = vec(client.simGetObjectPose(cfg.object).position)
                cube = rotate(object_position-position, -yaw)
                raw_centroid = raw.mean(axis=0)
                truth_centroid = truth.mean(axis=0)
                scan_rows = []
                for name, cloud in corrected.items():
                    centroid = cloud.mean(axis=0)
                    row = dict(trial=trial, method=name, scan=result['scans']+1,
                               elapsed_s=(actual_ns-started_ns)*1e-9,
                               lidar_timestamp=observation_ns, application_timestamp=actual_ns,
                               last_motion_timestamp=last_sample_ns, points=len(raw),
                               generation_delay_s=cfg.generation_delay,
                               actual_latency_s=actual_delay,
                               latency_overshoot_s=(actual_ns-target_ns)*1e-9,
                               assumed_measurement_time='per_point' if name == 'Full' else 'none' if name == 'Raw' else 'scan_end_tM',
                               compensation_end_s={'Raw': 0.0, 'scan_only': 0.0, 'scan_generation': cfg.generation_delay}.get(name, actual_delay),
                               vx=vx, vy=vy, wz=wz,
                               raw_x=raw_centroid[0], raw_y=raw_centroid[1],
                               corrected_x=centroid[0], corrected_y=centroid[1],
                               true_x=truth_centroid[0], true_y=truth_centroid[1],
                               raw_error_x=raw_centroid[0]-truth_centroid[0],
                               raw_error_y=raw_centroid[1]-truth_centroid[1],
                               corrected_error_x=centroid[0]-truth_centroid[0],
                               corrected_error_y=centroid[1]-truth_centroid[1],
                               cube_true_x=cube[0], cube_true_y=cube[1],
                               cube_raw_error_x=raw_centroid[0]-cube[0],
                               cube_raw_error_y=raw_centroid[1]-cube[1],
                               cube_corrected_error_x=centroid[0]-cube[0],
                               cube_corrected_error_y=centroid[1]-cube[1])
                    row.update(errors(raw, cloud, truth))
                    row.update(self.telemetry)
                    relative = errors(raw_centroid[None, :], centroid[None, :], truth_centroid[None, :])
                    row['position_error_m'] = relative['corrected_position_mae_m']
                    row['bearing_error_deg'] = relative['corrected_bearing_mae_deg']
                    scan_rows.append(row)
                # Commit all paired methods together only after every method succeeds.
                if scan_writer[0] is None:
                    scan_writer[0] = csv.DictWriter(scan_file, fieldnames=list(scan_rows[0]))
                    scan_writer[0].writeheader()
                scan_writer[0].writerows(scan_rows)
                scan_file.flush()
                for row in scan_rows:
                    rows[row['method']].append({key: row[key] for key in METRICS})
                result['scans'] += 1
                if result['scans'] % 10 == 0:
                    print(f"Trial {trial}/{cfg.trials}: {result['scans']} paired scans, "
                          f"speed={np.hypot(vx, vy):.2f} m/s, yaw={np.rad2deg(yaw):.1f} deg", flush=True)
            if self.path is not None:
                if self.path.project(self.current[4][:2]-self.path_origin) < self.path.length-.1:
                    raise RuntimeError('Fixed path completion timeout: vehicle did not reach destination')
                result['endpoint_error_m'] = float(np.linalg.norm(self.current[4][:2]-self.path_origin-self.path.xy[-1]))
                if result['endpoint_error_m'] > cfg.endpoint_tolerance:
                    raise RuntimeError('Fixed path endpoint tolerance exceeded')
            result['status'] = 'completed' if result['scans'] else 'no_data' 
        except KeyboardInterrupt:
            result.update(status='interrupted', error='KeyboardInterrupt')
            raise
        except Exception as exc:
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if started_ns is not None:
                result['elapsed_s'] = (self.current[0]-started_ns)*1e-9
            result['actual_path_length_m'] = self.actual_path_length if self.path else None
            result['max_tracking_error_m'] = self.max_tracking_error if self.path else None
            result['methods'] = [dict(trial=trial, method=name,
                                      **{k: v for k, v in result.items() if k != 'methods'},
                                      **{key: average(rows[name], key) for key in METRICS},
                                      **{key+'_std': sample_std(rows[name], key) for key in METRICS})
                                 for name in METHODS[self.experiment]]

    def execute(self):
        cfg, client = self.cfg, self.client
        cfg.validate()
        settings = json.loads(client.getSettingsString())
        sensor = settings.get('Vehicles', {}).get(cfg.vehicle, {}).get('Sensors', {}).get(cfg.lidar)
        if sensor is None or sensor.get('SensorType') != 6 or not sensor.get('Enabled', True):
            raise ValueError('Named enabled LiDAR sensor not found in server settings')
        self.frame = sensor.get('DataFrame', 'VehicleInertialFrame') or 'VehicleInertialFrame'
        if self.frame not in ('SensorLocalFrame', 'VehicleInertialFrame'):
            raise ValueError('Unsupported LiDAR DataFrame: '+self.frame)
        root = Path(cfg.output) if cfg.output else Path(__file__).resolve().parent/'experiment_results'/f'make_ablation_{self.experiment}'
        output = root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        print(f'Results: {output}', flush=True)
        saved = dict(asdict(cfg), experiment=self.experiment, data_frame=self.frame,
                     ablation='Full; w/o T1; w/o T1,T3; w/o T1,T2,T3; Raw',
                     T1='point acquisition time; disabled: all ti=0 at scan end tM',
                     T2='polynomial prediction; disabled: hold last pre-tM vx/vy/wz',
                     T3='final yaw rotation; yaw inside translation retained',
                     scan_model='static_surface_uniform_scan_reconstructed_from_independent_GT',
                     common_timestamp='tM (scan end)', history_cutoff='strictly before tM',
                     timing='simulation seconds; paused during host calculation',
                     metric='planar pointwise errors against same static surface at application time',
                     legacy_metric='cube_* centroid minus Cube_2; no object segmentation',
                     aggregation='equal scan means; equal completed trial means',
                     paper_l1_s=cfg.scan_time+cfg.generation_delay,
                     paper_l2_s=cfg.application_delay,
                     scene_assumption='static surfaces; no dynamic objects',
                     primary_metric='2D centroid distance and absolute wrapped centroid bearing error',
                     path_model='fixed arc-length S path' if self.path else 'time sinusoid',
                     path_length_m=self.path.length if self.path else None,
                     planned_duration_s=self.path.duration if self.path else cfg.duration,
                     duration_option='ignored for experiment 2; spatial arrival with 3*length/speed_min watchdog',
                     path_progress_model='nearest segment projection of actual position; time table is reference only',
                     summary_std='sample std of completed trial means (ddof=1)',
                     trial_std='sample std of scan errors within each trial (ddof=1)')
        write_csv(output/'settings.csv', [dict(name=k, value=v) for k, v in saved.items()])
        (output/'server_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        for source in ('fixed_s_path.py', 'make_ablation_common.py', f'make_ablation_{self.experiment}.py'):
            shutil.copyfile(Path(__file__).with_name(source), output/source)
        if self.path is not None:
            write_csv(output/'reference_path.csv', [dict(arc_length_m=float(d), path_progress=float(d/self.path.length),
                      local_x=float(xy[0]), local_y=float(xy[1]), heading_rad=float(h),
                      curvature_per_m=float(k), speed_mps=float(v), elapsed_s=float(t))
                      for d, xy, h, k, v, t in zip(self.path.s, self.path.xy, self.path.heading,
                                                  self.path.curvature, self.path.speed, self.path.time)])
        completed, attempted = [], 0
        try:
            with (output/'scans.csv').open('w', newline='', encoding='utf-8-sig') as sf, \
                 (output/'trials.csv').open('w', newline='', encoding='utf-8-sig') as tf, \
                 (output/'trajectory.csv').open('w', newline='', encoding='utf-8-sig') as pf:
                scan_writer, trial_writer = [None], None
                trajectory_writer = csv.writer(pf)
                trajectory_writer.writerow(['trial', 'timestamp', 'motion_elapsed_s', 'commanded_speed_mps',
                                            'command_heading_deg', 'actual_speed_mps', 'vx', 'vy', 'wz',
                                            'x', 'y', 'z', 'yaw_deg',
                                            'step_requested_s', 'step_advanced_s', 'step_retries',
                                            'path_progress', 'reference_x', 'reference_y', 'tracking_error_m',
                                            'commanded_yaw_rate_deg_s', 'actual_yaw_rate_deg_s',
                                            'command_vx', 'command_vy', 'path_active'])
                trajectory_writer = FlushingWriter(trajectory_writer, pf)
                for trial in range(1, cfg.trials+1):
                    result = {}
                    attempted += 1
                    print(f'Trial {trial}/{cfg.trials}: reset / takeoff / warmup', flush=True)
                    try:
                        self.run_trial(trial, scan_writer, sf, trajectory_writer, result)
                    finally:
                        method_rows = result['methods']
                        if trial_writer is None:
                            trial_writer = csv.DictWriter(tf, fieldnames=list(method_rows[0]))
                            trial_writer.writeheader()
                        trial_writer.writerows(method_rows)
                        tf.flush()
                        pf.flush()
                        if result['status'] == 'completed':
                            completed.extend(method_rows)
                        summaries = []
                        for name in METHODS[self.experiment]:
                            records = [r for r in completed if r['method'] == name]
                            summary = dict(method=name, requested_trials=cfg.trials, attempted_trials=attempted,
                                           completed_trials=len(records), complete=len(records) == cfg.trials)
                            for key in METRICS:
                                summary[key] = average(records, key)
                                vals = [r[key] for r in records if r[key] is not None]
                                summary[key+'_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
                            summaries.append(summary)
                        write_csv(output/'summary.csv', summaries)
                    print(f"Trial {trial}: {result['status']}, paired scans={result['scans']}", flush=True)
        finally:
            # Cleanup must not hide the original error. CSV survives even XLSX failure.
            try:
                client.hoverAsync(vehicle_name=cfg.vehicle)
                client.simPause(False)
            except Exception:
                traceback.print_exc()
            try:
                export_excel(output)
            except Exception:
                print('XLSX export failed; CSV logs remain available.', flush=True)
                traceback.print_exc()
        print('Method      Position Error (m)      Bearing Error (deg)')
        for row in summaries:
            def fmt(key):
                mean, std = row[key], row[key+'_std']
                return 'N/A' if mean is None else f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f} ± N/A"
            print(f"{row['method']:<12}{fmt('position_error_m'):<24}{fmt('bearing_error_deg')}")
        print(f'Finished. Results: {output}', flush=True)
        return output


def main(experiment):
    parser = argparse.ArgumentParser(description=f'Paired LiDAR ablation {experiment}; static scene required.')
    defaults = Config()
    for name, value in asdict(defaults).items():
        help_text = ('Experiment 1 only; experiment 2 flies the entire fixed path'
                     if name in ('duration', 'speed_period', 'turn_period', 'heading_amplitude') else None)
        parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value, help=help_text)
    cfg = Config(**vars(parser.parse_args()))
    cfg.validate()
    # Fail before flight if requested Excel logging cannot be provided.
    import openpyxl  # noqa: F401
    Experiment(cfg, experiment).execute()
