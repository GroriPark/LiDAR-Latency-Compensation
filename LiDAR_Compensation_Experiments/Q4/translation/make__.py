"""make.py의 계산식을 유지하고 100회 반복 및 CSV 저장만 추가한 별도 파일.

python make__.py
아래 상수로 회차/시간/속도를 조정한다. TRIAL_SECONDS는 실제 경과 시간이다.
현재 make.py의 속도 2m/s와 Cube_2 비교식을 그대로 사용한다.
LiDAR 설정/좌표계/보정식/객체 선택은 기존 make.py와 동일한 전제이다.
"""
import csv
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import airsim
import numpy as np

TRIAL_COUNT = 100
TRIAL_SECONDS = 10.0  # time.monotonic 기준: AirSim ClockSpeed=1에서 사용
SPEED = 25.0          
ALTITUDE = -20.0
LATENCY = 0.3
SCAN_TIME = 0.1
HISTORY_SAMPLES = 10  # N: 관측 이전 운동 샘플 수
POLY_DEGREE = 2       # m: 다항식 차수
HISTORY_INTERVAL = 0.02  # 초기 이력 수집/이력 부족 시 대기 간격(초)
VEHICLE_NAME = 'Follower0'
LIDAR_NAME = 'Follower0_LidarSensor'
OBJECT_NAME = 'Cylinder_2'
OUTPUT_ROOT = Path(__file__).resolve().parent / 'experiment_results' / 'make__'

ERROR_FIELDS = ['raw_x_mae_m', 'raw_y_mae_m', 'corrected_x_mae_m', 'corrected_y_mae_m']


class LidarTest:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def start_trial(self, vehicle_name):
        # 매 회차 동일 시작 상태로 초기화한 뒤 기존 코드와 같은 순서로 비행한다.
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        self.client.takeoffAsync(vehicle_name=vehicle_name).join()
        self.client.moveToZAsync(ALTITUDE, 1.0, vehicle_name=vehicle_name).join()
        self.move(vehicle_name)

    def move(self, vehicle_name):
        self.client.moveByVelocityZAsync(
            SPEED, 0, ALTITUDE, max(3600, TRIAL_SECONDS + 10),
            airsim.DrivetrainType.MaxDegreeOfFreedom, vehicle_name=vehicle_name)

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
        sums = np.zeros(4)
        started = None
        # 회차마다 새 이력. 최신 상태가 LiDAR보다 늦을 수 있어 N보다 여유 있게 보관한다.
        motion_history = deque(maxlen=HISTORY_SAMPLES * 2)
        result.update(trial=trial, status='running', error='')
        try:
            print(f"Trial {trial}/{TRIAL_COUNT}: reset / takeoff", flush=True)
            self.start_trial(vehicle_name)
            started = time.monotonic()
            print(f"Trial {trial}/{TRIAL_COUNT}: measuring for {TRIAL_SECONDS}s", flush=True)
            while time.monotonic() - started < TRIAL_SECONDS:
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
                object_pose = self.client.simGetObjectPose(OBJECT_NAME) 
                relx = object_pose.position.x_val - pose.position.x_val 
                rely = object_pose.position.y_val - pose.position.y_val 
                
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
                    raise ValueError("Nonfinite position/error; check LiDAR data and Cube_2 pose")
                count += 1
                elapsed = time.monotonic() - started
                sums += np.abs([noterrx, noterry, errx, erry])
                scan_writer.writerow({
                    'trial': trial, 'scan': count, 'elapsed_s': elapsed,
                    'lidar_timestamp': lidar_data.time_stamp, 'points': int(size),
                    'vx': vx, 'vy': vy, 'wz': wz,
                    'raw_x': notfinx, 'raw_y': notfiny,
                    'corrected_x': finx, 'corrected_y': finy,
                    'true_x': relx, 'true_y': rely,
                    'raw_error_x': noterrx, 'raw_error_y': noterry,
                    'corrected_error_x': errx, 'corrected_error_y': erry})
                scan_file.flush()
                print(f"Trial {trial}/{TRIAL_COUNT}, scan {count}, {elapsed:.1f}s, "
                      f"vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}", flush=True)
            result['status'] = 'completed' if count else 'no_data'
        except KeyboardInterrupt:
            result.update(status='interrupted', error='KeyboardInterrupt')
            raise  # 원래 traceback을 유지하고 반복 실험을 중단한다.
        except Exception as exc:
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise  # 오류를 한 줄 메시지로 숨기거나 다음 회차로 넘어가지 않는다.
        finally:
            result.update(scans=count, empty_scans=empty_scans,
                          elapsed_s=time.monotonic()-started if started is not None else 0)
            result.update(dict(zip(ERROR_FIELDS, (sums/count).tolist() if count else [None]*4)))

    def execute(self, vehicle_name=VEHICLE_NAME, lidar_name=LIDAR_NAME):
        if TRIAL_COUNT < 1 or TRIAL_SECONDS <= 0:
            raise ValueError('TRIAL_COUNT and TRIAL_SECONDS must be positive')
        if not 0 <= POLY_DEGREE < HISTORY_SAMPLES:
            raise ValueError('Require 0 <= POLY_DEGREE < HISTORY_SAMPLES')
        output = OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        print(f'Results: {output}', flush=True)
        scan_fields = ['trial', 'scan', 'elapsed_s', 'lidar_timestamp', 'points',
                       'vx', 'vy', 'wz', 'raw_x', 'raw_y', 'corrected_x', 'corrected_y',
                       'true_x', 'true_y', 'raw_error_x', 'raw_error_y',
                       'corrected_error_x', 'corrected_error_y']
        trial_fields = ['trial', 'status', 'error', 'scans', 'empty_scans', 'elapsed_s', *ERROR_FIELDS]
        completed = []
        attempted = 0
        # 실행 설정도 CSV로 저장한다.
        with (output / 'settings.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'value'])
            writer.writerows([
                ['trial_count', TRIAL_COUNT], ['trial_seconds_wall_clock', TRIAL_SECONDS],
                ['speed_mps', SPEED], ['altitude_ned_m', ALTITUDE], ['latency_s', LATENCY],
                ['scan_time_s', SCAN_TIME], ['vehicle', vehicle_name], ['lidar', lidar_name],
                ['history_samples_N', HISTORY_SAMPLES], ['polynomial_degree_m', POLY_DEGREE],
                ['history_wait_interval_s', HISTORY_INTERVAL],
                ['object', OBJECT_NAME], ['metric', 'Original make.py centroid minus Cube_2 position'],
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
        finally:
            # 종료 때 진행 중인 비행 명령을 hover로 교체한다. 원래 예외는 가리지 않는다.
            try:
                self.client.hoverAsync(vehicle_name=vehicle_name)
            except Exception:
                traceback.print_exc()
        print(f'Completed trials: {len(completed)}/{TRIAL_COUNT}; results: {output}', flush=True)


if __name__ == '__main__':
    lidarTest = LidarTest()
    lidarTest.execute(VEHICLE_NAME, LIDAR_NAME)
