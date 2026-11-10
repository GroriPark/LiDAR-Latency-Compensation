# MultiUAV

AirSim에서 여러 UAV가 협력해 리더 드론을 보호하고 접근하는 동적 장애물(적기)을 요격하도록 학습하는 다중 에이전트 강화학습 프로젝트입니다. MAPPO / RMAPPO 기반 정책과 LiDAR 관측을 사용하며, 원본 LiDAR와 이동·시간 오차를 보정한 LiDAR 환경을 비교하는 실험 코드를 포함합니다.

> 현재 저장소는 실험 코드 모음입니다. Unreal/AirSim 시뮬레이션 프로젝트, AirSim `settings.json`, 학습 체크포인트는 포함되어 있지 않습니다. 일부 진입점의 import와 저장 경로를 수정한 뒤 실행해야 합니다. 아래 안내는 소스 확인을 기준으로 작성했으며 실제 시뮬레이터 실행은 검증하지 않았습니다.

## Relationship to the original MAPPO implementation

This project builds on the publicly released **MAPPO/on-policy** implementation associated with:

Chao Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* ([paper](https://arxiv.org/abs/2103.01955); [original repository](https://github.com/marlbenchmark/on-policy)).

이 프로젝트의 기반이 되는 MAPPO/on-policy 원본 코드는 **Chao Yu 등 해당 논문의 저자들과 MAPPO 기여자들이 작성한 코드**입니다. 본 저장소는 해당 공개 구현을 바탕으로 AirSim 다중 UAV 환경, LiDAR 관측 및 보정, UAV 학습·평가 코드를 추가·수정한 프로젝트입니다. 원본 MAPPO 구현에 대한 저작권과 기여는 원저자 및 기여자에게 있으며, 원본의 MIT 라이선스와 저작권 표기를 유지합니다([LICENSE](on-policy/LICENSE)).

## 주요 기능

- 기본 3대의 팔로워가 공유 정책으로 협력하는 연속 제어
- MAPPO 및 순환 정책을 사용하는 RMAPPO 학습
- 원본 / 이동 보정 LiDAR 환경의 공통 구현과 실험별 환경 변형
- 요격 성공, 리더 피격, 아군 충돌, 거리 이탈 통계 기록
- 학습 체크포인트 저장, TensorBoard 및 Weights & Biases 로깅
- 학습된 Actor를 이용한 기본 100 에피소드 평가 및 JSON 결과 저장

## 디렉터리 구조

```text
MultiUAV/
├── README.md
└── on-policy/
    ├── train_UAV.py                   # 일반 학습 진입점 (환경 import 수정 필요)
    ├── train_UAV_LiDAR_Raw.py         # 원본 LiDAR 학습
    ├── eval_LiDAR_Raw.py              # 원본 LiDAR 평가 (환경 모듈 누락)
    ├── eval_time_correction.py        # 시간 오차 보정 평가
    ├── requirements.txt              # 기존 프레임워크의 의존성 목록
    ├── environment.yaml              # 기존 Linux / Python 3.6 환경 명세
    ├── setup.py
    ├── LICENSE
    └── onpolicy/
        ├── config_UAV.py             # UAV 학습 인자
        ├── algorithms/               # RMAPPO, HAPPO, HATRPO, MAT 구현
        ├── envs/                     # AirSim 학습·평가 환경
        ├── runner/shared/uav_runner.py
        └── utils/shared_buffer_uav.py
```

UAV 학습 진입점은 `rmappo`, `mappo`만 지원합니다. 다른 알고리즘과 벤치마크용 스크립트도 포함되어 있지만, UAV 실행 경로에서 지원하는 것은 아닙니다.

## 환경 구성

별도의 AirSim 멀티로터 시뮬레이터를 실행해야 합니다. 기본 연결 및 객체 이름은 다음과 같습니다.

| 항목 | 기본값 |
| --- | --- |
| 서버 | `127.0.0.1:41451` |
| 리더 | `Drone1` |
| 팔로워 | `Follower0`, `Follower1`, `Follower2` |
| 동적 장애물 | `DynamicObstacle` |
| 팔로워별 LiDAR | `Follower0_LidarSensor`, `Follower1_LidarSensor`, `Follower2_LidarSensor` |
| 학습 환경 제어 간격 | `dt=0.15`초 |
| 고정 Z 좌표 | `-10.0` (AirSim NED 좌표계) |

시뮬레이터의 차량·객체 이름과 LiDAR 센서 이름을 코드와 맞춰야 합니다. 센서 설정, 좌표계, 장착 위치도 함께 확인하세요. 공통 학습 환경의 기본 `lidar_offset`은 `(0.0, 0.0, 0.3)`입니다.

### Python 의존성

주요 패키지는 `torch`, `numpy`, `gym`, `gymnasium`, `airsim`, `tensorboardX`, `wandb`이며, 시간 보정 평가 환경에서는 `scikit-learn`도 사용합니다. `setproctitle`은 학습 진입점에서 선택적으로 사용합니다.

`requirements.txt`와 `environment.yaml`은 오래된 기반 프레임워크 환경을 담고 있습니다. 현재 UAV 코드에서 사용하는 AirSim·Gymnasium 등의 의존성이 모두 명시되어 있지 않으며, Windows에서 그대로 재현되는 설치 명세는 아닙니다. `setup.py`도 실행 의존성을 자동 설치하지 않습니다. 별도의 가상환경에서 시뮬레이터 및 PyTorch 환경과 호환되는 버전을 구성해야 합니다.

아래는 필수 import 확인용 명령입니다. 실제 학습 호환성 검증을 대신하지 않습니다.

```powershell
cd on-policy
python -c "import torch, numpy, gym, gymnasium, airsim, tensorboardX, wandb; from sklearn.cluster import DBSCAN; print('imports OK')"
```

`on-policy/`에서 스크립트를 직접 실행하면 로컬 `onpolicy` 패키지를 불러올 수 있습니다. 편집 가능 설치가 필요하다면 다음 절의 README 관련 사항을 먼저 확인하세요.

## 실행 전 확인 사항

1. **일반 학습 환경 import**: `train_UAV.py`는 존재하지 않는 `onpolicy.envs.make_env`를 참조합니다. 보정 환경을 사용하려는 경우 `onpolicy.envs.make_env_fin`의 `AirSimMultiDroneEnv`를 선택하도록 import를 수정할 수 있습니다. 원래 누락된 환경과 동일한 구현인지는 확인이 필요합니다.
2. **원본 평가 환경 누락**: `eval_LiDAR_Raw.py`가 참조하는 `onpolicy.envs.make_env_no_time_correction`이 없습니다. 해당 모듈을 복원하거나, 체크포인트와 관측 구성이 일치하는 평가 환경을 명시적으로 선택해야 합니다.
3. **학습 결과 경로**: 두 학습 스크립트의 `main()`에서 `base_dir` 기본값이 작성자의 PC 절대 경로입니다. 이를 `Path("results")` 등으로 수정하세요. `config_UAV.py`에는 `--results_dir` 인자가 등록되어 있지 않아 현재 CLI로 지정할 수 없습니다.
4. **체크포인트 경로**: 평가 스크립트 상단의 `actor_model_path`를 자신의 `actor.pt` 경로로 변경하세요. 모델의 관측 차원, 은닉 크기, RNN 설정을 학습 당시와 맞춰야 합니다.
5. **패키지 설치 위치**: `setup.py`는 `on-policy/README.md`를 읽습니다. 해당 위치에 안내 문서를 포함했으므로 편집 가능 설치는 `on-policy/`에서 `python -m pip install -e .`로 진행할 수 있습니다. 실행 의존성은 별도로 준비해야 합니다.

환경 파일마다 관측 차원과 동작 조건이 다를 수 있습니다. 특히 `eval_env_control_raw12.py` 등 별도 실험 환경을 기존 모델에 연결할 때는 파일 이름만으로 호환성을 판단하지 마세요.

## 학습

시뮬레이터를 먼저 실행하고 위의 결과 경로를 수정한 후, `on-policy/`에서 실행합니다.

```powershell
# 원본 LiDAR + RMAPPO, 로컬 TensorBoard 로그 사용
python train_UAV_LiDAR_Raw.py --algorithm_name rmappo --experiment_name raw_lidar --num_agents 3 --num_env_steps 20000000 --use_wandb

# 원본 LiDAR + MAPPO
python train_UAV_LiDAR_Raw.py --algorithm_name mappo --experiment_name raw_mappo --use_wandb
```

보정 학습은 `train_UAV.py`의 환경 import를 먼저 수정한 뒤 실행합니다.

```powershell
python train_UAV.py --algorithm_name rmappo --experiment_name corrected_lidar --use_wandb
```

**인자 주의:** 현재 파서는 `--use_wandb`와 `--cuda`를 `store_false`로 정의합니다. 따라서 W&B는 기본 활성화이며 `--use_wandb`를 넣으면 비활성화됩니다. GPU도 기본 활성화이며 `--cuda`를 넣으면 CPU를 사용합니다. GPU가 없을 때는 CPU로 전환합니다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--algorithm_name` | `rmappo` | `rmappo` 또는 `mappo` |
| `--experiment_name` | `drone_formation_v1` | 실험 이름 |
| `--seed` | `1` | 난수 시드 |
| `--num_agents` | `3` | 팔로워 수, 시뮬레이터 구성과 일치 필요 |
| `--ip_address` | `127.0.0.1` | AirSim 서버 IP |
| `--num_env_steps` | `20000000` | 총 학습 스텝 |
| `--rollout_length` | `2048` | 정책 업데이트당 수집 스텝 |
| `--episode_length` | `2048` | 버퍼 관련 길이 설정 |
| `--hidden_size` | `64` | 정책 은닉 크기 |
| `--lr` / `--critic_lr` | `3e-4` / `5e-4` | Actor / Critic 학습률 |
| `--model_dir` | `None` | 기존 모델 디렉터리 |

AirSim 학습 환경의 rollout 스레드는 코드에서 1개로 강제합니다. `dt`, `port`, `fixed_z`, `leader_name`, `dynamic_name` 등은 현재 CLI 파서에 등록되어 있지 않으므로 진입점 또는 환경 생성 코드를 수정해야 합니다. `step_length`, `leader_velocity`, `optimal_distance`는 파서에 존재하지만 현재 학습 진입점에서 환경 생성자로 전달하지 않습니다.

## LiDAR 실험 구성

`make_env_baseline.py`와 `make_env_fin.py`는 `make_env_lidar_common.py`를 공유하며 `ENABLE_LIDAR_CORRECTION`만 각각 `False` / `True`로 설정합니다.

공통 학습 환경은 기본 3대 기준 에이전트별 10차원 관측(리더 상대 XY, 다른 아군 상대 XY, 적 상대 XY, 자신의 XY 속도)과 2차원 연속 행동을 정의합니다. Critic용 공유 관측은 30차원입니다. 행동은 XY 속도 명령에 사용됩니다.

공통 학습 환경의 희소 보상은 요격 성공 `+1`, 리더 피격·아군 충돌·거리 이탈 `-1`입니다. 요격 인정 반경은 15m에서 시작하며, 같은 단계에서 20회 연속 성공할 때마다 0.1m씩 줄어 최소 1.5m까지 조정됩니다. 별도 평가 환경에는 다른 반경이나 관측 구성이 적용될 수 있습니다.

## 평가 및 결과

평가 환경 import와 `actor_model_path`를 준비한 후 실행합니다.

```powershell
# 체크포인트 및 시간 보정 평가 환경 설정 확인 후
python eval_time_correction.py

# 누락된 원본 평가 환경을 복원하거나 호환 환경을 선택한 후
python eval_LiDAR_Raw.py
```

두 스크립트는 기본 100 에피소드를 deterministic 정책으로 평가하며 W&B를 초기화합니다. 평가 횟수와 환경 설정은 스크립트 내부에서 변경합니다. 방어 성공률(DSR), 아군 충돌률(CR), 리더 피격률(LHR), 거리 이탈 등의 통계를 출력하고 현재 작업 디렉터리에 `lidar_eval_results.json`을 저장합니다. 두 평가가 같은 파일명을 사용하므로 비교 실험 시 결과를 별도로 보관하세요.

학습 출력은 수정한 `base_dir` 아래의 `<env_name>/<algorithm_name>/<experiment_name>/`에 저장됩니다.

- W&B 활성화: 해당 W&B run 디렉터리에 `actor.pt`, `critic.pt` 저장
- W&B 비활성화: `models/`에 체크포인트, `logs/`에 TensorBoard 로그 저장

실험 비교 시 체크포인트뿐 아니라 환경 파일, 관측 차원, 센서 설정, 보정 여부, 시드와 평가 조건도 함께 기록하세요. 저장소에는 성능 수치나 재현을 완료한 실험 결과가 포함되어 있지 않습니다.

## 라이선스

포함된 on-policy 프레임워크의 라이선스는 [MIT License](on-policy/LICENSE)입니다. 기반 패키지 정보와 저작권 표기는 `on-policy/setup.py` 및 `on-policy/LICENSE`에서 확인할 수 있습니다.

