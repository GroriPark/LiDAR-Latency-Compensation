# MultiUAV on-policy

AirSim 기반 다중 UAV 협력 제어를 위한 MAPPO / RMAPPO 실험 코드입니다.

기반 MAPPO/on-policy 원본 코드는 Chao Yu 등 *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*의 저자들과 MAPPO 기여자들이 작성했습니다([논문](https://arxiv.org/abs/2103.01955), [원본 저장소](https://github.com/marlbenchmark/on-policy)). 본 프로젝트는 해당 공개 코드를 기반으로 AirSim UAV 실험 기능을 추가·수정했으며, 원본의 MIT 라이선스와 저작권 표기를 유지합니다.

프로젝트 소개, 환경 구성, 학습·평가 명령 및 실행 전 수정 사항은 상위 폴더의 [프로젝트 README](../README.md)를 참고하세요.

주요 진입점:

- `train_UAV_LiDAR_Raw.py`: 원본 LiDAR 학습
- `train_UAV.py`: 일반 학습 (현재 환경 import 수정 필요)
- `eval_time_correction.py`: 시간 오차 보정 평가
- `eval_LiDAR_Raw.py`: 원본 LiDAR 평가 (현재 참조 환경 모듈 누락)

학습 실행 위치는 이 디렉터리입니다. Python 의존성과 별도의 AirSim 시뮬레이터를 먼저 준비해야 합니다.

라이선스는 [LICENSE](LICENSE)를 참고하세요.
