#!/usr/bin/env python
import os
import sys
from pathlib import Path
import numpy as np
import torch
import wandb

try:
    import setproctitle
    _HAS_SETPROC = True
except Exception:
    _HAS_SETPROC = False

import gym

# --- 프로젝트 파일 임포트 ---
from onpolicy.config_UAV import get_config
from onpolicy.envs.make_env_baseline import AirSimMultiDroneEnv
from onpolicy.envs.env_wrappers import DummyVecEnv


class AirSimRunnerGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_env: AirSimMultiDroneEnv, use_wandb=False):
        super().__init__()
        self.base = base_env
        self.agents = self.base.possible_agents
        self.num_agents = len(self.agents)
        self.use_wandb = bool(use_wandb)

        self.observation_space = self.base.observation_space
        self.share_observation_space = self.base.share_observation_space
        self.action_space = self.base.action_space

        self.single_observation_space = self.base.observation_spaces[self.agents[0]]
        self.single_action_space = self.base.action_spaces[self.agents[0]]

        # episode 누적 통계
        self._ep_total = 0
        self._cnt_success = 0
        self._cnt_leader_hit = 0
        self._cnt_agent_collision = 0
        self._cnt_far_cutoff = 0

        # reward 통계
        self._global_env_step = 0
        self._episode_reward_sum = 0.0
        self._episode_step_count = 0

        self._last_final_status = None

    def seed(self, seed=None):
        if seed is not None:
            self.base.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.base.seed(seed)

        self._episode_reward_sum = 0.0
        self._episode_step_count = 0

        obs_list = self.base.reset(seed=seed, options=options)
        obs = np.asarray(obs_list, dtype=np.float32)
        return obs

    def _safe_wandb_log(self, payload):
        if self.use_wandb and wandb.run is not None:
            wandb.log(payload, step=self._global_env_step)

    @staticmethod
    def _extract_final_status(infos_list, agents):
        if not isinstance(infos_list, (list, tuple)) or len(infos_list) == 0:
            return None

        # top-level final_status 먼저
        for item in infos_list:
            if isinstance(item, dict) and "final_status" in item:
                return item["final_status"]

        # agent별 nested dict fallback
        for i, item in enumerate(infos_list):
            if not isinstance(item, dict):
                continue
            a = agents[i] if i < len(agents) else None
            if a is not None and a in item and isinstance(item[a], dict):
                if "final_status" in item[a]:
                    return item[a]["final_status"]

        return None

    @staticmethod
    def _classify_status(status: str):
        flags = dict(success=0, leader_hit=0, agent_collision=0, far_cutoff=0)

        if not isinstance(status, str):
            return flags

        if status.startswith("SUCCESS"):
            flags["success"] = 1

        if "LEADER_AND_DYNAMIC" in status:
            flags["leader_hit"] = 1

        if ("AGENT_LEADER" in status) or ("AGENT_AGENT" in status) or ("DISTANCE_AGENT_LEADER" in status) or ("DISTANCE_AGENT_AGENT" in status):
            flags["agent_collision"] = 1

        if "FAR_CUTOFF" in status:
            flags["far_cutoff"] = 1

        return flags

    def _rates(self):
        denom = max(self._ep_total, 1)
        return dict(
            intercept_success_rate=self._cnt_success / denom,
            leader_hit_rate=self._cnt_leader_hit / denom,
            agent_collision_rate=self._cnt_agent_collision / denom,
            far_cutoff_rate=self._cnt_far_cutoff / denom,
            episodes_count=self._ep_total,
        )

    def step(self, actions):
        act_list = actions.tolist() if isinstance(actions, np.ndarray) else actions
        obs_list, rewards_list, dones_list, infos_list = self.base.step(act_list)

        obs = np.asarray(obs_list, dtype=np.float32)
        rewards = np.asarray(rewards_list, dtype=np.float32)[:, None]
        dones = np.asarray(dones_list, dtype=np.bool_)[:, None]

        per_agent = {
            self.agents[i]: (infos_list[i] if isinstance(infos_list[i], dict) else {})
            for i in range(self.num_agents)
        }

        info = {}
        info.update(per_agent)

        # -------------------------------------------------
        # step_average_reward
        # -------------------------------------------------
        self._global_env_step += 1
        step_avg_reward = float(np.mean(rewards))

        self._episode_reward_sum += step_avg_reward
        self._episode_step_count += 1

        info["step_average_reward"] = step_avg_reward

        self._safe_wandb_log({
            "global_env_step": self._global_env_step,
            "step_average_reward": step_avg_reward,
        })

        # -------------------------------------------------
        # episode 종료 시 통계 로깅
        # -------------------------------------------------
        if bool(dones.any()):
            status = self._extract_final_status(infos_list, self.agents)
            self._last_final_status = status if status is not None else "UNKNOWN"

            flags = self._classify_status(self._last_final_status)
            self._ep_total += 1
            self._cnt_success += int(flags["success"])
            self._cnt_leader_hit += int(flags["leader_hit"])
            self._cnt_agent_collision += int(flags["agent_collision"])
            self._cnt_far_cutoff += int(flags["far_cutoff"])

            rates = self._rates()

            # 에피소드 전체 평균 보상
            episode_average_reward = float(
                self._episode_reward_sum / max(self._episode_step_count, 1)
            )

            info["final_status"] = self._last_final_status
            info["episode_average_reward"] = episode_average_reward
            info["intercept_success_rate"] = rates["intercept_success_rate"]
            info["leader_hit_rate"] = rates["leader_hit_rate"]
            info["agent_collision_rate"] = rates["agent_collision_rate"]
            info["far_cutoff_rate"] = rates["far_cutoff_rate"]
            info["episodes_count"] = rates["episodes_count"]
            info["episode_length"] = self._episode_step_count

            # env info에서 hit radius / total_successes 있으면 같이 넘김
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0 and isinstance(infos_list[0], dict):
                first_agent = self.agents[0]
                if first_agent in infos_list[0] and isinstance(infos_list[0][first_agent], dict):
                    nested = infos_list[0][first_agent]

                    if "current_hit_radius" in nested:
                        info["current_hit_radius"] = nested["current_hit_radius"]

                    if "total_successes" in nested:
                        info["total_successes"] = nested["total_successes"]

            wandb_payload = {
                "global_env_step": self._global_env_step,
                "episode_average_reward": episode_average_reward,
                "intercept_success_rate": rates["intercept_success_rate"],
                "leader_hit_rate": rates["leader_hit_rate"],
                "agent_collision_rate": rates["agent_collision_rate"],
                "far_cutoff_rate": rates["far_cutoff_rate"],
                "episodes_count": rates["episodes_count"],
                "episode_length": self._episode_step_count,
                "episode_success": int(flags["success"]),
            }

            if "current_hit_radius" in info:
                wandb_payload["current_hit_radius"] = info["current_hit_radius"]
            if "total_successes" in info:
                wandb_payload["total_successes"] = info["total_successes"]

            self._safe_wandb_log(wandb_payload)

            print(
                f"[EP_END] status={self._last_final_status} | "
                f"episode_average_reward={episode_average_reward:.4f} | "
                f"step_average_reward(last)={step_avg_reward:.4f} | "
                f"intercept_success_rate={rates['intercept_success_rate']:.3f} | "
                f"leader_hit_rate={rates['leader_hit_rate']:.3f} | "
                f"agent_collision_rate={rates['agent_collision_rate']:.3f} | "
                f"far_cutoff_rate={rates['far_cutoff_rate']:.3f} | "
                f"episodes={rates['episodes_count']}"
            )

        return obs, rewards, dones, info

    def close(self):
        return


def _build_single_env(all_args, rank_seed):
    ip_address = getattr(all_args, "ip_address", "127.0.0.1")
    num_agents = int(getattr(all_args, "num_agents", 3))
    port = int(getattr(all_args, "port", 41451))
    dt = float(getattr(all_args, "dt", 0.15))
    fixed_z = float(getattr(all_args, "fixed_z", -10.0))
    do_visualize = bool(getattr(all_args, "do_visualize", False))
    dynamic_name = getattr(all_args, "dynamic_name", "DynamicObstacle")
    leader_name = getattr(all_args, "leader_name", "Drone1")
    use_wandb = bool(getattr(all_args, "use_wandb", False))

    base_env = AirSimMultiDroneEnv(
        ip_address=ip_address,
        follower_names=[f"Follower{i}" for i in range(num_agents)],
        port=port,
        dt=dt,
        fixed_z=fixed_z,
        do_visualize=do_visualize,
        dynamic_name=dynamic_name,
        leader_name=leader_name,
    )
    base_env.seed(rank_seed)
    return AirSimRunnerGymEnv(base_env, use_wandb=use_wandb)


def make_train_env(all_args):
    if getattr(all_args, "n_rollout_threads", 1) != 1:
        print("[WARN] AirSim 환경은 멀티프로세싱과 궁합이 안 좋아서 n_rollout_threads=1로 강제합니다.")
        all_args.n_rollout_threads = 1

    def get_env_fn(rank):
        def init_env():
            return _build_single_env(all_args, int(all_args.seed) + rank * 1000)
        return init_env

    return DummyVecEnv([get_env_fn(0)])


def make_eval_env(all_args):
    if not getattr(all_args, "use_eval", False):
        return None

    if getattr(all_args, "n_eval_rollout_threads", 1) != 1:
        print("[WARN] AirSim 평가 환경도 n_eval_rollout_threads=1로 강제합니다.")
        all_args.n_eval_rollout_threads = 1

    def get_env_fn(rank):
        def init_env():
            return _build_single_env(all_args, int(all_args.seed) * 50000 + rank * 10000)
        return init_env

    return DummyVecEnv([get_env_fn(0)])


def _infer_obs_shape(o: np.ndarray):
    if not isinstance(o, np.ndarray):
        o = np.asarray(o)

    if o.ndim == 2:
        n_envs = 1
        n_agents, obs_dim = o.shape
    elif o.ndim == 3:
        n_envs, n_agents, obs_dim = o.shape
    else:
        raise RuntimeError(f"Unexpected obs ndim={o.ndim}, shape={o.shape}")

    return int(n_envs), int(n_agents), int(obs_dim)


def _get_env_declared_obs_dim(envs: DummyVecEnv):
    try:
        inner = envs.envs[0]
        return int(inner.single_observation_space.shape[0])
    except Exception:
        try:
            space0 = envs.observation_space[0]
            if hasattr(space0, "shape"):
                return int(space0.shape[0])
        except Exception:
            pass
    raise RuntimeError("Failed to get declared obs_dim from env spaces.")


def main(args):
    parser = get_config()
    all_args = parser.parse_args(args)

    # 알고리즘 선택
    if all_args.algorithm_name == "rmappo":
        print("알고리즘: rmappo → use_recurrent_policy=True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("알고리즘: mappo → use_recurrent_policy=False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError("rmappo 또는 mappo만 지원합니다.")

    # CUDA 설정
    if getattr(all_args, "cuda", False) and torch.cuda.is_available():
        print("GPU를 사용합니다.")
        device = torch.device("cuda:0")
        torch.set_num_threads(getattr(all_args, "n_training_threads", 1))
        if getattr(all_args, "cuda_deterministic", False):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("CPU를 사용합니다.")
        device = torch.device("cpu")
        torch.set_num_threads(getattr(all_args, "n_training_threads", 1))

    # 결과 저장 경로
    base_dir = Path(getattr(
        all_args,
        "results_dir",
        r"C:\Users\pgr51\Desktop\Gardian\GARDIANS_DRONE_SWARM\on-policy\ES3_lidar_raw"
    ))
    run_dir = base_dir / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[CWD]", os.getcwd())
    print("[RUN_DIR]", run_dir.resolve())

    # WandB
    run = None
    if getattr(all_args, "use_wandb", False):
        run = wandb.init(
            config=vars(all_args),
            project=all_args.env_name,
            name=all_args.experiment_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True
        )

        # x축 통일
        wandb.define_metric("global_env_step")
        wandb.define_metric("step_average_reward", step_metric="global_env_step")
        wandb.define_metric("episode_average_reward", step_metric="global_env_step")
        wandb.define_metric("intercept_success_rate", step_metric="global_env_step")
        wandb.define_metric("leader_hit_rate", step_metric="global_env_step")
        wandb.define_metric("agent_collision_rate", step_metric="global_env_step")
        wandb.define_metric("far_cutoff_rate", step_metric="global_env_step")
        wandb.define_metric("episode_length", step_metric="global_env_step")
        wandb.define_metric("episodes_count", step_metric="global_env_step")
        wandb.define_metric("current_hit_radius", step_metric="global_env_step")
        wandb.define_metric("total_successes", step_metric="global_env_step")

    # 프로세스 이름
    if _HAS_SETPROC:
        try:
            setproctitle.setproctitle(
                "-".join([all_args.env_name, all_args.algorithm_name, all_args.experiment_name])
            )
        except Exception:
            pass

    # 시드
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Env 초기화
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args)

    # sanity check
    try:
        o = envs.reset()
        o = np.asarray(o)
        n_envs, n_agents, obs_dim = _infer_obs_shape(o)
        declared_obs_dim = _get_env_declared_obs_dim(envs)
        expected_agents = int(getattr(all_args, "num_agents", n_agents))

        print(f"[SANITY] obs shape = {o.shape}")
        print(f"[SANITY] n_envs={n_envs}, n_agents={n_agents}, obs_dim={obs_dim}, declared_obs_dim={declared_obs_dim}")

        assert n_agents == expected_agents, (
            f"Agent count mismatch: got {n_agents}, args {expected_agents}"
        )
        assert obs_dim == declared_obs_dim, (
            f"obs_dim mismatch: got {obs_dim}, env declares {declared_obs_dim}"
        )
        print("[SANITY] ✅ obs_dim / num_agents match. Good to go!")
    except Exception as e:
        print(f"[SANITY] ❌ reset/shape check failed: {e}")
        raise

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    from onpolicy.runner.shared.uav_runner import Runner

    try:
        runner = Runner(config)
        runner.run()

        if hasattr(runner, "save"):
            print("[FORCE SAVE] runner.save()")
            runner.save()
        else:
            print("[WARN] Runner has no save() method")

    finally:
        try:
            envs.close()
        except Exception:
            pass

        if getattr(all_args, "use_eval", False) and (eval_envs is not None) and (eval_envs is not envs):
            try:
                eval_envs.close()
            except Exception:
                pass

        if run is not None:
            try:
                run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main(sys.argv[1:])