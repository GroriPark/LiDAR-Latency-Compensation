"""Evaluate a shared MAPPO actor in make_env_lidar_common (requires running AirSim)."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor', type=Path, required=True, help='Trained actor.pt')
    p.add_argument('--mode', choices=['raw', 'corrected'], default='raw')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--success-radius', type=float, default=1.5)
    p.add_argument('--correction-latency', type=float, default=0.1)
    p.add_argument('--dt', type=float, default=0.15)
    p.add_argument('--ip', default='127.0.0.1')
    p.add_argument('--port', type=int, default=41451)
    p.add_argument('--hidden-size', type=int, default=64)
    p.add_argument('--recurrent-n', type=int, default=1)
    p.add_argument('--layer-n', type=int, default=1)
    p.add_argument('--no-recurrent', action='store_true')
    p.add_argument('--no-feature-normalization', action='store_true')
    p.add_argument('--tanh', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--output', type=Path, default=Path('lidar_common_eval'))
    a = p.parse_args(argv)
    for key in ('episodes', 'max_steps', 'success_radius', 'dt', 'hidden_size', 'recurrent_n'):
        if getattr(a, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if a.correction_latency < 0 or a.layer_n < 0:
        p.error('correction latency and layer count must be nonnegative')
    if not a.actor.is_file():
        p.error(f'Actor checkpoint does not exist: {a.actor}')
    return a


def evaluate(env, actor, cfg, torch, np):
    rows = []
    n = len(env.possible_agents)
    for episode in range(cfg.episodes):
        obs = env.reset(seed=cfg.seed + episode)
        states = torch.zeros((n, cfg.recurrent_n, cfg.hidden_size), device=cfg.device)
        masks = torch.ones((n, 1), device=cfg.device)
        previous = None
        control_sum = change_sum = episode_return = 0.0
        info = {}
        status = 'TIMEOUT'
        for step in range(1, cfg.max_steps + 1):
            with torch.no_grad():
                actions, _, states = actor(np.asarray(obs, dtype=np.float32), states,
                                           masks, deterministic=True)
            actions = actions.detach().cpu().numpy()
            if actions.shape != (n, 2) or not np.isfinite(actions).all():
                raise ValueError('Actor must produce finite actions of shape (agents, 2)')
            commands = np.clip(actions, -1, 1).astype(np.float64) * env.MAX_SPEED
            speeds = np.linalg.norm(commands, axis=1)
            limited = speeds > env.MAX_SPEED
            commands[limited] *= (env.MAX_SPEED / (speeds[limited] + 1e-6))[:, None]
            control_sum += float(np.linalg.norm(commands, axis=1).mean())
            if previous is not None:
                change_sum += float(np.linalg.norm(commands - previous, axis=1).mean())
            previous = commands.copy()
            obs, rewards, dones, infos = env.step(actions)
            episode_return += float(np.asarray(rewards).mean())
            if np.any(dones):
                info = infos[0][env.possible_agents[0]]
                status = info['final_status']
                break
            masks = torch.as_tensor(1 - np.asarray(dones, dtype=np.float32)[:, None],
                                    device=cfg.device)
        row = dict(episode=episode + 1, seed=cfg.seed + episode,
                   final_status=status, steps=step, episode_return=episode_return,
                   success=float(info.get('episode_success', 0)),
                   ally_collision=float(info.get('episode_ally_collision', 0)),
                   leader_hit=float(info.get('episode_leader_hit', 0)),
                   far_cutoff=float('FAR_CUTOFF' in status), timeout=float(status == 'TIMEOUT'),
                   hit_radius=env.success_radius,
                   mean_control_mps=control_sum / step,
                   summed_control_mps=control_sum * n,
                   mean_control_change_mps=change_sum / max(step - 1, 1))
        rows.append(row)
        # Save each completed episode so interruption does not lose previous results.
        cfg.output.mkdir(parents=True, exist_ok=True)
        with (cfg.output / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f'[{episode + 1}/{cfg.episodes}] {status} steps={step}', flush=True)
    return rows


def run(cfg):
    import numpy as np
    import torch
    from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
    from onpolicy.envs.eval_env_lidar_common import AirSimMultiDroneEvalEnv

    torch.manual_seed(cfg.seed)
    args = SimpleNamespace(hidden_size=cfg.hidden_size, gain=0.01, use_orthogonal=True,
        use_policy_active_masks=False, use_naive_recurrent_policy=False,
        use_recurrent_policy=not cfg.no_recurrent, recurrent_N=cfg.recurrent_n,
        algorithm_name='rmappo', use_popart=False,
        use_feature_normalization=not cfg.no_feature_normalization,
        stacked_frames=1, use_ReLU=not cfg.tanh, layer_N=cfg.layer_n)
    # Read and validate the checkpoint before connecting to the simulator.
    checkpoint = torch.load(cfg.actor, map_location=cfg.device, weights_only=True)
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    env = AirSimMultiDroneEvalEnv(ip_address=cfg.ip, port=cfg.port, dt=cfg.dt,
        do_visualize=cfg.visualize, eval_success_radius=cfg.success_radius,
        enable_lidar_correction=cfg.mode == 'corrected',
        correction_latency=cfg.correction_latency)
    try:
        actor = R_Actor(args, env.observation_space[0], env.action_space[0],
                        device=torch.device(cfg.device))
        actor.load_state_dict(state, strict=True)
        actor.eval()
        rows = evaluate(env, actor, cfg, torch, np)
        summary = {key + '_rate_percent': 100 * float(np.mean([r[key] for r in rows]))
                   for key in ('success', 'ally_collision', 'leader_hit', 'far_cutoff', 'timeout')}
        for key in ('steps', 'episode_return', 'mean_control_mps',
                    'summed_control_mps', 'mean_control_change_mps'):
            summary['mean_' + key] = float(np.mean([r[key] for r in rows]))
        payload = dict(config={k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(cfg).items()}, summary=summary, episodes=rows)
        (cfg.output / 'results.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        return payload
    finally:
        # Stop all commanded vehicles, including when inference or RPC fails.
        for name in [*env.possible_agents, env.leader_name, env.dynamic_name]:
            try:
                env.client.moveByVelocityAsync(0, 0, 0, duration=env.dt,
                                                vehicle_name=name).join()
            except Exception as exc:
                print(f'Could not stop {name}: {exc}')


if __name__ == '__main__':
    run(parse_args())
