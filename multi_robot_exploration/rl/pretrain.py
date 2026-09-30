"""
Warm start for RL: behaviour-clone the heuristic (frontier_coordinator's cost)
into the PPO policy, and pre-train the value head on Monte-Carlo returns.

  ros2 run multi_robot_exploration rl_pretrain --num-envs 4 --decisions 3000
  ros2 run multi_robot_exploration rl_train --resume <run-dir>/bc_init.zip --num-envs 4

Why: PPO from a random policy needs millions of decisions for this kind of
goal-selection task (Active Neural SLAM trained on 72 threads); we get ~20k.
Starting from a clone of a good teacher and fine-tuning is the standard
remedy (teacher/student knowledge distillation for multi-robot DRL
exploration, e.g. MDPI Mathematics 13(1):173, 2025). A random critic is what
usually wrecks a cloned policy in the first PPO updates, hence the value
pre-training.

Steps:
  1. collect  run the Gazebo env with the heuristic in --num-envs parallel sims
              -> dataset.npz (obs, mask, action, reward, done, episode) and
                 heuristic_episodes.csv (also the heuristic's baseline numbers)
  2. clone    masked cross-entropy to the heuristic action + value regression
              -> bc_init.zip (same architecture/hyperparameters as rl_train)
Use --dataset to skip step 1 and re-run step 2 on an existing dataset.
"""
import argparse
import csv
import multiprocessing as mp
import os
import time

import numpy as np

GAMMA = 0.99


def _collect_worker(rank, n_decisions, max_episode_sim_s, goal_fix, log_dir, seed):
    from multi_robot_exploration.rl.features import heuristic_action_from_obs
    from multi_robot_exploration.rl.gazebo_env import GazeboExplorationEnv

    env = GazeboExplorationEnv(instance_id=rank, max_episode_sim_s=max_episode_sim_s,
                               log_dir=log_dir, goal_fix=goal_fix)
    rows = {k: [] for k in ('obs', 'mask', 'action', 'reward', 'done', 'episode')}
    episodes = []
    try:
        obs, _ = env.reset(seed=seed + rank)
        ep = 0
        while True:
            mask = env.action_masks()
            a = heuristic_action_from_obs(obs)
            next_obs, r, term, trunc, info = env.step(a)
            rows['obs'].append(obs)
            rows['mask'].append(mask.copy())
            rows['action'].append(a)
            rows['reward'].append(r)
            rows['done'].append(term or trunc)
            rows['episode'].append(rank * 100000 + ep)
            obs = next_obs
            if term or trunc:
                episodes.append({k: info[k] for k in ('world_seed', 'end_reason', 'explored_m2',
                                                      'sim_time_s', 'decisions', 'failed_goals')})
                print(f"[collect {rank}] seed={info['world_seed']} end={info['end_reason']} "
                      f"explored={info['explored_m2']:.1f} m2 in {info['sim_time_s']:.0f} s, "
                      f"{len(rows['action'])}/{n_decisions} decisions", flush=True)
                ep += 1
                if len(rows['action']) >= n_decisions:   # only stop on episode boundaries
                    break
                obs, _ = env.reset()
    finally:
        env.close()
    return {k: np.asarray(v) for k, v in rows.items()}, episodes


def collect(args):
    per_env = int(np.ceil(args.decisions / args.num_envs))
    log_dir = os.path.join(args.run_dir, 'sim_logs')
    jobs = [(args.first_instance + i, per_env, args.max_episode_sim_s, not args.no_goal_fix, log_dir, args.seed)
            for i in range(args.num_envs)]
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.num_envs) as pool:
        results = pool.starmap(_collect_worker, jobs)

    data = {k: np.concatenate([r[0][k] for r in results]) for k in results[0][0]}
    episodes = [e for r in results for e in r[1]]
    np.savez_compressed(os.path.join(args.run_dir, 'dataset.npz'), **data)
    with open(os.path.join(args.run_dir, 'heuristic_episodes.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(episodes[0].keys()))
        w.writeheader()
        w.writerows(episodes)
    ex = np.array([e['explored_m2'] for e in episodes])
    tt = np.array([e['sim_time_s'] for e in episodes])
    print(f'\nHeuristic baseline over {len(episodes)} episodes: explored {ex.mean():.1f} ± '
          f'{ex.std():.1f} m2, episode time {tt.mean():.0f} s', flush=True)
    return data


def discounted_returns(reward, done, episode):
    """Monte-Carlo returns per episode (truncated episodes are treated as ended)."""
    ret = np.zeros_like(reward, dtype=np.float32)
    g = 0.0
    for i in range(len(reward) - 1, -1, -1):
        if done[i] or (i + 1 < len(reward) and episode[i + 1] != episode[i]):
            g = 0.0
        g = reward[i] + GAMMA * g
        ret[i] = g
    return ret


def clone(data, args):
    import torch as th

    from multi_robot_exploration.rl.train import SpacesOnlyEnv, build_model

    obs = data['obs'].astype(np.float32)
    mask = data['mask'].astype(bool)
    act = data['action'].astype(np.int64)
    ret = discounted_returns(data['reward'], data['done'], data['episode'])

    # Hold out whole episodes (not random steps) so validation is honest
    eps = np.unique(data['episode'])
    rng = np.random.default_rng(args.seed)
    val_eps = set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist())
    is_val = np.array([e in val_eps for e in data['episode']])
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    if len(va) == 0:
        print(f'WARNING: only {len(eps)} episodes (< 10); validation uses the training data, '
              'so accuracy is optimistic. Collect more decisions for a real run.', flush=True)
        va = tr
    print(f'Cloning on {len(tr)} decisions, validating on {len(va)} '
          f'({len(val_eps)} held-out episodes)', flush=True)

    model = build_model(SpacesOnlyEnv(), n_steps=64, seed=args.seed)
    policy = model.policy
    policy.set_training_mode(True)
    opt = th.optim.Adam(policy.parameters(), lr=args.lr)
    T = lambda x: th.as_tensor(x, device=policy.device)  # noqa: E731

    def evaluate(idx):
        with th.no_grad():
            dist = policy.get_distribution(T(obs[idx]), action_masks=mask[idx])
            pred = dist.distribution.probs.argmax(dim=1).cpu().numpy()
            v = policy.predict_values(T(obs[idx])).squeeze(1).cpu().numpy()
        acc = float((pred == act[idx]).mean())
        r2 = 1.0 - float(((v - ret[idx]) ** 2).sum() / max(((ret[idx] - ret[idx].mean()) ** 2).sum(), 1e-8))
        return acc, r2

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            dist = policy.get_distribution(T(obs[b]), action_masks=mask[b])
            policy_loss = -dist.log_prob(T(act[b])).mean()
            value_loss = th.nn.functional.mse_loss(
                policy.predict_values(T(obs[b])).squeeze(1), T(ret[b]))
            loss = policy_loss + 0.5 * value_loss
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            tr_acc, tr_r2 = evaluate(tr)
            va_acc, va_r2 = evaluate(va)
            print(f'epoch {epoch:3d}: action accuracy train {tr_acc:.3f} val {va_acc:.3f} | '
                  f'value R2 train {tr_r2:.2f} val {va_r2:.2f}', flush=True)

    policy.set_training_mode(False)
    out = os.path.join(args.run_dir, 'bc_init')
    model.save(out)
    va_acc, _ = evaluate(va)
    verdict = 'OK' if va_acc >= 0.85 else 'LOW (< 0.85): collect more decisions before PPO'
    print(f'\nSaved {out}.zip  (held-out action accuracy {va_acc:.3f}: {verdict})', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--decisions', type=int, default=3000,
                   help='heuristic decisions to collect (whole episodes are kept)')
    p.add_argument('--num-envs', type=int, default=1, help='parallel simulations')
    p.add_argument('--max-episode-sim-s', type=float, default=300.0)
    p.add_argument('--no-goal-fix', action='store_true',
                   help='raw frontier centroids as goals (must match rl_train)')
    p.add_argument('--dataset', help='existing dataset.npz: skip collection, only clone')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--first-instance', type=int, default=0,
                   help='first sim slot; give concurrent jobs non-overlapping slots')
    p.add_argument('--run-dir', default=os.path.expanduser(
        f'~/swarm_rl_runs/pretrain_{time.strftime("%Y%m%d_%H%M%S")}'))
    args = p.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    print(f'Run dir: {args.run_dir}', flush=True)

    if args.dataset:
        data = dict(np.load(os.path.expanduser(args.dataset)))
    else:
        data = collect(args)
    clone(data, args)


if __name__ == '__main__':
    main()
