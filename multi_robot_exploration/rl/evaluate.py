"""
Evaluate a frontier-selection policy on fixed random worlds and compare it
with the heuristic from frontier_coordinator.py (and random choice).

  ros2 run multi_robot_exploration rl_evaluate --policy heuristic
  ros2 run multi_robot_exploration rl_evaluate --policy ~/swarm_rl_runs/<run>/frontier_ppo_final.zip

Every policy sees the same world seeds, so results are directly comparable.
Writes one CSV row per episode to --out (appends) and prints a summary.
"""
import argparse
import csv
import os
import time

import numpy as np

from multi_robot_exploration.rl.features import heuristic_action_from_obs
from multi_robot_exploration.rl.gazebo_env import GazeboExplorationEnv

DEFAULT_SEEDS = [101, 202, 303, 404, 505]


def make_policy(spec, env):
    if spec == 'heuristic':
        return heuristic_action_from_obs   # exactly the slot the observation flags
    if spec == 'random':
        rng = np.random.default_rng(0)
        return lambda _obs: int(rng.choice(np.flatnonzero(env.action_masks())))

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(os.path.expanduser(spec))
    return lambda obs: int(model.predict(obs, action_masks=env.action_masks(),
                                         deterministic=True)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy', required=True,
                   help="'heuristic', 'random', or a path to a MaskablePPO .zip")
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--max-episode-sim-s', type=float, default=300.0)
    p.add_argument('--no-goal-fix', action='store_true',
                   help='raw frontier centroids as goals (A/B test; must match training)')
    p.add_argument('--rtf', type=float, default=1.0)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--instance-id', type=int, default=9,
                   help='sim slot (ROS domain / Gazebo port); differs from training by default')
    p.add_argument('--out', default=os.path.expanduser('~/swarm_rl_runs/eval.csv'))
    args = p.parse_args()

    env = GazeboExplorationEnv(instance_id=args.instance_id, gui=args.gui, rtf=args.rtf,
                               max_episode_sim_s=args.max_episode_sim_s,
                               world_seeds=args.seeds, goal_fix=not args.no_goal_fix,
                               log_dir=os.path.join(os.path.dirname(args.out), 'eval_sim_logs'))
    policy = make_policy(args.policy, env)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)
    rows = []
    try:
        with open(args.out, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(['timestamp', 'policy', 'goal_fix', 'world_seed', 'end_reason', 'explored_m2',
                            'sim_time_s', 'return', 'decisions', 'failed_goals'])
            for _ in args.seeds:
                obs, info = env.reset()
                ret, done = 0.0, False
                while not done:
                    obs, r, term, trunc, info = env.step(policy(obs))
                    ret += r
                    done = term or trunc
                row = [time.strftime('%F %T'), args.policy, not args.no_goal_fix, info['world_seed'],
                       info['end_reason'], round(info['explored_m2'], 2),
                       round(info['sim_time_s'], 1), round(ret, 2), info['decisions'],
                       info['failed_goals']]
                w.writerow(row)
                fh.flush()
                rows.append(row)
                print(f'seed={row[3]} end={row[4]} explored={row[5]} m2 '
                      f'time={row[6]} s return={row[7]} failed={row[9]}', flush=True)
    finally:
        env.close()

    if rows:
        explored = np.array([r[5] for r in rows])
        times = np.array([r[6] for r in rows])
        failed = np.array([r[9] for r in rows])
        done = sum(r[4] in ('explored', 'saturated') for r in rows)
        print(f'\n{args.policy} (goal_fix={not args.no_goal_fix}): explored {explored.mean():.1f} '
              f'± {explored.std():.1f} m2, episode time {times.mean():.0f} s, '
              f'failed goals {failed.mean():.1f}, finished before limit {done}/{len(rows)}')
        print(f'Results appended to {args.out}')


if __name__ == '__main__':
    main()
