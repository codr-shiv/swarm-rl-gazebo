"""
Compare weight sets on fixed random worlds (held out from tuning).

  ros2 run multi_robot_exploration evaluate_weights --weights heuristic
  ros2 run multi_robot_exploration evaluate_weights --weights ~/swarm_tuning_runs/<run>/best_weights.json

Every weight set sees the same world seeds, so rows are directly comparable.
Appends one CSV row per episode to --out and prints a summary.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from multi_robot_exploration.tuning.episode import EpisodeRunner
from multi_robot_exploration.tuning.features import HEURISTIC_WEIGHTS, TERMS

DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]


def load_weights(spec):
    if spec == 'heuristic':
        return dict(HEURISTIC_WEIGHTS)
    with open(os.path.expanduser(spec)) as fh:
        return json.load(fh)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', required=True, help="'heuristic' or a weights .json file")
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--max-episode-sim-s', type=float, default=300.0)
    p.add_argument('--no-goal-fix', action='store_true',
                   help='raw frontier centroids as goals (A/B test; must match tuning)')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--rtf', type=float, default=1.0)
    p.add_argument('--instance-id', type=int, default=9,
                   help='sim slot (ROS domain / Gazebo port); keep different from running jobs')
    p.add_argument('--out', default=os.path.expanduser('~/swarm_tuning_runs/eval.csv'))
    args = p.parse_args()

    weights = load_weights(args.weights)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    runner = EpisodeRunner(instance_id=args.instance_id, gui=args.gui, rtf=args.rtf,
                           max_episode_sim_s=args.max_episode_sim_s,
                           log_dir=os.path.join(os.path.dirname(args.out), 'eval_sim_logs'),
                           goal_fix=not args.no_goal_fix)
    new_file = not os.path.exists(args.out)
    rows = []
    try:
        with open(args.out, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(['timestamp', 'weights', 'goal_fix', 'world_seed', 'score', 'end_reason',
                            'explored_m2', 'sim_time_s', 'decisions', 'failed_goals']
                           + [f'w_{t}' for t in TERMS])
            for seed in args.seeds:
                try:
                    st = runner.run(weights, seed)
                except Exception as e:   # one bad world shouldn't end the evaluation
                    print(f'seed={seed}: FAILED ({type(e).__name__}: {e}), skipped', flush=True)
                    continue
                row = [time.strftime('%F %T'), args.weights, not args.no_goal_fix, seed,
                       round(st['score'], 3), st['end_reason'], round(st['explored_m2'], 2),
                       round(st['sim_time_s'], 1), st['decisions'], st['failed_goals']] \
                    + [weights.get(t, 0.0) for t in TERMS]
                w.writerow(row)
                fh.flush()
                rows.append(st)
                print(f"seed={seed} end={st['end_reason']} score={st['score']:.3f} "
                      f"explored={st['explored_m2']:.1f} m2 time={st['sim_time_s']:.0f} s "
                      f"failed={st['failed_goals']}", flush=True)
    finally:
        runner.close()

    if rows:
        f = lambda k: np.array([r[k] for r in rows])  # noqa: E731
        print(f"\n{args.weights} (goal_fix={not args.no_goal_fix}): score {f('score').mean():.3f} "
              f"± {f('score').std():.3f}, explored {f('explored_m2').mean():.1f} m2, "
              f"time {f('sim_time_s').mean():.0f} s, failed goals {f('failed_goals').mean():.1f}")
        print(f'Results appended to {args.out}')
    if len(rows) < len(args.seeds):
        print(f'WARNING: {len(args.seeds) - len(rows)} of {len(args.seeds)} worlds failed; '
              'compare weight sets only on worlds both completed.')


if __name__ == '__main__':
    main()
