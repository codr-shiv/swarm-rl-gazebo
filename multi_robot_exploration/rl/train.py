"""
Train a frontier-selection policy with MaskablePPO on the headless Gazebo stack.

  ros2 run multi_robot_exploration rl_train --resume <run>/bc_init.zip --num-envs 4

Recommended: start from the behaviour-cloned policy made by rl_pretrain
(--resume bc_init.zip); PPO then fine-tunes from heuristic-level play.
See docs/rl_training.md for the full pipeline and why these settings.

Every episode launches a fresh random world. Checkpoints, TensorBoard logs
and sim logs go to --run-dir. Ctrl+C saves the model and shuts the sims down.
"""
import argparse
import os
import signal
import sys
import time

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from multi_robot_exploration.rl.features import MAX_CANDIDATES, OBS_DIM
from multi_robot_exploration.rl.gazebo_env import GazeboExplorationEnv

# PPO settings for fine-tuning a warm-started (behaviour-cloned) policy.
# Sources: Active Neural SLAM global policy (Chaplot et al. 2020: gamma 0.99,
# entropy 0.001, 4 epochs, value coef 0.5); small lr + clip 0.1 + target_kl
# keep PPO close to the cloned heuristic instead of destroying it early.
LEARNING_RATE = 1e-4        # decays linearly to 0 over the run
PPO_KWARGS = dict(
    batch_size=64, n_epochs=4, gamma=0.99, gae_lambda=0.95,
    clip_range=0.1, target_kl=0.015, ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
)
POLICY_KWARGS = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))


def linear_schedule(initial):
    return lambda progress_remaining: progress_remaining * initial


class SpacesOnlyEnv(gym.Env):
    """Has the env's spaces but no simulation; used to build a model offline (rl_pretrain)."""

    observation_space = gym.spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
    action_space = gym.spaces.Discrete(MAX_CANDIDATES)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS_DIM, np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.ones(MAX_CANDIDATES, dtype=bool)


def build_model(env, n_steps, tensorboard_log=None, seed=0):
    return MaskablePPO('MlpPolicy', env, n_steps=n_steps,
                       learning_rate=linear_schedule(LEARNING_RATE),
                       policy_kwargs=POLICY_KWARGS, tensorboard_log=tensorboard_log,
                       seed=seed, verbose=1, **PPO_KWARGS)


def make_env(rank, args, log_dir):
    def _init():
        if args.num_envs > 1:
            # Worker process: Ctrl+C is handled by the main process, which then
            # SIGTERMs us; exiting via sys.exit runs the env's atexit sim cleanup.
            # (A no-op handler rather than SIG_IGN: ignored signals are inherited
            # by the sim's nodes, which then ignore the launch's shutdown request.)
            signal.signal(signal.SIGINT, lambda *_: None)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        return GazeboExplorationEnv(instance_id=args.first_instance + rank, gui=args.gui and rank == 0,
                                    rtf=args.rtf, max_episode_sim_s=args.max_episode_sim_s,
                                    log_dir=log_dir, goal_fix=not args.no_goal_fix)
    return _init


class ExplorationStatsCallback(BaseCallback):
    """Log exploration metrics to TensorBoard at the end of each episode."""

    def _on_step(self):
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self.logger.record_mean('explore/explored_m2', info['explored_m2'])
                self.logger.record_mean('explore/episode_sim_time_s', info['sim_time_s'])
                self.logger.record_mean('explore/decisions', info['decisions'])
                self.logger.record_mean('explore/failed_goals', info['failed_goals'])
                self.logger.record_mean('explore/fully_explored',
                                        float(info.get('end_reason') == 'explored'))
                print(f"[episode] seed={info['world_seed']} end={info.get('end_reason')} "
                      f"explored={info['explored_m2']:.1f} m2 in {info['sim_time_s']:.0f} s "
                      f"({info['decisions']} decisions, {info['failed_goals']} failed)",
                      flush=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--timesteps', type=int, default=20000,
                   help='total decisions to train for (default 20000)')
    p.add_argument('--num-envs', type=int, default=1,
                   help='parallel simulations (each needs ~2 CPU cores and ~2 GB RAM)')
    p.add_argument('--max-episode-sim-s', type=float, default=300.0)
    p.add_argument('--no-goal-fix', action='store_true',
                   help='offer raw frontier centroids as goals (must match rl_pretrain)')
    p.add_argument('--rtf', type=float, default=1.0,
                   help='requested Gazebo real-time factor (>1 = faster, CPU permitting)')
    p.add_argument('--gui', action='store_true', help='show Gazebo for env 0')
    p.add_argument('--n-steps', type=int, default=64,
                   help='decisions per env between PPO updates')
    p.add_argument('--run-dir', default=os.path.expanduser(
        f'~/swarm_rl_runs/{time.strftime("%Y%m%d_%H%M%S")}'))
    p.add_argument('--resume',
                   help='model .zip to start from: bc_init.zip from rl_pretrain, or a checkpoint')
    p.add_argument('--checkpoint-every', type=int, default=1000, help='decisions')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--first-instance', type=int, default=0,
                   help='first sim slot (ROS domain 40+slot, Gazebo port 11445+slot); '
                        'give concurrent jobs non-overlapping slots')
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    log_dir = os.path.join(args.run_dir, 'sim_logs')
    fns = [make_env(i, args, log_dir) for i in range(args.num_envs)]
    venv = DummyVecEnv(fns) if args.num_envs == 1 else SubprocVecEnv(fns, start_method='spawn')
    venv = VecMonitor(venv, os.path.join(args.run_dir, 'monitor'))
    venv.seed(args.seed)

    tb = os.path.join(args.run_dir, 'tb')
    if args.resume:
        # Override the saved hyperparameters so a resumed model (incl. the
        # BC init) always trains with the settings above.
        custom = dict(PPO_KWARGS, n_steps=args.n_steps,
                      learning_rate=linear_schedule(LEARNING_RATE),
                      lr_schedule=linear_schedule(LEARNING_RATE))
        model = MaskablePPO.load(os.path.expanduser(args.resume), env=venv,
                                 tensorboard_log=tb, custom_objects=custom)
    else:
        print('WARNING: training from scratch. Expect little progress in 20k decisions; '
              'run rl_pretrain first and pass --resume <run>/bc_init.zip.', flush=True)
        model = build_model(venv, args.n_steps, tb, args.seed)

    callbacks = [
        ExplorationStatsCallback(),
        CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.num_envs),
                           save_path=os.path.join(args.run_dir, 'checkpoints'),
                           name_prefix='frontier_ppo'),
    ]
    final = os.path.join(args.run_dir, 'frontier_ppo_final')
    print(f'Run dir: {args.run_dir}', flush=True)
    interrupted = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        interrupted = True
        final = os.path.join(args.run_dir, 'frontier_ppo_interrupted')
        print('Interrupted, saving model...', flush=True)
    finally:
        model.save(final)
        print(f'Saved {final}.zip', flush=True)
        inner = venv.venv
        if interrupted and isinstance(inner, SubprocVecEnv):
            # Workers may be busy for minutes (e.g. waiting for a sim), so
            # don't wait for them to answer close(); stop them directly.
            print('Stopping simulations...', flush=True)
            for proc in inner.processes:
                proc.terminate()
            for proc in inner.processes:
                proc.join(timeout=60)
        else:
            venv.close()


if __name__ == '__main__':
    main()
