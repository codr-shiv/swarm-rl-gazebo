"""
Train a frontier-selection policy with MaskablePPO on the headless Gazebo stack.

  ros2 run multi_robot_exploration rl_train --timesteps 20000 --num-envs 2

Every episode launches a fresh random world. Checkpoints, TensorBoard logs
and sim logs go to --run-dir. Ctrl+C saves the model and shuts the sims down.
"""
import argparse
import os
import signal
import sys
import time

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from multi_robot_exploration.rl.gazebo_env import GazeboExplorationEnv


def make_env(rank, args, log_dir):
    def _init():
        if args.num_envs > 1:
            # Worker process: Ctrl+C is handled by the main process, which then
            # SIGTERMs us; exiting via sys.exit runs the env's atexit sim cleanup.
            # (A no-op handler rather than SIG_IGN: ignored signals are inherited
            # by the sim's nodes, which then ignore the launch's shutdown request.)
            signal.signal(signal.SIGINT, lambda *_: None)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        return GazeboExplorationEnv(instance_id=rank, gui=args.gui and rank == 0,
                                    rtf=args.rtf, max_episode_sim_s=args.max_episode_sim_s,
                                    log_dir=log_dir)
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
    p.add_argument('--max-episode-sim-s', type=float, default=600.0)
    p.add_argument('--rtf', type=float, default=1.0,
                   help='requested Gazebo real-time factor (>1 = faster, CPU permitting)')
    p.add_argument('--gui', action='store_true', help='show Gazebo for env 0')
    p.add_argument('--n-steps', type=int, default=128,
                   help='decisions per env between PPO updates')
    p.add_argument('--run-dir', default=os.path.expanduser(
        f'~/swarm_rl_runs/{time.strftime("%Y%m%d_%H%M%S")}'))
    p.add_argument('--resume', help='path to a saved model .zip to continue training')
    p.add_argument('--checkpoint-every', type=int, default=1000, help='decisions')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    log_dir = os.path.join(args.run_dir, 'sim_logs')
    fns = [make_env(i, args, log_dir) for i in range(args.num_envs)]
    venv = DummyVecEnv(fns) if args.num_envs == 1 else SubprocVecEnv(fns, start_method='spawn')
    venv = VecMonitor(venv, os.path.join(args.run_dir, 'monitor'))
    venv.seed(args.seed)

    if args.resume:
        model = MaskablePPO.load(args.resume, env=venv,
                                 tensorboard_log=os.path.join(args.run_dir, 'tb'))
    else:
        model = MaskablePPO(
            'MlpPolicy', venv,
            n_steps=args.n_steps, batch_size=64, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            policy_kwargs={'net_arch': [128, 128]},
            tensorboard_log=os.path.join(args.run_dir, 'tb'),
            seed=args.seed, verbose=1)

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
