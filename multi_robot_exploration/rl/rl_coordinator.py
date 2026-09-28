#!/usr/bin/env python3
"""
RL frontier coordinator: drop-in replacement for frontier_coordinator in the
normal pipeline (terminal 5), choosing frontiers with a trained policy.

  ros2 launch multi_robot_exploration rl_frontier_exploration.launch.py \
      model_path:=$HOME/swarm_rl_runs/<run>/frontier_ppo_final.zip

Uses the same ExplorationInterface (observations, goal handling) as training.
"""
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from sb3_contrib import MaskablePPO

from multi_robot_exploration.rl.ros_interface import ROBOTS, ExplorationInterface


class RLFrontierCoordinator(ExplorationInterface):

    def __init__(self):
        super().__init__('rl_frontier_coordinator')
        self.declare_parameter('model_path', '')
        self.declare_parameter('max_episode_sim_s', 600.0)
        path = os.path.expanduser(self.get_parameter('model_path').value)
        if not path or not os.path.exists(path):
            raise RuntimeError(f'model_path does not exist: {path!r}')
        self.model = MaskablePPO.load(path)
        self.max_episode_s = self.get_parameter('max_episode_sim_s').value
        self.create_timer(1.0, self._tick)
        self.get_logger().info(f'Loaded policy {path}; waiting for Nav2 + map...')

    def _tick(self):
        if not all(c.server_is_ready() for c in self.nav_clients.values()):
            return
        if self.latest_map is None or any(self.robot_pose(r) is None for r in ROBOTS):
            return
        if self.episode_start is None:
            self.episode_start = self.sim_now()
            self.get_logger().info('Exploration started.')

        self.update_homes()
        pts, sizes = self.frontiers()
        for robot, reason in self.poll_events(pts).items():
            self.get_logger().info(f'{robot}: goal {reason}')
        if not pts:
            self.get_logger().info('No frontiers remain.', throttle_duration_sec=30)
            return

        for robot in ROBOTS:
            if not self.is_idle(robot):
                continue
            d = self.decision(robot, pts, sizes, self.max_episode_s)
            if d is None or not d[1].any():
                continue
            obs, mask, cands = d
            a = int(self.model.predict(obs, action_masks=mask, deterministic=True)[0])
            self.send_goal(robot, cands[a])
            self.get_logger().info(
                f'Assigned {robot} -> ({cands[a][0]:.2f}, {cands[a][1]:.2f}) [slot {a}]')


def main(args=None):
    # Plain KeyboardInterrupt on Ctrl+C (rclpy's handler races spin() at
    # shutdown), and the context stays valid so goals can be cancelled.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RLFrontierCoordinator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cancel_all()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
