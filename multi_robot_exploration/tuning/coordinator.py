#!/usr/bin/env python3
"""
Tuned frontier coordinator: drop-in replacement for frontier_coordinator in
the normal pipeline (terminal 5), choosing frontiers with the weighted formula.

  ros2 launch multi_robot_exploration tuned_frontier_exploration.launch.py \
      weights_file:=$HOME/swarm_tuning_runs/<run>/best_weights.json

Uses the same ExplorationInterface (candidates, goal handling) as tuning.
"""
import json
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions

from multi_robot_exploration.tuning.features import HEURISTIC_WEIGHTS, choose
from multi_robot_exploration.tuning.ros_interface import ROBOTS, ExplorationInterface


class TunedFrontierCoordinator(ExplorationInterface):

    def __init__(self):
        super().__init__('tuned_frontier_coordinator')
        self.declare_parameter('weights_file', '')
        self.declare_parameter('goal_fix', True)     # must match tuning
        path = os.path.expanduser(self.get_parameter('weights_file').value)
        if path:
            with open(path) as fh:
                self.weights = json.load(fh)
        else:
            self.weights = dict(HEURISTIC_WEIGHTS)
        self.goal_fix = self.get_parameter('goal_fix').value
        self.create_timer(1.0, self._tick)
        self.get_logger().info(f'Weights {self.weights}; waiting for Nav2 + map...')

    def _tick(self):
        if not all(c.server_is_ready() for c in self.nav_clients.values()):
            return
        if self.latest_map is None or any(self.robot_pose(r) is None for r in ROBOTS):
            return

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
            d = self.decision(robot, pts, sizes)
            if d is None or not len(d[0]) or not d[2].any():
                continue
            candidates, terms, mask = d
            goal = candidates[choose(terms, mask, self.weights)]
            self.send_goal(robot, goal)
            self.get_logger().info(f'Assigned {robot} -> ({goal[0]:.2f}, {goal[1]:.2f})')


def main(args=None):
    # Plain KeyboardInterrupt on Ctrl+C (rclpy's handler races spin() at
    # shutdown), and the context stays valid so goals can be cancelled.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TunedFrontierCoordinator()
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
