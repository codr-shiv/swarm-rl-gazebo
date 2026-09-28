#!/usr/bin/env python3
"""
Waypoint Navigator – Demo for Map Merging

Sends a short list of pre-defined waypoints to each robot so they drive
around different parts of the world.  As they move, SLAM builds per-robot
maps and map_merge_node stitches them together into a single /map.

This script is intentionally simple: no frontier detection, no stuck
recovery — just sequential waypoint following to showcase map merging.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus


# ── Waypoints (in the global "map" frame) ─────────────────────────────
# turtlebot3_world has cylindrical pillars (r=0.15m) on a 3x3 grid at
# every 1.1m: (-1.1,-1.1) .. (0,0) .. (1.1,1.1).
# Outer walls are at roughly ±2.5m. Hexagons at (±1.8, ±2.7).
# All waypoints below are in the OPEN CORRIDORS between pillars.
# Requires USE_TB3_WORLD=1 for spawn_two_turtlebots.launch.py.
#
# Robot1 starts at (-2.0, -0.5) → explores left half corridors
ROBOT1_WAYPOINTS = [
    (-1.7, -0.55),     # corridor left of pillar column
    (-0.55, -0.55),    # corridor between pillars row 1-2
    (-0.55,  0.55),    # move up through corridor
    (-1.7,   0.55),    # corridor top-left
]

# Robot2 starts at (2.0, 0.5) → explores right half corridors
ROBOT2_WAYPOINTS = [
    ( 1.7,  0.55),     # corridor right of pillar column
    ( 0.55,  0.55),    # corridor between pillars row 2-3
    ( 0.55, -0.55),    # move down through corridor
    ( 1.7,  -0.55),    # corridor bottom-right
]


class SingleRobotWaypointFollower:
    """Manages sequential waypoint following for one robot via Nav2."""

    def __init__(self, node: Node, robot_name: str, waypoints: list):
        self.node = node
        self.name = robot_name
        self.waypoints = waypoints
        self.current_idx = 0
        self.done = False

        self.action_client = ActionClient(
            node, NavigateToPose, f'/{robot_name}/navigate_to_pose')

        node.get_logger().info(
            f'[{self.name}] Waiting for Nav2 action server...')
        self.action_client.wait_for_server()
        node.get_logger().info(
            f'[{self.name}] Nav2 action server available.')

    def send_next_waypoint(self):
        if self.current_idx >= len(self.waypoints):
            self.node.get_logger().info(
                f'[{self.name}] ✅ All {len(self.waypoints)} waypoints completed!')
            self.done = True
            return

        x, y = self.waypoints[self.current_idx]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.w = 1.0

        self.node.get_logger().info(
            f'[{self.name}] Navigating to waypoint '
            f'{self.current_idx + 1}/{len(self.waypoints)}: '
            f'({x:.1f}, {y:.1f})')

        future = self.action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().warn(
                f'[{self.name}] Goal rejected! Skipping waypoint '
                f'{self.current_idx + 1}.')
            self.current_idx += 1
            self.send_next_waypoint()
            return

        self.node.get_logger().info(
            f'[{self.name}] Goal accepted, navigating...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(
                f'[{self.name}] ✅ Reached waypoint '
                f'{self.current_idx + 1}/{len(self.waypoints)}')
        else:
            self.node.get_logger().warn(
                f'[{self.name}] ⚠️  Waypoint {self.current_idx + 1} '
                f'finished with status {status}. Moving on.')

        self.current_idx += 1
        self.send_next_waypoint()


class WaypointNavigatorNode(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')
        self.declare_parameter('start_delay_sec', 15.0)
        self.delay_sec = int(self.get_parameter('start_delay_sec').value)
        self.remaining_sec = self.delay_sec

        self.get_logger().info('Waypoint Navigator started.')
        self.get_logger().info(f'🎬 OBS Recording Delay: Navigation will begin in {self.delay_sec} seconds...')

        self.robot1 = SingleRobotWaypointFollower(
            self, 'robot1', ROBOT1_WAYPOINTS)
        self.robot2 = SingleRobotWaypointFollower(
            self, 'robot2', ROBOT2_WAYPOINTS)

        # Start 1-second countdown timer for OBS setup
        self._timer = self.create_timer(1.0, self._countdown_cb)

    def _countdown_cb(self):
        if self.remaining_sec > 0:
            if self.remaining_sec in [15, 10, 5, 4, 3, 2, 1]:
                self.get_logger().info(
                    f'⏳ [OBS READY COUNTDOWN] Navigation starting in {self.remaining_sec}s...')
            self.remaining_sec -= 1
        else:
            self._timer.cancel()
            self.get_logger().info('🚀 Starting waypoint navigation NOW!')
            self.robot1.send_next_waypoint()
            self.robot2.send_next_waypoint()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
