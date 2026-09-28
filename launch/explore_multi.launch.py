"""
explore_multi.launch.py

WARNING: Do NOT launch this alongside frontier_coordinator — they both
send NavigateToPose goals to each robot's Nav2 and will fight over them.
Use ONE exploration system at a time:
  • frontier_coordinator  (custom, uses merged map, coordinates robots)
  • explore_lite           (off-the-shelf, per-robot, no cross-robot coordination)

Runs one explore_lite instance per robot (package: explore_lite, from
m-explore-ros2: https://github.com/robo-friends/m-explore-ros2), each
using that robot's OWN local_costmap/global_costmap to pick frontiers and
sending goals through that robot's OWN Nav2 stack.

This is the standard multi-robot pattern for explore_lite — it has no
built-in cross-robot coordination, so both robots occasionally head for
the same area. That's expected/normal, not a bug.

Prereq: build explore_lite into this workspace first (not an apt package
for Humble):
  cd ~/ros2_ws/src
  git clone https://github.com/robo-friends/m-explore-ros2.git
  cd ~/ros2_ws && colcon build --packages-select explore_lite

Run LAST, after spawn, multi_robot_slam, map_merge, and
nav2_bringup_multi are all already running:
  ros2 launch multi_robot_exploration explore_multi.launch.py
"""

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

ROBOTS = ["robot1", "robot2"]


def generate_launch_description():
    ld = LaunchDescription()

    for name in ROBOTS:
        explore_node = Node(
            package="explore_lite",
            executable="explore",
            name="explore_node",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "robot_base_frame": f"{name}/base_footprint",
                    "costmap_topic": "global_costmap/costmap",
                    "costmap_updates_topic": "global_costmap/costmap_updates",
                    "visualize": True,
                    "planner_frequency": 0.33,
                    "progress_timeout": 30.0,
                    "potential_scale": 3.0,
                    "orientation_scale": 0.0,
                    "gain_scale": 1.0,
                    "transform_tolerance": 0.3,
                    "min_frontier_size": 0.5,
                }
            ],
        )
        ld.add_action(GroupAction([PushRosNamespace(name), explore_node]))

    return ld
