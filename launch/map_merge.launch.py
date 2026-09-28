"""
map_merge.launch.py

Launches our own map_merge_node (multi_robot_exploration.map_merge_node), which
fuses /robot1/map and /robot2/map into /map using known spawn poses.

Run AFTER spawn_two_turtlebots.launch.py + multi_robot_slam.launch.py:
  ros2 launch multi_robot_exploration map_merge.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("multi_robot_exploration")
    params_file = os.path.join(pkg_share, "config", "map_merge_params.yaml")

    map_merge_node = Node(
        package="multi_robot_exploration",
        executable="map_merge_node",
        name="map_merge_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": True}],
    )

    return LaunchDescription([map_merge_node])
