"""
multi_robot_slam.launch.py

Runs one slam_toolbox (async) instance per robot, namespaced, so each
robot builds its own local map: /robot1/map and /robot2/map.

Run AFTER spawn_two_turtlebots.launch.py:
  ros2 launch multi_robot_exploration multi_robot_slam.launch.py
"""

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

ROBOTS = ["robot1", "robot2"]


def generate_launch_description():
    ld = LaunchDescription()

    for name in ROBOTS:
        group = GroupAction(
            [
                PushRosNamespace(name),
                Node(
                    package="slam_toolbox",
                    executable="async_slam_toolbox_node",
                    name="slam_toolbox",
                    output="screen",
                    parameters=[
                        {
                            "use_sim_time": True,
                            "odom_frame": f"{name}/odom",
                            "base_frame": f"{name}/base_footprint",
                            "map_frame": f"{name}/map",
                            "scan_topic": "scan",
                        }
                    ],
                    # slam_toolbox publishes map/map_metadata on an ABSOLUTE
                    # topic internally, which bypasses PushRosNamespace above.
                    # Without this remap both robots collide onto the same
                    # global /map. Remapping the absolute name to a relative
                    # one lets namespace resolution apply again -> /robot1/map.
                    remappings=[
                        ("/map", "map"),
                        ("/map_metadata", "map_metadata"),
                    ],
                ),
            ]
        )
        ld.add_action(group)

    return ld
