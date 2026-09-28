"""
nav2_bringup_multi.launch.py
A bulletproof, auto-activating multi-robot launch script.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, TimerAction
from launch_ros.actions import Node, PushRosNamespace

ROBOTS = ["robot1", "robot2"]

LIFECYCLE_NODE_NAMES = [
    "controller_server",
    "smoother_server",
    "planner_server",
    "behavior_server",
    "bt_navigator",
    "waypoint_follower",
    "velocity_smoother",
]

def generate_launch_description():
    pkg_share = get_package_share_directory("multi_robot_exploration")
    ld = LaunchDescription()

    for name in ROBOTS:
        params_file = os.path.join(pkg_share, "config", f"nav2_params_{name}.yaml")

        nodes = [
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[params_file],
                remappings=[("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_smoother",
                executable="smoother_server",
                name="smoother_server",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[params_file],
                remappings=[("map", "/map")],
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                parameters=[params_file, {"use_sim_time": True}],
            ),
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                output="screen",
                parameters=[params_file, {"use_sim_time": True}],
                remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        # FORCE AUTOSTART ON: This tells Nav2 to wake itself up immediately!
                        "autostart": True,
                        "node_names": LIFECYCLE_NODE_NAMES,
                        "bond_timeout": 0.0,
                    }
                ],
            ),
        ]

        group = GroupAction([PushRosNamespace(name), *nodes])
        if name == "robot2":
            ld.add_action(TimerAction(period=8.0, actions=[group]))
        else:
            ld.add_action(group)

    return ld

