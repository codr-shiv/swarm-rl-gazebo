"""
waypoint_demo.launch.py

Launches the waypoint_navigator node for map-merge demonstration.
Run this AFTER spawn, SLAM, map_merge, and nav2_bringup are all active.

The waypoints are hardcoded for turtlebot3_world, so start the spawn launch
with that world:
  USE_TB3_WORLD=1 ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    start_delay_arg = DeclareLaunchArgument(
        'start_delay_sec',
        default_value='15.0',
        description='Delay in seconds before navigation starts to give time for OBS setup'
    )

    return LaunchDescription([
        start_delay_arg,
        Node(
            package='multi_robot_exploration',
            executable='waypoint_navigator',
            name='waypoint_navigator',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'start_delay_sec': LaunchConfiguration('start_delay_sec'),
            }],
        ),
    ])
