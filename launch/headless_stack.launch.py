"""
headless_stack.launch.py

Brings up the whole exploration stack in ONE launch, headless by default:
Gazebo (gzserver only) + 2 robots -> per-robot SLAM -> map merge -> Nav2.
There is no exploration brain: the weight-tuning episodes (or
tuned_frontier_coordinator) send the NavigateToPose goals.

Used by multi_robot_exploration.tuning.sim_manager for every tuning episode,
but it also works by hand (e.g. to watch what the agent sees):
  ros2 launch multi_robot_exploration headless_stack.launch.py gui:=true

World selection uses the same environment variables as
spawn_two_turtlebots.launch.py (GAZEBO_WORLD_SEED, USE_HOUSE, USE_TB3_WORLD,
GAZEBO_RTF).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = os.path.join(
        get_package_share_directory("multi_robot_exploration"), "launch")

    def include(name, **kwargs):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, name)), **kwargs)

    gui_arg = DeclareLaunchArgument(
        "gui", default_value="false", description="Start the Gazebo client window")

    # Same order as the manual 5-terminal bring-up; delays are wall-clock
    # seconds giving each stage time to come up before the next one needs it.
    return LaunchDescription([
        gui_arg,
        include("spawn_two_turtlebots.launch.py",
                launch_arguments={"gui": LaunchConfiguration("gui")}.items()),
        TimerAction(period=10.0, actions=[include("multi_robot_slam.launch.py")]),
        TimerAction(period=14.0, actions=[include("map_merge.launch.py")]),
        TimerAction(period=18.0, actions=[include("nav2_bringup_multi.launch.py")]),
    ])
