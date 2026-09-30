"""
tuned_frontier_exploration.launch.py
Runs the frontier formula with tuned weights instead of frontier_coordinator (terminal 5).

  ros2 launch multi_robot_exploration tuned_frontier_exploration.launch.py \
      weights_file:=$HOME/swarm_tuning_runs/<run>/best_weights.json

Without weights_file the heuristic weights (= frontier_coordinator) are used.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('weights_file', default_value='',
                              description='best_weights.json from tune_weights'),
        Node(
            package='multi_robot_exploration',
            executable='tuned_frontier_coordinator',
            name='tuned_frontier_coordinator',
            output='screen',
            parameters=[{'use_sim_time': True,
                         'weights_file': LaunchConfiguration('weights_file')}],
        ),
    ])
