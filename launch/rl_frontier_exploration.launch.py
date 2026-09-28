"""
rl_frontier_exploration.launch.py
Runs the trained RL policy instead of frontier_coordinator (terminal 5).

  ros2 launch multi_robot_exploration rl_frontier_exploration.launch.py \
      model_path:=$HOME/swarm_rl_runs/<run>/frontier_ppo_final.zip
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', description='MaskablePPO .zip to load'),
        Node(
            package='multi_robot_exploration',
            executable='rl_frontier_coordinator',
            name='rl_frontier_coordinator',
            output='screen',
            parameters=[{'use_sim_time': True,
                         'model_path': LaunchConfiguration('model_path')}],
        ),
    ])
