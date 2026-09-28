import os
from glob import glob
from setuptools import find_packages, setup

package_name = "multi_robot_exploration"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")) + glob(os.path.join("config", "*.rviz"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Two-TurtleBot3 Gazebo spawn + SLAM + map merging",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ['map_merge_node = multi_robot_exploration.map_merge_node:main',
        'frontier_coordinator = multi_robot_exploration.frontier_coordinator:main',
        'waypoint_navigator = multi_robot_exploration.waypoint_navigator:main',],
    },
)
