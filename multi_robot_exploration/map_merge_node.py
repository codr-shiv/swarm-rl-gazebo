#!/usr/bin/env python3
"""
map_merge_node.py

From-scratch multi-robot map merging.

Strategy: each robot's SLAM map (slam_toolbox) lives in its own
`<robot>/map` frame. A per-robot (x, y, yaw) offset is used as the static
transform from a common `map` frame to that robot's local map frame. Every occupied/free cell from each
submap is reprojected into the common frame and rasterized into one
fused OccupancyGrid, which is published on /map.

Fusion rule per merged cell:
  - if either source says occupied (100) -> occupied wins
  - if only one source has data -> use it
  - if both have free/partial data -> average them

Params:
  robot_names   (string array) e.g. ['robot1', 'robot2']
  <name>_x, <name>_y, <name>_yaw  (double, per robot) map offset.
      Keep these at 0 in Gazebo: the diff-drive plugin's odom already starts
      at the world spawn pose (see config/map_merge_params.yaml).
  merged_frame  (string) default 'map'
  publish_rate  (double) Hz, default 2.0
  merge_resolution (double) default -1.0 -> auto (min of source resolutions)
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                        QoSReliabilityPolicy)
from tf2_ros import StaticTransformBroadcaster


class RobotMapSource:
    def __init__(self, name, x, y, yaw):
        self.name = name
        self.x = x
        self.y = y
        self.yaw = yaw
        self.map = None  # latest nav_msgs/OccupancyGrid


class MapMergeNode(Node):

    def __init__(self):
        super().__init__('map_merge_node')

        self.declare_parameter('robot_names', ['robot1', 'robot2'])
        self.declare_parameter('merged_frame', 'map')
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('merge_resolution', -1.0)

        robot_names = self.get_parameter('robot_names').value
        self.merged_frame = self.get_parameter('merged_frame').value
        publish_rate = self.get_parameter('publish_rate').value
        self.resolution_override = self.get_parameter('merge_resolution').value

        self.robots = {}
        for name in robot_names:
            self.declare_parameter(f'{name}_x', 0.0)
            self.declare_parameter(f'{name}_y', 0.0)
            self.declare_parameter(f'{name}_yaw', 0.0)
            x = self.get_parameter(f'{name}_x').value
            y = self.get_parameter(f'{name}_y').value
            yaw = self.get_parameter(f'{name}_yaw').value
            self.robots[name] = RobotMapSource(name, x, y, yaw)

        # Matches slam_toolbox's default map publishing QoS
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.history = QoSHistoryPolicy.KEEP_LAST

        for name, robot in self.robots.items():
            self.create_subscription(
                OccupancyGrid,
                f'/{name}/map',
                lambda msg, r=robot: self._map_cb(msg, r),
                map_qos,
            )

        self.merged_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_transforms()

        self.create_timer(1.0 / publish_rate, self._merge_and_publish)
        self.get_logger().info(
            f'map_merge_node started, fusing: {list(self.robots.keys())}'
        )

    # ---- TF: map -> <robot>/map, from known spawn poses -----------------
    def _broadcast_static_transforms(self):
        transforms = []
        stamp = self.get_clock().now().to_msg()
        for name, r in self.robots.items():
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.merged_frame
            t.child_frame_id = f'{name}/map'
            t.transform.translation.x = r.x
            t.transform.translation.y = r.y
            t.transform.translation.z = 0.0
            qx, qy, qz, qw = self._yaw_to_quat(r.yaw)
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            transforms.append(t)
        self.tf_broadcaster.sendTransform(transforms)

    @staticmethod
    def _yaw_to_quat(yaw):
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def _map_cb(self, msg, robot):
        robot.map = msg

    def _transform_point(self, x, y, robot):
        """2D rigid transform: robot's local map frame -> merged frame."""
        c = math.cos(robot.yaw)
        s = math.sin(robot.yaw)
        wx = c * x - s * y + robot.x
        wy = s * x + c * y + robot.y
        return wx, wy

    # ---- Merge core -------------------------------------------------------
    def _merge_and_publish(self):
        active = {n: r for n, r in self.robots.items() if r.map is not None}
        if not active:
            return

        resolution = self.resolution_override
        if resolution is None or resolution <= 0.0:
            resolution = min(r.map.info.resolution for r in active.values())

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for r in active.values():
            info = r.map.info
            w_m = info.width * info.resolution
            h_m = info.height * info.resolution
            for cx, cy in [(0.0, 0.0), (w_m, 0.0), (0.0, h_m), (w_m, h_m)]:
                ox = info.origin.position.x + cx
                oy = info.origin.position.y + cy
                wx, wy = self._transform_point(ox, oy, r)
                min_x, max_x = min(min_x, wx), max(max_x, wx)
                min_y, max_y = min(min_y, wy), max(max_y, wy)

        if not math.isfinite(min_x):
            return

        pad = 1.0  # metres
        min_x -= pad
        min_y -= pad
        max_x += pad
        max_y += pad

        width = max(1, int(math.ceil((max_x - min_x) / resolution)))
        height = max(1, int(math.ceil((max_y - min_y) / resolution)))

        merged = np.full((height, width), -1, dtype=np.int16)

        for r in active.values():
            info = r.map.info
            src = np.array(r.map.data, dtype=np.int16).reshape(
                (info.height, info.width)
            )
            occ_rows, occ_cols = np.where(src >= 0)  # skip unknown cells
            if len(occ_rows) == 0:
                continue

            vals = src[occ_rows, occ_cols]

            # Vectorised local-to-world coordinate transform
            lx = info.origin.position.x + (occ_cols.astype(np.float64) + 0.5) * info.resolution
            ly = info.origin.position.y + (occ_rows.astype(np.float64) + 0.5) * info.resolution
            c = math.cos(r.yaw)
            s = math.sin(r.yaw)
            wx = c * lx - s * ly + r.x
            wy = s * lx + c * ly + r.y

            # Vectorised world-to-merged-grid index conversion
            mx = ((wx - min_x) / resolution).astype(np.intp)
            my = ((wy - min_y) / resolution).astype(np.intp)

            # Filter to valid grid bounds
            valid = (mx >= 0) & (mx < width) & (my >= 0) & (my < height)
            mx, my, vals = mx[valid], my[valid], vals[valid]

            # Vectorised fusion (occupied wins > direct assign > average)
            cur = merged[my, mx]

            # Where merged cell is still unknown: assign incoming directly
            unk = cur == -1
            merged[my[unk], mx[unk]] = vals[unk]

            # Where incoming is occupied (100): occupied always wins
            occ = ~unk & (vals == 100)
            merged[my[occ], mx[occ]] = 100

            # Where both are known and non-occupied: average
            avg = ~unk & ~occ & (cur >= 0) & (cur < 100) & (vals >= 0)
            if np.any(avg):
                merged[my[avg], mx[avg]] = (
                    (cur[avg] + vals[avg]) // 2).astype(np.int16)

        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.merged_frame
        out.info.resolution = float(resolution)
        out.info.width = width
        out.info.height = height
        out.info.origin.position.x = min_x
        out.info.origin.position.y = min_y
        out.info.origin.position.z = 0.0
        out.info.origin.orientation.w = 1.0
        out.data = merged.astype(np.int8).flatten().tolist()

        self.merged_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MapMergeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
