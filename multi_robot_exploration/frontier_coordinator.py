#!/usr/bin/env python3
"""
Multi-Robot Frontier Coordinator

Features:
  1. Spatial partitioning — robots are assigned to different map regions so
     they explore opposite sides instead of crowding the same area.
  2. Greedy cost-based assignment — each robot takes the cheapest frontier
     by distance + separation penalty + region ownership bias.
  3. Rich RViz markers — frontier spheres, per-robot goal arrows, and
     assignment-colored frontier regions for full visual debugging.
  4. Nav2 action client — uses NavigateToPose action for proper goal
     tracking, failure detection, and cancellation instead of raw
     /goal_pose publishing.
"""
from enum import Enum, auto

import cv2  # type: ignore[import-untyped]
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                        QoSReliabilityPolicy)
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


# ── Robot state machine ──────────────────────────────────────────────────
class RobotState(Enum):
    IDLE = auto()          # No goal assigned yet
    NAVIGATING = auto()    # Actively driving towards a frontier


# ── Per-robot colour palette ─────────────────────────────────────────────
ROBOT_COLORS = {
    'robot1': ColorRGBA(r=0.0, g=0.94, b=1.0, a=0.9),  # cyan
    'robot2': ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.9),   # orange
}


class RobotContext:
    """Tracks per-robot navigation state."""
    def __init__(self, name):
        self.name = name
        self.state = RobotState.IDLE
        self.goal = None            # np.array([x, y]) or None
        self.goal_handle = None     # action goal handle
        self.goal_start_time = None
        self.stuck_counter = 0
        self.last_position = None   # np.array([x, y])


class MultiRobotFrontierCoordinator(Node):
    def __init__(self):
        super().__init__('frontier_coordinator')

        self.robot_names = ['robot1', 'robot2']
        self.robots = {n: RobotContext(n) for n in self.robot_names}

        # TF2 for looking up each robot's current position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS matching map_merge_node's publisher (RELIABLE + TRANSIENT_LOCAL)
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.history = QoSHistoryPolicy.KEEP_LAST

        # Subscribe to the MERGED map (single source of truth)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, map_qos)

        self.latest_map = None

        # Nav2 action clients (one per robot)
        self.nav_clients = {}
        for name in self.robot_names:
            self.nav_clients[name] = ActionClient(
                self, NavigateToPose, f'/{name}/navigate_to_pose')

        # Cmd_vel publishers for emergency stop
        self.cmd_pubs = {}
        for name in self.robot_names:
            self.cmd_pubs[name] = self.create_publisher(
                Twist, f'/{name}/cmd_vel', 10)

        # ── Visualisation publishers ──
        self.frontier_marker_pub = self.create_publisher(
            MarkerArray, '/frontiers_markers', 10)
        self.goal_marker_pub = self.create_publisher(
            MarkerArray, '/frontier_goals_markers', 10)

        # ── Tuning knobs ──
        self.dedup_radius = 1.5            # metres – merge centroids closer than this
        self.goal_retention_radius = 2.5
        self.min_commitment_time = 10.0
        self.min_frontier_size = 15        # pixels – ignore tiny noise clusters
        self.goal_reached_threshold = 0.4  # metres – consider goal reached
        self.min_goal_distance = 0.6       # metres – skip too-close frontiers
        self.separation_weight = 50.0       # weight for inter-robot separation cost (INCREASED)
        self.region_weight = 50.0           # weight for region ownership bias (INCREASED)
        self.stuck_threshold = 3           # replan ticks of no movement before re-goal

        # Blacklisted goals to prevent infinite assignment loops (temporary blacklist)
        self.blacklisted_frontiers = {}  # tuple(x,y) -> timestamp
        self.blacklist_duration = 30.0   # seconds

        # Store initial positions for stable spatial partitioning
        self.home_positions = {}

        # Fairness flag: don't start assigning until all robots have valid TF
        self.all_robots_ready = False

        # Wait for all Nav2 action servers before starting exploration
        self.get_logger().info('Waiting for Nav2 action servers...')
        for name in self.robot_names:
            while not self.nav_clients[name].wait_for_server(timeout_sec=2.0):
                self.get_logger().info(f'  ...waiting for /{name}/navigate_to_pose')
        self.get_logger().info('All Nav2 action servers ready!')

        # Timer-based replanning
        self.replan_period = 3.0  # seconds
        self.replan_timer = self.create_timer(self.replan_period, self._replan)

        self.get_logger().info(
            "Frontier Coordinator started (spatial partitioning, "
            f"replan every {self.replan_period}s).")

    # ══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════════
    def _map_callback(self, msg: OccupancyGrid):
        """Store the latest merged map; replanning happens on the timer."""
        self.latest_map = msg

    # ══════════════════════════════════════════════════════════════════════
    # Robot pose via TF2
    # ══════════════════════════════════════════════════════════════════════
    def _get_robot_position(self, robot_name):
        """Look up robot's (x, y) in the merged 'map' frame."""
        try:
            # No timeout: a blocking wait inside a timer callback stalls the
            # single-threaded executor (and the TF listener with it).
            t = self.tf_buffer.lookup_transform(
                'map', f'{robot_name}/base_footprint',
                rclpy.time.Time())
            pos = np.array([t.transform.translation.x,
                             t.transform.translation.y])
            return pos
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed for {robot_name}: {e}",
                throttle_duration_sec=10)
            return None

    # ══════════════════════════════════════════════════════════════════════
    # Frontier detection (morphological dilation)
    # ══════════════════════════════════════════════════════════════════════
    def _detect_frontiers(self, msg):
        """Return a list of (x, y) world-frame frontier centroids."""
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))

        unknown_mask = np.where(grid == -1, 255, 0).astype(np.uint8)
        free_mask = np.where(grid == 0, 255, 0).astype(np.uint8)
        obstacle_mask = np.where(grid >= 50, 255, 0).astype(np.uint8)

        # Remove small ghost obstacles (like the other robot's laser signature)
        # by applying morphological opening (erode then dilate)
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, noise_kernel)

        # Inflate obstacles by ~0.15 meters to ensure frontiers are a safe distance away
        # Resolution is usually 0.05m/pixel. 0.15m / 0.05m = 3 pixels radius.
        obs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_obstacles = cv2.dilate(obstacle_mask, obs_kernel, iterations=1)
        safe_free_mask = cv2.bitwise_and(free_mask, cv2.bitwise_not(dilated_obstacles))

        # Frontier = safe free cell adjacent to at least one unknown cell
        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(unknown_mask, kernel, iterations=1)
        frontier_mask = cv2.bitwise_and(safe_free_mask, dilated_unknown)

        num_labels, _labels, stats, centroids = \
            cv2.connectedComponentsWithStats(frontier_mask)

        points = []
        for i in range(1, num_labels):                      # skip background
            if stats[i, cv2.CC_STAT_AREA] > self.min_frontier_size:
                cx, cy = centroids[i]
                points.append(np.array([ox + cx * res,
                                        oy + cy * res]))

        return points

    # ══════════════════════════════════════════════════════════════════════
    # Deduplication
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _deduplicate(frontiers, radius):
        """Keep only one centroid per cluster within *radius* metres."""
        if not frontiers:
            return []
        unique = [frontiers[0]]
        for f in frontiers[1:]:
            if all(np.linalg.norm(f - u) > radius for u in unique):
                unique.append(f)
        return unique

    # ══════════════════════════════════════════════════════════════════════
    # Spatial partitioning — region ownership
    # ══════════════════════════════════════════════════════════════════════
    def _region_cost(self, robot_name, frontier_pt, positions):
        """
        Return a cost penalty based on how far the frontier is from
        this robot's 'preferred' region.

        Strategy: statically partition the map by drawing a perpendicular
        bisector between the two robots' HOME (initial) positions. Each robot
        prefers the half of the map on its own side. This ensures stable
        regions that don't flip when robots get close to each other.

        If we can't determine regions (e.g. a robot position unknown), return 0.
        """
        other_name = 'robot2' if robot_name == 'robot1' else 'robot1'
        curr_self = positions.get(robot_name)
        curr_other = positions.get(other_name)

        if curr_self is None or curr_other is None:
            return 0.0

        # Update locked home positions until map_merge successfully aligns them
        # (they start at (0,0) before map_merge feature-matches their maps)
        if np.linalg.norm(curr_self - curr_other) > 1.0:
            if robot_name not in self.home_positions:
                self.home_positions[robot_name] = curr_self
            if other_name not in self.home_positions:
                self.home_positions[other_name] = curr_other

        if robot_name not in self.home_positions or other_name not in self.home_positions:
            return 0.0

        pos_self = self.home_positions[robot_name]
        pos_other = self.home_positions[other_name]

        # Vector from self to other robot
        midpoint = (pos_self + pos_other) / 2.0
        
        # Signed distance: positive means frontier is on other robot's side
        to_other = pos_other - pos_self
        to_other_norm = np.linalg.norm(to_other)
        if to_other_norm < 0.1:
            return 0.0  # robots are on top of each other, can't partition
        to_other_unit = to_other / to_other_norm

        # Project frontier relative to the midpoint onto the self->other axis
        frontier_offset = frontier_pt - midpoint
        projection = np.dot(frontier_offset, to_other_unit)

        # Positive projection = frontier is on the OTHER robot's side -> penalize
        return max(0.0, projection) * self.region_weight

    # ══════════════════════════════════════════════════════════════════════
    # Timer-based replanning
    # ══════════════════════════════════════════════════════════════════════
    def _replan(self):
        if self.latest_map is None:
            return

        # Ensure all robots have a valid TF before we assign ANY frontiers
        # This prevents one robot from stealing frontiers while the other is starting
        if not self.all_robots_ready:
            ready_count = 0
            for name in self.robot_names:
                if self._get_robot_position(name) is not None:
                    ready_count += 1
            if ready_count < len(self.robot_names):
                self.get_logger().warn(
                    "Waiting for all robots to have valid TF positions before assigning frontiers...",
                    throttle_duration_sec=3.0)
                return
            self.all_robots_ready = True
            self.get_logger().info("All robots have valid TF positions! Starting assignment.")

        # 1. Detect & deduplicate frontiers
        frontiers = self._detect_frontiers(self.latest_map)
        frontiers = self._deduplicate(frontiers, self.dedup_radius)

        # Filter out blacklisted frontiers (e.g. Nav2 rejected them previously)
        valid_frontiers = []
        current_time = self.get_clock().now().nanoseconds / 1e9
        for f in frontiers:
            is_blacklisted = False
            for bf, timestamp in list(self.blacklisted_frontiers.items()):
                if current_time - timestamp > self.blacklist_duration:
                    del self.blacklisted_frontiers[bf]
                    continue
                if np.linalg.norm(f - np.array(bf)) < 1.0:
                    is_blacklisted = True
                    break
            if not is_blacklisted:
                valid_frontiers.append(f)
        frontiers = valid_frontiers

        if not frontiers:
            self.get_logger().info(
                "Exploration complete – no frontiers remain.",
                throttle_duration_sec=30)
            self._publish_frontier_markers([])
            self._publish_goal_markers()
            return

        # 2. Get current robot positions
        positions = {}
        for name in self.robot_names:
            pos = self._get_robot_position(name)
            if pos is not None:
                positions[name] = pos

        if not positions:
            self.get_logger().warn(
                "No robot positions available, skipping replan.",
                throttle_duration_sec=10)
            return

        # 3. Determine which robots need a new goal
        robots_needing_goal = []
        for name in self.robot_names:
            if name not in positions:
                continue
            ctx = self.robots[name]
            cur_pos = positions[name]

            # Stuck detection
            if ctx.goal is not None and ctx.last_position is not None:
                moved = np.linalg.norm(cur_pos - ctx.last_position)
                if moved < 0.1:
                    ctx.stuck_counter += 1
                else:
                    ctx.stuck_counter = 0

                if ctx.stuck_counter >= self.stuck_threshold:
                    self.get_logger().info(
                        f"{name} stuck ({ctx.stuck_counter} ticks). "
                        "Cancelling and blacklisting goal.")
                    # Blacklist it, or the same (cheapest) frontier is
                    # reassigned immediately and the robot stays stuck.
                    self.blacklisted_frontiers[tuple(ctx.goal)] = \
                        self.get_clock().now().nanoseconds / 1e9
                    self._cancel_nav_goal(name)
                    ctx.goal = None
                    ctx.state = RobotState.IDLE
                    ctx.stuck_counter = 0

            ctx.last_position = cur_pos

            # Check if current goal is still valid
            if ctx.goal is not None:
                dist_to_goal = np.linalg.norm(cur_pos - ctx.goal)
                
                time_navigating = 0.0
                if ctx.goal_start_time is not None:
                    time_navigating = (self.get_clock().now() - ctx.goal_start_time).nanoseconds / 1e9

                still_frontier = any(
                    np.linalg.norm(ctx.goal - f) < self.goal_retention_radius
                    for f in frontiers)
                
                if dist_to_goal > self.goal_reached_threshold and (
                    still_frontier or time_navigating < self.min_commitment_time):
                    continue  # keep navigating to current goal

                # Goal reached or frontier disappeared
                self._cancel_nav_goal(name)
                ctx.goal = None
                ctx.state = RobotState.IDLE

            robots_needing_goal.append(name)

        if not robots_needing_goal:
            self._publish_frontier_markers(frontiers)
            self._publish_goal_markers()
            return

        # 4. Remove frontiers already assigned to robots keeping their goal
        available = list(frontiers)
        for name in self.robot_names:
            ctx = self.robots[name]
            if name not in robots_needing_goal and ctx.goal is not None:
                available = [f for f in available
                             if np.linalg.norm(f - ctx.goal) > self.dedup_radius]

        # 5. Cost-based assignment with spatial partitioning
        for name in robots_needing_goal:
            if not available:
                break
            pos = positions[name]

            # Compute cost for each available frontier
            costs = []
            for i, f in enumerate(available):
                # Base cost: distance to frontier
                dist_cost = np.linalg.norm(pos - f)

                # Separation cost: penalize if this frontier is close to
                # the OTHER robot's current goal
                sep_cost = 0.0
                for other_name in self.robot_names:
                    if other_name == name:
                        continue
                    other_ctx = self.robots[other_name]
                    if other_ctx.goal is not None:
                        d_other = np.linalg.norm(f - other_ctx.goal)
                        # Inverse: closer to other's goal = higher penalty
                        sep_cost += self.separation_weight / max(d_other, 0.1)

                # Region ownership cost
                reg_cost = self._region_cost(name, f, positions)

                # Skip frontiers that are too close (already observed area)
                if dist_cost < self.min_goal_distance:
                    costs.append(float('inf'))
                else:
                    costs.append(dist_cost + sep_cost + reg_cost)

            if not costs or all(c == float('inf') for c in costs):
                # Fall back: pick the farthest frontier
                dists = [np.linalg.norm(pos - f) for f in available]
                idx = int(np.argmax(dists))
            else:
                idx = int(np.argmin(costs))

            target = available.pop(idx)
            ctx = self.robots[name]
            ctx.goal = target
            ctx.state = RobotState.NAVIGATING
            ctx.goal_start_time = self.get_clock().now()
            ctx.stuck_counter = 0

            # Send goal via Nav2 action client
            self._send_nav_goal(name, target)

            self.get_logger().info(
                f"Assigned {name} -> ({target[0]:.2f}, {target[1]:.2f})")

        # 6. Publish visualisation markers
        self._publish_frontier_markers(frontiers)
        self._publish_goal_markers()

    # ══════════════════════════════════════════════════════════════════════
    # Nav2 action client
    # ══════════════════════════════════════════════════════════════════════
    def _cancel_nav_goal(self, robot_name):
        """Explicitly cancel the Nav2 action to prevent ghost navigation."""
        ctx = self.robots[robot_name]
        if ctx.goal_handle is not None:
            try:
                ctx.goal_handle.cancel_goal_async()
            except Exception:
                pass
            ctx.goal_handle = None
            
        # Send zero velocity to immediately halt
        stop = Twist()
        self.cmd_pubs[robot_name].publish(stop)

    def _send_nav_goal(self, robot_name, target):
        """Send a NavigateToPose goal to the robot's Nav2 stack."""
        client = self.nav_clients[robot_name]

        if not client.server_is_ready():
            self.get_logger().warn(
                f"Nav2 action server not ready for {robot_name}, skipping.",
                throttle_duration_sec=10)
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = float(target[0])
        goal_msg.pose.pose.position.y = float(target[1])
        goal_msg.pose.pose.orientation.w = 1.0

        ctx = self.robots[robot_name]

        # Cancel previous goal if any
        if ctx.goal_handle is not None:
            ctx.goal_handle.cancel_goal_async()

        future = client.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda f, name=robot_name: self._goal_response_callback(f, name))

    def _goal_response_callback(self, future, robot_name):
        """Handle Nav2 goal acceptance/rejection."""
        goal_handle = future.result()
        ctx = self.robots[robot_name]

        if not goal_handle.accepted:
            self.get_logger().warn(f"Nav2 rejected goal for {robot_name}, blacklisting frontier temporarily.")
            if ctx.goal is not None:
                self.blacklisted_frontiers[tuple(ctx.goal)] = self.get_clock().now().nanoseconds / 1e9
            ctx.state = RobotState.IDLE
            ctx.goal = None
            return

        ctx.goal_handle = goal_handle
        ctx.state = RobotState.NAVIGATING

        # Listen for result — capture goal_handle so we can detect stale callbacks
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, name=robot_name, gh=goal_handle: self._goal_result_callback(f, name, gh))

    def _goal_result_callback(self, future, robot_name, original_goal_handle):
        """Handle Nav2 goal completion / failure."""
        ctx = self.robots[robot_name]

        # Ignore stale callbacks from preempted goals
        if ctx.goal_handle is not original_goal_handle:
            return

        result = future.result()

        if result.status == 4:  # SUCCEEDED
            self.get_logger().info(f"{robot_name} reached its goal!")
        elif result.status == 5:  # CANCELED
            self.get_logger().info(f"{robot_name} goal was canceled.")
        else:
            self.get_logger().warn(
                f"{robot_name} goal failed (status={result.status}). "
                "Blacklisting temporarily and will reassign on next replan.")
            if ctx.goal is not None:
                self.blacklisted_frontiers[tuple(ctx.goal)] = self.get_clock().now().nanoseconds / 1e9

        ctx.goal_handle = None
        ctx.goal = None
        ctx.state = RobotState.IDLE

    # ══════════════════════════════════════════════════════════════════════
    # Visualisation markers
    # ══════════════════════════════════════════════════════════════════════
    def _publish_frontier_markers(self, frontiers):
        """Publish spheres for each detected frontier centroid, colored by assignment."""
        ma = MarkerArray()

        # Delete old markers
        delete = Marker()
        delete.action = Marker.DELETEALL
        delete.ns = 'frontiers'
        ma.markers.append(delete)

        if not frontiers:
            self.frontier_marker_pub.publish(ma)
            return

        for i, f in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'frontiers'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(f[0])
            m.pose.position.y = float(f[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = 0.25
            m.scale.y = 0.25
            m.scale.z = 0.25
            
            # Check if this frontier is assigned to a robot
            assigned_robot = None
            for name in self.robot_names:
                ctx = self.robots[name]
                if ctx.goal is not None and np.linalg.norm(f - ctx.goal) < 0.1:
                    assigned_robot = name
                    break
            
            if assigned_robot:
                m.color.r = ROBOT_COLORS[assigned_robot].r
                m.color.g = ROBOT_COLORS[assigned_robot].g
                m.color.b = ROBOT_COLORS[assigned_robot].b
                m.color.a = ROBOT_COLORS[assigned_robot].a
                text_label = f"[Target: {assigned_robot}]"
            else:
                # Gray for unassigned
                m.color.r = 0.5
                m.color.g = 0.5
                m.color.b = 0.5
                m.color.a = 0.9
                text_label = "[Unassigned]"
                
            m.lifetime.sec = int(self.replan_period + 2)
            ma.markers.append(m)

            # Explicit Text Label hovering over frontier
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = m.header.stamp
            # CRITICAL: Use the exact same namespace so RViz automatically displays it
            label.ns = 'frontiers'
            label.id = i + 1000  # Offset ID to avoid conflict with sphere
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(f[0])
            label.pose.position.y = float(f[1])
            label.pose.position.z = 0.4  # Hover above the sphere
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25  # Large text height
            label.color.r = m.color.r
            label.color.g = m.color.g
            label.color.b = m.color.b
            label.color.a = 1.0  # Make text fully opaque
            label.text = text_label
            label.lifetime.sec = int(self.replan_period + 2)
            ma.markers.append(label)

        self.frontier_marker_pub.publish(ma)

    def _publish_goal_markers(self):
        """Publish an arrow marker at each robot's assigned goal."""
        ma = MarkerArray()

        # Delete old goal markers
        delete = Marker()
        delete.action = Marker.DELETEALL
        delete.ns = 'goals'
        ma.markers.append(delete)

        for i, name in enumerate(self.robot_names):
            ctx = self.robots[name]
            if ctx.goal is None:
                continue

            color = ROBOT_COLORS.get(name, ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))

            # Goal arrow
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'goals'
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = float(ctx.goal[0])
            m.pose.position.y = float(ctx.goal[1])
            m.pose.position.z = 0.3
            # Point arrow downward
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.7071068
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 0.7071068
            m.scale.x = 0.4   # shaft length
            m.scale.y = 0.08  # shaft diameter
            m.scale.z = 0.08  # head diameter
            m.color = color
            m.lifetime.sec = int(self.replan_period + 2)
            ma.markers.append(m)

            # Goal label
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = self.get_clock().now().to_msg()
            label.ns = 'goal_labels'
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(ctx.goal[0])
            label.pose.position.y = float(ctx.goal[1])
            label.pose.position.z = 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.15  # text height
            label.color = color
            state_label = ctx.state.name
            label.text = f"{name}\n[{state_label}]"
            label.lifetime.sec = int(self.replan_period + 2)
            ma.markers.append(label)

        self.goal_marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotFrontierCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
