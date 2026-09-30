"""
ROS side of RL frontier selection, shared by the training env and the
deployment node: merged map, robot poses (TF), Nav2 goals, and the events
that mean "this robot needs a new frontier".

Goal lifecycle mirrors frontier_coordinator.py (reach threshold, stuck
detection, frontier-vanished check, temporary blacklist), so the RL policy
is judged under the same rules as the heuristic.
"""
import math
import threading

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from multi_robot_exploration.frontier_utils import (deduplicate, detect_frontiers,
                                                     occupancy_grid_to_array)
from multi_robot_exploration.rl.features import RobotView, build_observation

ROBOTS = ('robot1', 'robot2')

# Goal lifecycle knobs (same as frontier_coordinator.py)
MIN_FRONTIER_SIZE = 15
DEDUP_RADIUS = 1.5
GOAL_REACHED = 0.4
GOAL_RETENTION_RADIUS = 2.5
MIN_COMMITMENT_S = 10.0
STUCK_WINDOW_S = 9.0        # coordinator: 3 replan ticks x 3 s
STUCK_MOVE_M = 0.1
GOAL_TIMEOUT_S = 90.0
BLACKLIST_S = 30.0


class _RobotState:
    def __init__(self):
        self.goal = None
        self.goal_handle = None
        self.goal_start = None
        self.anchor_pos = None       # position at the start of the stuck window
        self.anchor_time = None
        self.done_reason = None      # set by action callbacks
        self.token = None            # identifies the live goal; stale callbacks are ignored
        self.home = None


class ExplorationInterface(Node):

    def __init__(self, node_name='rl_exploration', **kwargs):
        super().__init__(node_name, **kwargs)
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self._lock = threading.RLock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.latest_map = None
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, qos)

        self.nav_clients = {r: ActionClient(self, NavigateToPose, f'/{r}/navigate_to_pose')
                            for r in ROBOTS}
        self.state_clients = {r: self.create_client(GetState, f'/{r}/bt_navigator/get_state')
                              for r in ROBOTS}
        self.robots = {r: _RobotState() for r in ROBOTS}
        self.blacklist = {}          # (x, y) -> sim time
        self.episode_start = None
        self.goal_fix = True         # nudge goals into safe free space (features.safe_goals)

    # ── basic queries ────────────────────────────────────────────────────
    def _map_cb(self, msg):
        self.latest_map = msg

    def sim_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def robot_pose(self, name):
        """(np.array([x, y]), yaw) in the merged map frame, or None."""
        try:
            t = self.tf_buffer.lookup_transform('map', f'{name}/base_footprint',
                                                rclpy.time.Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return np.array([t.transform.translation.x, t.transform.translation.y]), yaw

    def known_area_m2(self):
        m = self.latest_map
        if m is None:
            return 0.0
        known = np.count_nonzero(occupancy_grid_to_array(m) >= 0)
        return known * m.info.resolution ** 2

    def frontiers(self):
        """Deduplicated (points, sizes) from the latest merged map."""
        if self.latest_map is None:
            return [], []
        pts, sizes = detect_frontiers(self.latest_map, MIN_FRONTIER_SIZE)
        return deduplicate(pts, DEDUP_RADIUS, sizes)

    def nav2_active(self, name):
        """Non-blocking: returns True/False, or None while the answer is pending."""
        client = self.state_clients[name]
        if not client.service_is_ready():
            return False
        fut = getattr(self, f'_state_future_{name}', None)
        if fut is None:
            setattr(self, f'_state_future_{name}', client.call_async(GetState.Request()))
            return None
        if not fut.done():
            return None
        setattr(self, f'_state_future_{name}', None)
        res = fut.result()
        return res is not None and res.current_state.id == 3   # PRIMARY_STATE_ACTIVE

    # ── goals ────────────────────────────────────────────────────────────
    def is_idle(self, name):
        with self._lock:
            return self.robots[name].goal is None

    def send_goal(self, name, target):
        client = self.nav_clients[name]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(target[0])
        goal.pose.pose.position.y = float(target[1])
        goal.pose.pose.orientation.w = 1.0

        now = self.sim_now()
        pose = self.robot_pose(name)
        with self._lock:
            st = self.robots[name]
            st.goal = np.asarray(target, dtype=float)
            st.goal_handle = None
            st.goal_start = now
            st.anchor_pos = pose[0] if pose else None
            st.anchor_time = now
            st.done_reason = None
            token = object()
            st.token = token
        fut = client.send_goal_async(goal)
        fut.add_done_callback(lambda f: self._goal_response(f, name, token))

    def _goal_response(self, fut, name, token):
        handle = fut.result()
        with self._lock:
            st = self.robots[name]
            if getattr(st, 'token', None) is not token:
                return
            if handle is None or not handle.accepted:
                st.done_reason = 'rejected'
                return
            st.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f: self._goal_result(f, name, token))

    def _goal_result(self, fut, name, token):
        status = fut.result().status
        with self._lock:
            st = self.robots[name]
            if getattr(st, 'token', None) is not token:
                return
            st.done_reason = {GoalStatus.STATUS_SUCCEEDED: 'succeeded',
                              GoalStatus.STATUS_CANCELED: 'canceled'}.get(status, 'failed')

    def clear_goal(self, name, blacklist=False):
        with self._lock:
            st = self.robots[name]
            if blacklist and st.goal is not None:
                self.blacklist[tuple(st.goal)] = self.sim_now()
            if st.goal_handle is not None:
                try:
                    st.goal_handle.cancel_goal_async()
                except Exception:
                    pass
            st.goal = None
            st.goal_handle = None
            st.token = None

    def cancel_all(self):
        for r in ROBOTS:
            self.clear_goal(r)

    # ── events ───────────────────────────────────────────────────────────
    def poll_events(self, frontier_points):
        """
        Close finished goals and return {robot: reason} for every robot that
        just became idle. Reasons: succeeded, reached, failed, rejected,
        canceled, stuck, timeout, vanished.
        """
        now = self.sim_now()
        for t in [k for k, v in self.blacklist.items() if now - v > BLACKLIST_S]:
            del self.blacklist[t]

        events = {}
        for name in ROBOTS:
            with self._lock:
                st = self.robots[name]
                if st.goal is None:
                    continue
                goal, reason = st.goal, st.done_reason
                goal_start, anchor_pos, anchor_time = st.goal_start, st.anchor_pos, st.anchor_time
            pose = self.robot_pose(name)

            if reason is None and pose is not None:
                pos = pose[0]
                if np.linalg.norm(pos - goal) <= GOAL_REACHED:
                    reason = 'reached'
                elif now - goal_start > GOAL_TIMEOUT_S:
                    reason = 'timeout'
                elif (now - goal_start > MIN_COMMITMENT_S and
                      not any(np.linalg.norm(goal - f) < GOAL_RETENTION_RADIUS
                              for f in frontier_points)):
                    reason = 'vanished'
                elif anchor_pos is None or np.linalg.norm(pos - anchor_pos) >= STUCK_MOVE_M:
                    with self._lock:
                        st.anchor_pos, st.anchor_time = pos, now
                elif now - anchor_time > STUCK_WINDOW_S:
                    reason = 'stuck'

            if reason is not None:
                self.clear_goal(name, blacklist=reason in ('failed', 'rejected', 'stuck', 'timeout'))
                events[name] = reason
        return events

    # ── observations ─────────────────────────────────────────────────────
    def update_homes(self):
        """Lock start positions once both robots are >1 m apart (as the coordinator does)."""
        poses = {r: self.robot_pose(r) for r in ROBOTS}
        if all(poses.values()) and np.linalg.norm(poses['robot1'][0] - poses['robot2'][0]) > 1.0:
            for r in ROBOTS:
                if self.robots[r].home is None:
                    self.robots[r].home = poses[r][0]

    def robot_view(self, name):
        pose = self.robot_pose(name)
        if pose is None:
            return None
        with self._lock:
            st = self.robots[name]
            return RobotView(pos=pose[0], yaw=pose[1], goal=st.goal, home=st.home)

    def decision(self, name, frontiers, sizes, max_episode_s):
        """(obs, mask, candidates) for *name*, or None if poses/map are missing."""
        other = 'robot2' if name == 'robot1' else 'robot1'
        ego_v, other_v = self.robot_view(name), self.robot_view(other)
        if ego_v is None or other_v is None or self.latest_map is None:
            return None
        start = self.episode_start if self.episode_start is not None else self.sim_now()
        elapsed = min((self.sim_now() - start) / max_episode_s, 1.0)
        m = self.latest_map
        return build_observation(frontiers, sizes, occupancy_grid_to_array(m), m.info,
                                 ego_v, other_v, self.known_area_m2(), elapsed,
                                 blacklist=list(self.blacklist.keys()),
                                 goal_fix=self.goal_fix)
