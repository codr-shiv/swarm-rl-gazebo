"""
Runs one exploration episode on the real Gazebo/SLAM/Nav2 stack with a
given set of formula weights, and scores it.

Episode: a fresh headless sim (world from the given seed) is launched, both
robots pick frontiers with the formula (features.choose) whenever they need
a goal, and the episode ends when no frontiers remain ('explored'), when the
map stops growing ('saturated': < SATURATION_MIN_M2 new in
SATURATION_WINDOW_S), or at the sim-time limit ('time_limit').

Score (higher is better, what tune_weights maximises):
  + SCORE_PER_M2    * mapped area (m²) gained in the merged map
  - TIME_PENALTY    * sim seconds until the episode ended
  - FAIL_PENALTY    per goal that failed / was rejected / got stuck / timed out

Every weight set maps about the same total area (the arena saturates), so
the score is really "how fast, and with how few wasted trips": ending
sooner means less time penalty. Coverage-increase scoring as in Active
Neural SLAM (Chaplot et al. 2020).
"""
import atexit
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from multi_robot_exploration.tuning.features import choose
from multi_robot_exploration.tuning.ros_interface import ROBOTS, ExplorationInterface
from multi_robot_exploration.tuning.sim_manager import SimManager

SCORE_PER_M2 = 0.1
TIME_PENALTY = 0.005
FAIL_PENALTY = 0.2
BAD_EVENTS = ('failed', 'rejected', 'stuck', 'timeout')

EXPLORED_GRACE_S = 5.0      # frontiers must stay gone this long (sim s) to end the episode
SATURATION_WINDOW_S = 60.0  # episode ends if the map grew less than
SATURATION_MIN_M2 = 0.5     # this much over the last window (sim s)
STALL_TIMEOUT_S = 60.0      # wall seconds without sim-clock progress => sim is dead
WALL_FACTOR = 3.0           # an episode may take at most this x its sim-time limit
                            # (+60 s) in wall time; slower means an overloaded machine
POLL_PERIOD_S = 0.25


class SimulationError(RuntimeError):
    pass


class EpisodeRunner:
    """Owns one sim slot (ROS domain + Gazebo port); runs episodes one after another."""

    def __init__(self, instance_id=0, gui=False, rtf=1.0, max_episode_sim_s=300.0,
                 log_dir='/tmp/tuning_sim_logs', ready_timeout_s=240.0, goal_fix=True):
        self.max_episode_sim_s = max_episode_sim_s
        self.ready_timeout_s = ready_timeout_s
        self.goal_fix = goal_fix
        self.sim = SimManager(instance_id, gui=gui, rtf=rtf, log_dir=log_dir)
        atexit.register(self.sim.stop)   # never leave a headless sim running

        self.context = rclpy.Context()
        # No rclpy signal handlers: Ctrl+C stays a normal KeyboardInterrupt.
        rclpy.init(context=self.context, domain_id=self.sim.domain_id,
                   signal_handler_options=SignalHandlerOptions.NO)
        self.node = self.executor = self.spin_thread = None
        self.ready_status = {}
        self._episodes = 0

    # ── public ───────────────────────────────────────────────────────────
    def run(self, weights, world_seed):
        """Run one episode with *weights* (dict or vector) in world *world_seed*; returns stats."""
        decisions = failed = 0
        turn = 0
        try:   # the sim is torn down on every exit path, including startup errors
            self._start_episode(world_seed)
            n = self.node
            start, area0 = n.sim_now(), n.known_area_m2()
            self.area_history = [(start, area0)]
            self.wall_deadline = time.time() + WALL_FACTOR * self.max_episode_sim_s + 60.0
            while True:
                end, pending, events = self._advance(start, turn)
                failed += sum(1 for e in events.values() if e in BAD_EVENTS)
                if end is not None:
                    break
                robot, (candidates, terms, mask) = pending
                n.send_goal(robot, candidates[choose(terms, mask, weights)])
                decisions += 1
                turn += 1
            sim_time = n.sim_now() - start
            area = n.known_area_m2()
            n.cancel_all()
        finally:
            self._teardown()
        if end == 'sim_died':
            raise SimulationError(f'sim died or froze during episode (seed {world_seed})')
        if end == 'too_slow':
            raise SimulationError(f'episode exceeded {WALL_FACTOR:g}x real time (seed {world_seed}); '
                                  'the machine is overloaded, use fewer --num-envs')
        score = SCORE_PER_M2 * (area - area0) - TIME_PENALTY * sim_time - FAIL_PENALTY * failed
        return {'world_seed': world_seed, 'score': score, 'end_reason': end,
                'explored_m2': area, 'sim_time_s': sim_time,
                'decisions': decisions, 'failed_goals': failed}

    def close(self):
        self._teardown()
        if self.context.ok():
            rclpy.shutdown(context=self.context)

    # ── sim / node lifecycle ─────────────────────────────────────────────
    def _start_episode(self, world_seed):
        self._teardown()
        for attempt in range(3):
            self.sim.start(world_seed)
            self._start_node()
            if self._wait_ready():
                return
            print(f'[slot {self.sim.instance_id}] sim not ready (attempt {attempt + 1}): '
                  f'{self.ready_status}; restarting', flush=True)
            self._teardown()
        raise SimulationError('Simulation failed to come up 3 times; see sim_*.log in log_dir')

    def _start_node(self):
        # Unique name per episode (avoids "Publisher already registered" warnings)
        self._episodes += 1
        self.node = ExplorationInterface(f'frontier_tuning_{self._episodes}', context=self.context)
        self.node.goal_fix = self.goal_fix
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def _stop_node(self):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2.0)
            self.spin_thread.join(timeout=5.0)
        if self.node is not None:
            self.node.destroy_node()
        self.node = self.executor = self.spin_thread = None

    def _teardown(self):
        self._stop_node()
        self.sim.stop()

    def _wait_ready(self):
        """Map, both robot poses and both Nav2 stacks active."""
        deadline = time.time() + self.ready_timeout_s
        active = {r: False for r in ROBOTS}
        status = {}
        while time.time() < deadline:
            if not self.sim.alive():
                self.ready_status = {'sim_process': 'exited'}
                return False
            n = self.node
            for r in ROBOTS:
                if not active[r] and n.nav_clients[r].server_is_ready():
                    active[r] = bool(n.nav2_active(r))
            status = {'nav2_active': dict(active), 'map': n.latest_map is not None,
                      'tf': {r: n.robot_pose(r) is not None for r in ROBOTS}}
            if all(active.values()) and status['map'] and all(status['tf'].values()):
                return True
            time.sleep(1.0)
        self.ready_status = status
        return False

    # ── episode loop ─────────────────────────────────────────────────────
    def _saturated(self, now, area):
        """True if the map grew < SATURATION_MIN_M2 over the last SATURATION_WINDOW_S."""
        hist = self.area_history
        if now - hist[-1][0] >= 1.0:
            hist.append((now, area))
        if now - hist[0][0] < SATURATION_WINDOW_S:
            return False
        while len(hist) > 1 and now - hist[1][0] >= SATURATION_WINDOW_S:
            hist.pop(0)
        return area - hist[0][1] < SATURATION_MIN_M2

    def _advance(self, start, turn):
        """
        Run the sim until a robot needs a goal or the episode ends.
        Returns (end_reason or None, (robot, decision) or None, goal events seen).
        """
        n = self.node
        events = {}
        no_frontier_since = None
        last_sim, last_progress = n.sim_now(), time.time()
        while True:
            n.update_homes()
            pts, sizes = n.frontiers()
            events.update(n.poll_events(pts))
            now = n.sim_now()

            if now != last_sim:
                last_sim, last_progress = now, time.time()
            elif time.time() - last_progress > STALL_TIMEOUT_S or not self.sim.alive():
                return 'sim_died', None, events
            if time.time() > self.wall_deadline:
                return 'too_slow', None, events
            if now - start >= self.max_episode_sim_s:
                return 'time_limit', None, events
            if self._saturated(now, n.known_area_m2()):
                return 'saturated', None, events
            if pts:
                no_frontier_since = None
            elif no_frontier_since is None:
                no_frontier_since = now
            elif now - no_frontier_since > EXPLORED_GRACE_S:
                return 'explored', None, events

            # Alternate which robot is asked first so neither is favoured
            for r in (ROBOTS if turn % 2 == 0 else ROBOTS[::-1]):
                if n.is_idle(r):
                    d = n.decision(r, pts, sizes)
                    if d is not None and len(d[0]) and d[2].any():
                        return None, (r, d), events
            time.sleep(POLL_PERIOD_S)
