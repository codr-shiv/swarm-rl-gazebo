"""
Gymnasium environment: RL frontier selection on the real Gazebo/SLAM/Nav2 stack.

Episode: a fresh headless sim (new random world) is launched, and the
episode ends when no frontiers remain (explored) or the sim-time limit hits.

Step: one frontier decision for one robot. The agent picks a candidate
index for whichever robot needs a goal, the goal goes to that robot's Nav2,
and the simulation runs (both robots keep driving) until some robot needs
a new goal again. This is a semi-MDP: steps take variable sim time, and the
time penalty makes slow choices cost more.

Reward per step (cooperative, shared by both robots):
  + REWARD_PER_M2   * newly mapped area (m²) in the merged map
  - TIME_PENALTY    * sim seconds elapsed
  - FAIL_PENALTY    per goal that failed / was rejected / got stuck / timed out
  + DONE_BONUS      when the map is fully explored
"""
import atexit
import threading
import time

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from multi_robot_exploration.rl.features import MAX_CANDIDATES, OBS_DIM
from multi_robot_exploration.rl.ros_interface import ROBOTS, ExplorationInterface
from multi_robot_exploration.rl.sim_manager import SimManager

REWARD_PER_M2 = 1.0
TIME_PENALTY = 0.02
FAIL_PENALTY = 1.0
DONE_BONUS = 10.0
BAD_EVENTS = ('failed', 'rejected', 'stuck', 'timeout')

EXPLORED_GRACE_S = 5.0      # frontiers must stay gone this long (sim s) to end the episode
STALL_TIMEOUT_S = 60.0      # wall seconds without sim-clock progress => sim is dead
POLL_PERIOD_S = 0.25


class GazeboExplorationEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, instance_id=0, gui=False, rtf=1.0, max_episode_sim_s=600.0,
                 world_seeds=None, log_dir='/tmp/rl_sim_logs', ready_timeout_s=240.0):
        super().__init__()
        self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(MAX_CANDIDATES)

        self.max_episode_sim_s = max_episode_sim_s
        self.world_seeds = list(world_seeds) if world_seeds else None
        self.ready_timeout_s = ready_timeout_s
        self.sim = SimManager(instance_id, gui=gui, rtf=rtf, log_dir=log_dir)
        atexit.register(self.sim.stop)   # never leave a headless sim running

        self.context = rclpy.Context()
        # No rclpy signal handlers: Ctrl+C stays a normal KeyboardInterrupt
        # so train/evaluate can save and tear the sims down themselves.
        rclpy.init(context=self.context, domain_id=self.sim.domain_id,
                   signal_handler_options=SignalHandlerOptions.NO)
        self.node = None
        self.executor = None
        self.spin_thread = None

        self.episode = 0
        self.world_seed = None
        self.decisions = 0
        self.fail_count = 0
        self.pending = None          # (robot, (obs, mask, candidates)) awaiting an action
        self._last_events = {}
        self.ready_status = {}
        self._turn = 0

    # ── node / sim lifecycle ─────────────────────────────────────────────
    def _start_node(self):
        self.node = ExplorationInterface(context=self.context)
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

    # ── gym API ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._teardown()
        for attempt in range(3):
            if self.world_seeds:
                world_seed = self.world_seeds[self.episode % len(self.world_seeds)]
            else:
                world_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.sim.start(world_seed)
            self._start_node()
            if self._wait_ready():
                break
            print(f'[env {self.sim.instance_id}] sim not ready (attempt {attempt + 1}): '
                  f'{self.ready_status}; restarting', flush=True)
            self._teardown()
        else:
            raise RuntimeError('Simulation failed to come up 3 times; see sim_*.log in log_dir')

        self.episode += 1
        self.world_seed = world_seed
        self.node.episode_start = self.node.sim_now()
        self.prev_area = self.node.known_area_m2()
        self.prev_time = self.node.sim_now()
        self.decisions = 0
        self.fail_count = 0

        end = self._advance()
        if end is not None:
            # Nothing to decide at all (should not happen on a fresh map)
            raise RuntimeError(f'Episode ended during reset: {end}')
        return self.pending[1][0], self._info()

    def step(self, action):
        robot, (_obs, mask, candidates) = self.pending
        a = int(action)
        if a >= len(candidates) or not mask[a]:
            a = int(np.flatnonzero(mask)[0])     # only reachable without action masking
        self.node.send_goal(robot, candidates[a])
        self.decisions += 1

        end, events = self._advance(), self._last_events
        now, area = self.node.sim_now(), self.node.known_area_m2()
        n_bad = sum(1 for e in events.values() if e in BAD_EVENTS)
        self.fail_count += n_bad
        reward = (REWARD_PER_M2 * (area - self.prev_area)
                  - TIME_PENALTY * (now - self.prev_time)
                  - FAIL_PENALTY * n_bad)
        self.prev_area, self.prev_time = area, now

        terminated = end == 'explored'
        truncated = end in ('time_limit', 'sim_died')
        if terminated:
            reward += DONE_BONUS
        obs = self.pending[1][0] if end is None else np.zeros(OBS_DIM, np.float32)
        info = self._info(end_reason=end, events=events)
        if end is not None:
            self.node.cancel_all()
        return obs, float(reward), terminated, truncated, info

    def action_masks(self):
        if self.pending is None:
            return np.ones(MAX_CANDIDATES, dtype=bool)
        return self.pending[1][1]

    def close(self):
        self._teardown()
        if self.context.ok():
            rclpy.shutdown(context=self.context)

    # ── internals ────────────────────────────────────────────────────────
    def _info(self, **extra):
        n = self.node
        info = {
            'explored_m2': n.known_area_m2() if n else 0.0,
            'sim_time_s': (n.sim_now() - n.episode_start) if n and n.episode_start else 0.0,
            'decisions': self.decisions,
            'failed_goals': self.fail_count,
            'world_seed': self.world_seed,
        }
        if self.pending is not None:
            info['deciding_robot'] = self.pending[0]
        info.update(extra)
        return info

    def _advance(self):
        """
        Run the sim until a robot needs a decision (sets self.pending, returns
        None) or the episode ends (returns 'explored' / 'time_limit' / 'sim_died').
        Goal events seen on the way are stored in self._last_events.
        """
        n = self.node
        self.pending = None
        self._last_events = {}
        no_frontier_since = None
        last_sim, last_progress = n.sim_now(), time.time()

        while True:
            n.update_homes()
            pts, sizes = n.frontiers()
            self._last_events.update(n.poll_events(pts))
            now = n.sim_now()

            if now != last_sim:
                last_sim, last_progress = now, time.time()
            elif time.time() - last_progress > STALL_TIMEOUT_S or not self.sim.alive():
                return 'sim_died'
            if now - n.episode_start >= self.max_episode_sim_s:
                return 'time_limit'
            if pts:
                no_frontier_since = None
            elif no_frontier_since is None:
                no_frontier_since = now
            elif now - no_frontier_since > EXPLORED_GRACE_S:
                return 'explored'

            # Alternate which robot is asked first so neither is favoured
            order = ROBOTS if self._turn % 2 == 0 else ROBOTS[::-1]
            for r in order:
                if n.is_idle(r):
                    d = n.decision(r, pts, sizes, self.max_episode_sim_s)
                    if d is not None and d[1].any():
                        self._turn += 1
                        self.pending = (r, d)
                        return None
            time.sleep(POLL_PERIOD_S)
