"""
Observation / action encoding for RL frontier selection (pure numpy, no ROS).

One decision = "which frontier should robot X go to next?". The candidates
are the K frontiers nearest to the deciding robot; the policy outputs an
index into that list (Discrete(K)) and an action mask hides empty or
disallowed slots. Features are ego-centric so a single shared policy works
for both robots.
"""
import math
from dataclasses import dataclass
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

MAX_CANDIDATES = 12
CANDIDATE_FEATURES = 10
GLOBAL_FEATURES = 5
OBS_DIM = MAX_CANDIDATES * CANDIDATE_FEATURES + GLOBAL_FEATURES

DIST_SCALE = 8.0            # metres, ~arena size
AREA_SCALE = 64.0           # m², 8x8 m arena
INFO_GAIN_RADIUS = 1.0      # metres, window for the unknown-fraction feature

# Same values as frontier_coordinator.py so the heuristic baseline matches it
MIN_GOAL_DISTANCE = 0.6
DEDUP_RADIUS = 1.5
SEPARATION_WEIGHT = 50.0
REGION_WEIGHT = 50.0


@dataclass
class RobotView:
    pos: np.ndarray                      # (x, y) in the merged map frame
    yaw: float
    goal: Optional[np.ndarray] = None    # current goal, None when idle
    home: Optional[np.ndarray] = None    # locked start position (region split)


def unknown_fraction_map(grid, resolution):
    """Per-cell fraction of unknown cells in a square window of ±INFO_GAIN_RADIUS."""
    k = 2 * max(1, int(round(INFO_GAIN_RADIUS / resolution))) + 1
    unknown = (grid == -1).astype(np.float32)
    return cv2.boxFilter(unknown, -1, (k, k), normalize=True,
                         borderType=cv2.BORDER_CONSTANT)


def region_projection(ego, other, f):
    """Signed distance of f past the bisector of the two home positions (+ = other's side)."""
    if ego.home is None or other.home is None:
        return 0.0
    axis = other.home - ego.home
    n = np.linalg.norm(axis)
    if n < 0.1:
        return 0.0
    return float(np.dot(f - (ego.home + other.home) / 2.0, axis / n))


def heuristic_cost(ego, other, f):
    """frontier_coordinator.py's cost: distance + separation + region penalty."""
    cost = float(np.linalg.norm(ego.pos - f))
    if other.goal is not None:
        cost += SEPARATION_WEIGHT / max(float(np.linalg.norm(f - other.goal)), 0.1)
    cost += max(0.0, region_projection(ego, other, f)) * REGION_WEIGHT
    return cost


def build_observation(frontiers, sizes, grid, info, ego, other,
                      explored_m2, elapsed_frac, blacklist=()):
    """
    Returns (obs, mask, candidates):
      obs        float32[OBS_DIM]
      mask       bool[MAX_CANDIDATES], True = selectable
      candidates list of np.array([x, y]) aligned with the mask slots
    *info* is the OccupancyGrid.info of *grid* (resolution + origin).
    """
    res = info.resolution
    ox, oy = info.origin.position.x, info.origin.position.y
    h, w = grid.shape

    items = [(f, s) for f, s in zip(frontiers, sizes)
             if all(np.linalg.norm(f - np.asarray(b)) >= 1.0 for b in blacklist)]
    items.sort(key=lambda fs: float(np.linalg.norm(fs[0] - ego.pos)))
    items = items[:MAX_CANDIDATES]
    candidates = [f for f, _ in items]

    # Same exclusions as the coordinator: too close to the robot, or already
    # taken by the other robot. Fall back to everything rather than nothing.
    mask = np.zeros(MAX_CANDIDATES, dtype=bool)
    for i, f in enumerate(candidates):
        too_close = np.linalg.norm(f - ego.pos) < MIN_GOAL_DISTANCE
        taken = other.goal is not None and np.linalg.norm(f - other.goal) <= DEDUP_RADIUS
        mask[i] = not (too_close or taken)
    if candidates and not mask.any():
        mask[:len(candidates)] = True

    gain = unknown_fraction_map(grid, res) if candidates else None
    feats = np.zeros((MAX_CANDIDATES, CANDIDATE_FEATURES), dtype=np.float32)
    for i, (f, size) in enumerate(items):
        d_ego = float(np.linalg.norm(f - ego.pos))
        bearing = math.atan2(f[1] - ego.pos[1], f[0] - ego.pos[0]) - ego.yaw
        col = min(max(int((f[0] - ox) / res), 0), w - 1)
        row = min(max(int((f[1] - oy) / res), 0), h - 1)
        feats[i] = [
            1.0,
            d_ego / DIST_SCALE,
            np.linalg.norm(f - other.pos) / DIST_SCALE,
            (np.linalg.norm(f - other.goal) / DIST_SCALE) if other.goal is not None else 1.5,
            min(size / 100.0, 3.0),
            gain[row, col],
            np.clip(region_projection(ego, other, f) / DIST_SCALE, -1.0, 1.0),
            math.cos(bearing),
            math.sin(bearing),
            min(heuristic_cost(ego, other, f) / 50.0, 3.0),
        ]

    glob = np.array([
        explored_m2 / AREA_SCALE,
        elapsed_frac,
        len(frontiers) / MAX_CANDIDATES,
        1.0 if other.goal is not None else 0.0,
        np.linalg.norm(ego.pos - other.pos) / DIST_SCALE,
    ], dtype=np.float32)

    obs = np.concatenate([feats.ravel(), glob]).astype(np.float32)
    return obs, mask, candidates


def heuristic_action(candidates, mask, ego, other):
    """Index the coordinator would pick (lowest cost among allowed slots)."""
    costs = [heuristic_cost(ego, other, f) if mask[i] else float('inf')
             for i, f in enumerate(candidates)]
    return int(np.argmin(costs))
