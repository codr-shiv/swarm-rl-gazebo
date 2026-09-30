"""
The frontier-selection formula whose weights are tuned (pure numpy, no ROS).

When a robot needs a goal, every candidate frontier gets a cost

    cost = Σ_k  w_k · term_k        (all w_k ≥ 0, lowest cost wins)

TERMS (each is "lower is better"):
  distance        metres from this robot to the frontier
  crowding        1 / distance (m) from the frontier to the other robot's goal (0 if none)
  other_half      metres the frontier lies past the bisector of the two start positions
  unknown_around  −(fraction of unknown cells within 1 m)      -> prefers information gain
  frontier_size   −(frontier cells / 100)                       -> prefers big frontiers
  turning         (1 − cos(bearing)) / 2, 0 ahead … 1 behind    -> prefers no U-turns

HEURISTIC_WEIGHTS reproduces frontier_coordinator.py exactly. Only the ratios
matter (scaling all weights doesn't change the argmin), so tuning fixes
distance = 1 and learns the other five.
"""
import math
from dataclasses import dataclass
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

TERMS = ('distance', 'crowding', 'other_half', 'unknown_around', 'frontier_size', 'turning')
HEURISTIC_WEIGHTS = {'distance': 1.0, 'crowding': 50.0, 'other_half': 50.0,
                     'unknown_around': 0.0, 'frontier_size': 0.0, 'turning': 0.0}

MAX_CANDIDATES = 12         # frontiers nearest the robot that are considered
INFO_GAIN_RADIUS = 1.0      # metres, window for unknown_around

# Goal nudge (see safe_goals): Nav2 rejects goals inside inflated obstacles
# ("start or goal pose are an obstacle"), so candidates are moved to the
# nearest known-free cell with this clearance, or dropped if none is close.
SAFE_CLEARANCE = 0.25       # metres from the nearest occupied cell
NUDGE_RADIUS = 0.6          # metres a centroid may be moved

# Same values as frontier_coordinator.py
MIN_GOAL_DISTANCE = 0.6
DEDUP_RADIUS = 1.5


@dataclass
class RobotView:
    pos: np.ndarray                      # (x, y) in the merged map frame
    yaw: float
    goal: Optional[np.ndarray] = None    # current goal, None when idle
    home: Optional[np.ndarray] = None    # locked start position (region split)


def weight_vector(weights):
    """dict {term: w} -> np.array in TERMS order (missing terms = 0)."""
    return np.array([float(weights.get(t, 0.0)) for t in TERMS])


def unknown_fraction_map(grid, resolution):
    """Per-cell fraction of unknown cells in a square window of ±INFO_GAIN_RADIUS."""
    k = 2 * max(1, int(round(INFO_GAIN_RADIUS / resolution))) + 1
    unknown = (grid == -1).astype(np.float32)
    return cv2.boxFilter(unknown, -1, (k, k), normalize=True,
                         borderType=cv2.BORDER_CONSTANT)


def safe_goals(frontiers, sizes, grid, info):
    """
    Move each frontier centroid to the nearest cell (within NUDGE_RADIUS) that
    is known free and at least SAFE_CLEARANCE from any occupied cell; drop
    frontiers with no such cell. The goal moves *into known free space*,
    unlike snapping onto the frontier edge (next to unknown), which was
    measured to halve exploration.
    """
    res = info.resolution
    ox, oy = info.origin.position.x, info.origin.position.y
    h, w = grid.shape
    not_obstacle = (grid < 50).astype(np.uint8)           # distance to nearest occupied cell
    clearance = cv2.distanceTransform(not_obstacle, cv2.DIST_L2, 5) * res
    safe = (grid == 0) & (clearance >= SAFE_CLEARANCE)
    r = max(1, int(round(NUDGE_RADIUS / res)))

    out_f, out_s = [], []
    for f, size in zip(frontiers, sizes):
        cx, cy = (f[0] - ox) / res, (f[1] - oy) / res
        c0, r0 = int(cx), int(cy)
        rows = slice(max(r0 - r, 0), min(r0 + r + 1, h))
        cols = slice(max(c0 - r, 0), min(c0 + r + 1, w))
        rr, cc = np.nonzero(safe[rows, cols])
        if len(rr) == 0:
            continue
        rr = rr + rows.start
        cc = cc + cols.start
        d2 = (cc + 0.5 - cx) ** 2 + (rr + 0.5 - cy) ** 2
        k = int(np.argmin(d2))
        if d2[k] > r * r:
            continue
        out_f.append(np.array([ox + (cc[k] + 0.5) * res, oy + (rr[k] + 0.5) * res]))
        out_s.append(size)
    return out_f, out_s


def region_projection(ego, other, f):
    """Signed distance of f past the bisector of the two home positions (+ = other's side)."""
    if ego.home is None or other.home is None:
        return 0.0
    axis = other.home - ego.home
    n = np.linalg.norm(axis)
    if n < 0.1:
        return 0.0
    return float(np.dot(f - (ego.home + other.home) / 2.0, axis / n))


def build_candidates(frontiers, sizes, grid, info, ego, other, blacklist=(), goal_fix=True):
    """
    Returns (candidates, terms, mask) for the robot *ego*:
      candidates  list of np.array([x, y]) goals (up to MAX_CANDIDATES, nearest first)
      terms       float array (len(candidates), len(TERMS)), the formula's inputs
      mask        bool array, False = not allowed (too close / other robot's goal)
    *info* is the OccupancyGrid.info of *grid* (resolution + origin).
    With goal_fix, candidates are moved into safe free space (safe_goals).
    """
    if goal_fix:
        frontiers, sizes = safe_goals(frontiers, sizes, grid, info)
    res = info.resolution
    ox, oy = info.origin.position.x, info.origin.position.y
    h, w = grid.shape

    items = [(f, s) for f, s in zip(frontiers, sizes)
             if all(np.linalg.norm(f - np.asarray(b)) >= 1.0 for b in blacklist)]
    items.sort(key=lambda fs: float(np.linalg.norm(fs[0] - ego.pos)))
    items = items[:MAX_CANDIDATES]
    candidates = [f for f, _ in items]
    if not candidates:
        return [], np.zeros((0, len(TERMS))), np.zeros(0, dtype=bool)

    # Same exclusions as the coordinator: too close to the robot, or already
    # taken by the other robot. Fall back to everything rather than nothing.
    mask = np.array([
        not (np.linalg.norm(f - ego.pos) < MIN_GOAL_DISTANCE or
             (other.goal is not None and np.linalg.norm(f - other.goal) <= DEDUP_RADIUS))
        for f in candidates])
    if not mask.any():
        mask[:] = True

    gain = unknown_fraction_map(grid, res)
    terms = np.zeros((len(candidates), len(TERMS)))
    for i, (f, size) in enumerate(items):
        col = min(max(int((f[0] - ox) / res), 0), w - 1)
        row = min(max(int((f[1] - oy) / res), 0), h - 1)
        bearing = math.atan2(f[1] - ego.pos[1], f[0] - ego.pos[0]) - ego.yaw
        terms[i] = [
            np.linalg.norm(f - ego.pos),
            (1.0 / max(float(np.linalg.norm(f - other.goal)), 0.1)) if other.goal is not None else 0.0,
            max(0.0, region_projection(ego, other, f)),
            -gain[row, col],
            -size / 100.0,
            (1.0 - math.cos(bearing)) / 2.0,
        ]
    return candidates, terms, mask


def choose(terms, mask, weights):
    """Index of the lowest-cost allowed candidate under *weights* (dict or vector)."""
    w = weight_vector(weights) if isinstance(weights, dict) else np.asarray(weights)
    cost = terms @ w
    cost[~mask] = np.inf
    return int(np.argmin(cost))
