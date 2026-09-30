"""
Frontier detection shared by frontier_coordinator (heuristic) and the weight-tuning
stack, so both see exactly the same frontier candidates.
"""
import cv2  # type: ignore[import-untyped]
import numpy as np


def occupancy_grid_to_array(msg):
    """Return the OccupancyGrid data as an (h, w) int8 array."""
    return np.array(msg.data, dtype=np.int8).reshape(
        (msg.info.height, msg.info.width))


def detect_frontiers(msg, min_frontier_size):
    """
    Return (points, sizes): world-frame (x, y) frontier centroids and the
    pixel area of each frontier cluster.
    """
    res = msg.info.resolution
    ox = msg.info.origin.position.x
    oy = msg.info.origin.position.y

    grid = occupancy_grid_to_array(msg)

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
    sizes = []
    for i in range(1, num_labels):                      # skip background
        if stats[i, cv2.CC_STAT_AREA] > min_frontier_size:
            cx, cy = centroids[i]
            points.append(np.array([ox + cx * res,
                                    oy + cy * res]))
            sizes.append(int(stats[i, cv2.CC_STAT_AREA]))

    return points, sizes


def deduplicate(frontiers, radius, sizes=None):
    """
    Keep only one centroid per cluster within *radius* metres.
    If *sizes* is given, returns (points, sizes) with the sizes of merged
    clusters summed into the kept one.
    """
    if not frontiers:
        return ([], []) if sizes is not None else []
    unique = [frontiers[0]]
    unique_sizes = [sizes[0]] if sizes is not None else None
    for k, f in enumerate(frontiers[1:], start=1):
        dists = [np.linalg.norm(f - u) for u in unique]
        if all(d > radius for d in dists):
            unique.append(f)
            if sizes is not None:
                unique_sizes.append(sizes[k])
        elif sizes is not None:
            unique_sizes[int(np.argmin(dists))] += sizes[k]
    return (unique, unique_sizes) if sizes is not None else unique
