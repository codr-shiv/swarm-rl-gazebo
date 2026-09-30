# 🤖 Multi-Robot Autonomous Frontier Exploration

**Two TurtleBot3 robots autonomously exploring unknown environments using coordinated frontier-based exploration with ROS 2 and Nav2.**

> A complete multi-robot autonomy stack: Gazebo simulation → per-robot SLAM → map merging → Nav2 navigation → intelligent frontier coordination with spatial partitioning.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Inventory](#file-inventory)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Deep-Dive](#pipeline-deep-dive)
- [Configuration Reference](#configuration-reference)
- [RViz Visualization](#rviz-visualization)
- [Troubleshooting](#troubleshooting)
- [RL Frontier Selection (experimental)](#rl-frontier-selection-experimental)

---

## Overview

This project implements a **fully autonomous multi-robot exploration system** where two TurtleBot3 Burger robots cooperatively map an unknown environment. The system uses:

- **Frontier-based exploration** — robots identify boundaries between explored and unexplored space and navigate to them
- **Spatial partitioning** — each robot is assigned a region of the map to prevent redundant coverage
- **Greedy cost-based assignment** — each robot takes its cheapest frontier by distance, separation penalty, and region bias
- **Inter-robot avoidance via Nav2** — each robot sees the other in its local costmap and plans around it
- **Custom map merging** — fuses individual SLAM maps into a single global occupancy grid using known spawn transforms
- **Randomized worlds** — a new procedurally generated Gazebo world (walls, obstacles, spawn points) on every launch, unless seeded

---

## Architecture

The system is structured as a **5-layer sequential pipeline**, where each layer builds on the data from the previous one:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE OVERVIEW                            │
│                                                                 │
│  1. SIMULATION        Gazebo world + 2 TurtleBot3 spawns       │
│        ↓                                                        │
│  2. MULTI-ROBOT SLAM  Per-robot slam_toolbox (async)           │
│        ↓              /robot1/map, /robot2/map                  │
│  3. MAP MERGING       Custom map_merge_node                    │
│        ↓              → unified /map                            │
│  4. NAV2 STACK        Per-robot controller, planner, BT nav    │
│        ↓              /{robot}/navigate_to_pose action          │
│  5. FRONTIER COORD    Cost-matrix assignment + spatial regions  │
│                       → sends goals to Nav2 per robot           │
└─────────────────────────────────────────────────────────────────┘
```

### TF Frame Tree

```
                    map (global merged frame)
                   /    \
            robot1/map   robot2/map        ← static transforms (map_merge_node)
               |            |
          robot1/odom   robot2/odom        ← published by Gazebo diff-drive plugin
               |            |
      robot1/base_footprint robot2/base_footprint
               |            |
        robot1/base_link  robot2/base_link ← published by robot_state_publisher
               |            |
        robot1/base_scan  robot2/base_scan ← LiDAR frame
```

### ROS 2 Topic & Action Flow

```
/robot1/scan ─────→ slam_toolbox ─→ /robot1/map ──┐
                                                    ├─→ map_merge_node ─→ /map
/robot2/scan ─────→ slam_toolbox ─→ /robot2/map ──┘
                                                         ↓
                                              frontier_coordinator
                                              (reads /map, assigns goals)
                                                    /         \
                          /robot1/navigate_to_pose   /robot2/navigate_to_pose
                                    ↓                          ↓
                              Nav2 Stack              Nav2 Stack
                              (robot1)                (robot2)
                                    ↓                          ↓
                           /robot1/cmd_vel            /robot2/cmd_vel
```

---

## File Inventory

Every file in this package and what it does:

### Python Nodes (`multi_robot_exploration/`)

| File | Role | Status |
|------|------|--------|
| [`frontier_coordinator.py`](multi_robot_exploration/frontier_coordinator.py) | **Primary brain** — detects frontiers from merged map, assigns robots using cost-matrix with spatial partitioning, manages Nav2 action goals, publishes RViz markers | ✅ Active |
| [`map_merge_node.py`](multi_robot_exploration/map_merge_node.py) | Fuses `/robot1/map` + `/robot2/map` → `/map` using known spawn transforms. Vectorized NumPy rasterization | ✅ Active |
| [`generate_random_world.py`](multi_robot_exploration/generate_random_world.py) | Procedurally generates Gazebo SDF worlds with random box/cylinder obstacles, boundary walls, and collision-free spawn zones | ✅ Active |
| [`waypoint_navigator.py`](multi_robot_exploration/waypoint_navigator.py) | Demo node — sends pre-defined waypoints to showcase map merging without frontier logic | ✅ Demo |
| [`frontier_utils.py`](multi_robot_exploration/frontier_utils.py) | Frontier detection + deduplication shared by the coordinator and the RL stack | ✅ Active |
| [`rl/`](multi_robot_exploration/rl/) | RL frontier selection: Gym env on the headless Gazebo stack, training, evaluation, and the `rl_frontier_coordinator` node — see [docs/rl_training.md](docs/rl_training.md) | 🧪 Experimental |
| [`__init__.py`](multi_robot_exploration/__init__.py) | Package init | — |

### Launch Files (`launch/`)

| File | What it launches | Order |
|------|-----------------|-------|
| [`spawn_two_turtlebots.launch.py`](launch/spawn_two_turtlebots.launch.py) | Gazebo world + 2 namespaced TurtleBot3 robots with patched SDF (namespace-safe TF frames, custom colors) | **1st** |
| [`multi_robot_slam.launch.py`](launch/multi_robot_slam.launch.py) | One `slam_toolbox` (async) per robot, producing `/robot1/map` and `/robot2/map` | **2nd** |
| [`map_merge.launch.py`](launch/map_merge.launch.py) | `map_merge_node` — fuses individual maps into `/map` | **3rd** |
| [`nav2_bringup_multi.launch.py`](launch/nav2_bringup_multi.launch.py) | Full Nav2 stack per robot (controller, planner, behavior, BT navigator, smoother, lifecycle manager with autostart) | **4th** |
| [`frontier_exploration.launch.py`](launch/frontier_exploration.launch.py) | `frontier_coordinator` node — the exploration brain | **5th** |
| [`waypoint_demo.launch.py`](launch/waypoint_demo.launch.py) | `waypoint_navigator` demo (alternative to frontier exploration; needs `USE_TB3_WORLD=1`) | Alt to 5th |
| [`rl_sim_stack.launch.py`](launch/rl_sim_stack.launch.py) | Whole stack (Gazebo + SLAM + merge + Nav2) in one headless launch, no exploration brain — used by RL training | RL |
| [`rl_frontier_exploration.launch.py`](launch/rl_frontier_exploration.launch.py) | `rl_frontier_coordinator` with a trained policy (`model_path:=...`) | Alt to 5th |
| [`explore_multi.launch.py`](launch/explore_multi.launch.py) | `explore_lite` per robot (alternative off-the-shelf exploration, no coordination) | Alt to 5th |

### Configuration (`config/`)

| File | Purpose |
|------|---------|
| [`nav2_params_robot1.yaml`](config/nav2_params_robot1.yaml) | Full Nav2 parameter set for robot1 — DWB local planner, Theta* global planner, recovery behaviors, costmap layers with namespaced frames |
| [`nav2_params_robot2.yaml`](config/nav2_params_robot2.yaml) | Same as robot1 but with `robot2/` frame prefixes |
| [`map_merge_params.yaml`](config/map_merge_params.yaml) | Map merge node params — robot names, spawn offsets (0,0,0 — see file comments for why), publish rate |
| [`multi_robot_exploration.rviz`](config/multi_robot_exploration.rviz) | RViz2 saved config for visualizing both robots, maps, frontiers, and goals |
| [`multi_robot_exploration_cinematic.rviz`](config/multi_robot_exploration_cinematic.rviz) | Cinematic RViz2 config with dark background and neon color scheme |

### Package Metadata

| File | Purpose |
|------|---------|
| [`package.xml`](package.xml) | ROS 2 package manifest — declares dependencies (`gazebo_ros`, `slam_toolbox`, `nav2_bringup`, `tf2_ros`, etc.) |
| [`setup.py`](setup.py) | Python package setup — registers `map_merge_node`, `frontier_coordinator`, `waypoint_navigator` as console_scripts |
| [`setup.cfg`](setup.cfg) | Install script paths for ament_python |
| [`resource/multi_robot_exploration`](resource/multi_robot_exploration) | Ament resource index marker |

### Documentation (`docs/`)

| File | Contents |
|------|----------|
| [`gazebo_visual_guide.md`](docs/gazebo_visual_guide.md) | Guide for disabling LiDAR ray visualization in Gazebo |
| [`rl_training.md`](docs/rl_training.md) | RL frontier selection: design, install, training, evaluation, deployment |
| [`rviz_visual_enhancement_guide.md`](docs/rviz_visual_enhancement_guide.md) | Color palette and styling guide for cinematic RViz demos |

### Test Files (`test/`)

| File | Contents |
|------|----------|
| [`test_copyright.py`](test/test_copyright.py) | Standard ament copyright check |
| [`test_flake8.py`](test/test_flake8.py) | Standard ament flake8 lint |
| [`test_pep257.py`](test/test_pep257.py) | Standard ament pep257 docstring check |

---

## Prerequisites

- **ROS 2 Humble** (Ubuntu 22.04)
- **Gazebo Classic** (gazebo11)
- **TurtleBot3 packages**:
  ```bash
  sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description
  ```
- **Nav2**:
  ```bash
  sudo apt install ros-humble-nav2-bringup ros-humble-navigation2
  ```
- **SLAM Toolbox**:
  ```bash
  sudo apt install ros-humble-slam-toolbox
  ```
- **Python dependencies**: `numpy`, `opencv` (cv2)
  ```bash
  sudo apt install python3-numpy python3-opencv
  ```

### Environment Setup

```bash
# Required — tells TurtleBot3 which model to use
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models' >> ~/.bashrc
source ~/.bashrc
```

---

## Installation

```bash
# Clone into your ROS 2 workspace
cd ~/ros2_ws/src
git clone <this-repo-url> multi_robot_exploration

# Build
cd ~/ros2_ws
colcon build --packages-select multi_robot_exploration
source install/setup.bash
```

---

## Usage

### Full Autonomous Exploration (5 terminals)

Launch each command in a **separate terminal**, waiting for each to fully initialize before starting the next:

```bash
# Terminal 1 — Spawn Gazebo world + robots
ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py

# Terminal 2 — Start per-robot SLAM
ros2 launch multi_robot_exploration multi_robot_slam.launch.py

# Terminal 3 — Start map merging
ros2 launch multi_robot_exploration map_merge.launch.py

# Terminal 4 — Start Nav2 navigation stacks
ros2 launch multi_robot_exploration nav2_bringup_multi.launch.py

# Terminal 5 — Start frontier exploration
ros2 launch multi_robot_exploration frontier_exploration.launch.py
```

### Worlds

By default every launch generates a **new random world** (obstacles and spawn points).

```bash
# Set a seed for a reproducible random world
GAZEBO_WORLD_SEED=42 ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py

# Or use the TurtleBot3 house world
USE_HOUSE=1 ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py

# Or use the TurtleBot3 pillar world (required by the waypoint demo)
USE_TB3_WORLD=1 ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py
```

### Alternative: Waypoint Demo (map merge showcase)

The waypoints are hardcoded for turtlebot3_world, so start Terminal 1 with `USE_TB3_WORLD=1`, then replace Terminal 5 with:
```bash
ros2 launch multi_robot_exploration waypoint_demo.launch.py
```

### Visualize in RViz2

```bash
rviz2 -d $(ros2 pkg prefix multi_robot_exploration)/share/multi_robot_exploration/config/multi_robot_exploration_cinematic.rviz
```

---

## Pipeline Deep-Dive

### Layer 1: Simulation & Spawning

**File**: `spawn_two_turtlebots.launch.py` + `generate_random_world.py`

The launch file dynamically generates a Gazebo SDF world with random obstacles, then spawns two TurtleBot3 robots with:

- **Namespace isolation** — each robot's SDF `<odometry_frame>`, `<robot_base_frame>`, and `<frame_name>` are patched to `robot1/odom`, `robot1/base_footprint`, `robot1/base_scan` (and similarly for robot2)
- **Visual differentiation** — robot1 chassis → Turquoise, robot2 chassis → Orange
- **LiDAR ray fans disabled** — cleaner visualization

### Layer 2: Multi-Robot SLAM

**File**: `multi_robot_slam.launch.py`

One `slam_toolbox` async node per robot:
- Each robot maps independently in its own frame (`robot1/map`, `robot2/map`)
- Topic remapping (`/map` → `map`) prevents namespace collision
- Outputs: `/robot1/map` and `/robot2/map` (OccupancyGrid)

### Layer 3: Map Merging

**File**: `map_merge_node.py`

Custom from-scratch map fusion (no external `multirobot_map_merge` dependency):

1. Reads `/robot1/map` and `/robot2/map`
2. Uses known spawn transforms to reproject each cell into the global `map` frame
3. Vectorized NumPy rasterization for performance
4. Fusion rule: **occupied wins** > direct assign > average
5. Publishes static TF: `map` → `robot1/map`, `map` → `robot2/map`
6. Output: `/map` at 2 Hz

**Why spawn offsets are (0,0,0)**: Gazebo's diff-drive plugin initializes `/odom` to the robot's true spawn pose in the world frame. Since `slam_toolbox` builds on top of odom, each robot's map frame is already in the world frame — applying spawn offsets again would double-offset.

### Layer 4: Nav2 Navigation Stack

**File**: `nav2_bringup_multi.launch.py`

Per-robot Nav2 stack with:
- **DWB local planner** — differential-wheel-base controller
- **Theta\* global planner** — any-angle A* on the global costmap (unknown space allowed)
- **Recovery behaviors** — Spin, BackUp, Wait (for stuck situations)
- **Velocity smoother** — smooths `cmd_vel_nav` → `cmd_vel`
- **Autostart lifecycle** — nodes self-activate without manual `ros2 lifecycle set`
- Robot2 launches with an 8-second delay to prevent resource contention

### Layer 5: Frontier Coordinator

**File**: `frontier_coordinator.py`

The exploration brain, implementing:

1. **Frontier detection** — morphological dilation on the merged occupancy grid. Safe free cells adjacent to unknown cells form frontier regions. Obstacles are inflated by ~0.15m.
2. **Clustering** — connected components with area filtering (ignoring noise clusters < 15 pixels)
3. **Deduplication** — merge centroids within 1.5m radius
4. **Greedy cost-based assignment** for each robot needing a goal:
   - `distance_cost` — Euclidean distance to frontier
   - `separation_cost` — inverse penalty if frontier is near another robot's goal
   - `region_cost` — spatial partitioning using perpendicular bisector between robots' home positions
5. **Nav2 action client** — `NavigateToPose` with goal acceptance/rejection handling and frontier blacklisting
6. **Stuck detection** — if a robot hasn't moved >0.1m for 3 replan ticks (~9s), cancel, blacklist that frontier, and reassign
7. **Rich RViz markers** — frontier spheres (gray=unassigned, colored=assigned), goal arrows, text labels with robot state

---

## Configuration Reference

### Frontier Coordinator Tuning Knobs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replan_period` | 3.0s | How often to detect frontiers and reassign goals |
| `dedup_radius` | 1.5m | Merge frontier centroids closer than this |
| `min_frontier_size` | 15 px | Ignore frontier clusters smaller than this |
| `goal_reached_threshold` | 0.4m | Consider goal reached within this distance |
| `min_goal_distance` | 0.6m | Skip frontiers closer than this |
| `separation_weight` | 50.0 | Penalty weight for frontiers near other robot's goal |
| `region_weight` | 50.0 | Penalty weight for frontiers in other robot's region |
| `stuck_threshold` | 3 ticks | Ticks of no movement before cancelling goal |
| `blacklist_duration` | 30.0s | How long rejected frontiers stay blacklisted |

### Nav2 Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_vel_x` | 0.15 m/s | TurtleBot3 Burger max speed |
| `max_vel_theta` | 1.0 rad/s | Max rotation speed |
| `xy_goal_tolerance` | 0.25m | How close to get to goal |
| `controller_frequency` | 20 Hz | Local planner update rate |
| `costmap resolution` | 0.05 m | Matches SLAM resolution |

---

## RViz Visualization

### Recommended Topics to Display

| Display Type | Topic | What it shows |
|-------------|-------|---------------|
| Map | `/map` | Merged global occupancy grid |
| Map | `/robot1/map` | Robot 1's individual SLAM map |
| Map | `/robot2/map` | Robot 2's individual SLAM map |
| LaserScan | `/robot1/scan` | Robot 1's LiDAR data |
| LaserScan | `/robot2/scan` | Robot 2's LiDAR data |
| MarkerArray | `/frontiers_markers` | Frontier spheres + text labels |
| MarkerArray | `/frontier_goals_markers` | Goal arrows per robot |
| Path | `/robot1/plan` | Robot 1's planned path |
| Path | `/robot2/plan` | Robot 2's planned path |
| TF | All | Frame tree visualization |

### Cinematic Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Background | Deep slate | `#05070E` |
| Grid | Subtle dark | `#1E2330` (alpha 0.15) |
| Robot 1 | Electric Cyan | `#00F0FF` |
| Robot 2 | Neon Orange | `#FF5500` |
| Shared/Conflict | Vibrant Magenta | `#E024C3` |

---

## Troubleshooting

### Robots not moving
- Check Nav2 lifecycle: `ros2 lifecycle list /robot1/controller_server`
- All nodes should be in `active` state. If stuck in `unconfigured`, relaunch `nav2_bringup_multi.launch.py`

### Maps not merging / duplicate rooms
- Verify `map_merge_params.yaml` has spawn offsets at `0.0` (not the actual spawn coordinates — see [Layer 3](#layer-3-map-merging) for why)
- Check TF tree: `ros2 run tf2_tools view_frames`

### Robots exploring the same area
- Increase `separation_weight` and `region_weight` in `frontier_coordinator.py`
- Verify both robots have valid TF before assignment starts (coordinator waits for this)

### Nav2 goal rejected
- Frontier is inside an obstacle or outside the map bounds
- Coordinator automatically blacklists rejected frontiers for 30s

### "Waiting for Nav2 action servers" hangs
- Nav2 stack hasn't fully started yet — ensure `nav2_bringup_multi.launch.py` shows all lifecycle nodes as `active`

---

## License

Apache-2.0

---

## RL Frontier Selection (experimental)

A reinforcement-learning agent can replace the coordinator's hand-tuned frontier cost. It trains on the **real stack**: headless Gazebo, SLAM, map merge and Nav2, with a new random world every episode. It is warm-started from the heuristic and then fine-tuned with PPO. Full guide, design and rationale: **[docs/rl_training.md](docs/rl_training.md)**.

**Run training on a server:** each env is a full sim stack, and the pipeline takes about 1.5 days with 4 envs.

```bash
# once (server / distrobox): see docs/rl_training.md §2 for the apt packages
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user -r ~/swarm/requirements-rl.txt && pip3 uninstall -y setuptools
cd ~/swarm && colcon build --symlink-install && source install/setup.bash

# pipeline (each step has a pass/fail gate in the guide)
ros2 run multi_robot_exploration rl_evaluate --policy heuristic                       # baseline
ros2 run multi_robot_exploration rl_pretrain --num-envs 4 --decisions 3000            # clone heuristic -> bc_init.zip
ros2 run multi_robot_exploration rl_evaluate --policy ~/swarm_rl_runs/pretrain_<time>/bc_init.zip
ros2 run multi_robot_exploration rl_train --resume ~/swarm_rl_runs/pretrain_<time>/bc_init.zip --num-envs 4 --timesteps 20000
ros2 run multi_robot_exploration rl_evaluate --policy ~/swarm_rl_runs/<time>/frontier_ppo_final.zip

# use it (in place of terminal 5)
ros2 launch multi_robot_exploration rl_frontier_exploration.launch.py model_path:=$HOME/swarm_rl_runs/<time>/frontier_ppo_final.zip
```

| Design choice | Value | Source |
|---|---|---|
| Decision | pick 1 of the 12 nearest frontiers per robot (macro action), invalid ones masked | Tan et al. 2021; MaskablePPO |
| Reward | +0.1 per new m², −0.005 per sim s, −0.2 per failed goal; episode ends when the map saturates | Active Neural SLAM (coverage reward) |
| Warm start | behaviour cloning of the heuristic + critic pre-training | DRL + knowledge distillation (MDPI 2025) |
| PPO | lr 1e-4 → 0, clip 0.1, target_kl 0.015, 4 epochs, ent 0.001, γ 0.99 | Active Neural SLAM; conservative fine-tuning |
