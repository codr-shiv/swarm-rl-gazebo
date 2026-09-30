# Frontier Weight Tuning

The frontier choice uses **one fixed formula**. Only its **5 weights are learned**, by running the real stack (headless Gazebo, SLAM, map merge, Nav2) many times and keeping the weights that map the arena fastest. Nothing else is learned: there is no neural network, and no torch or RL libraries are needed.

> Run the tuning on a **server**. Each episode is a full Gazebo + 2×SLAM + 2×Nav2 stack. A default run is roughly 12 h with 8 parallel sims, or 1 day with 4.

---

## 1. The formula

When a robot needs a new goal, each of the 12 frontiers nearest to it gets a cost, and **the lowest cost wins**:

```
cost = 1·distance + w_crowding·crowding + w_other_half·other_half
     + w_unknown·unknown_around + w_size·frontier_size + w_turning·turning
```

| Term | Meaning (every term is "lower is better") | Heuristic weight |
|---|---|---|
| `distance` | metres from the robot to the frontier | 1 (fixed) |
| `crowding` | 1 / distance from the frontier to the *other* robot's goal (0 if it has none) | 50 |
| `other_half` | metres the frontier lies inside the other robot's half (split between the two start positions) | 50 |
| `unknown_around` | −(fraction of unknown cells within 1 m): favours information gain | 0 |
| `frontier_size` | −(frontier cells / 100): favours big frontiers | 0 |
| `turning` | (1 − cos(bearing)) / 2: 0 straight ahead, 1 directly behind | 0 |

- **Heuristic = baseline.** With the heuristic weights, the formula is exactly `frontier_coordinator.py`'s cost. That makes the heuristic a point in the search space.
- **Why distance is fixed.** Scaling every weight by the same factor never changes which frontier wins, so fixing `distance` at 1 loses nothing.

**Hardcoded (not learned):**
- **Frontier detection:** [`frontier_utils.py`](../multi_robot_exploration/frontier_utils.py), the same code as the coordinator.
- **Goal nudge:** each goal is moved at most 0.6 m, into known-free space at least 0.25 m from walls, because Nav2 rejects goals inside obstacles.
- **Masking:** frontiers closer than 0.6 m and the other robot's goal are never chosen.
- **Goal lifecycle:** reached within 0.4 m; stuck after 9 s with less than 0.1 m progress; frontier vanished; 30 s blacklist for failed goals.
- **Driving:** Nav2.

## 2. What "better" means (the score)

Each episode is a new random world. It ends when no frontiers are left, or when the **map stops growing** (under 0.5 m² in 60 s), or at 300 sim-seconds. Score:

| Event | Score |
|---|---|
| Each new m² mapped | +0.1 |
| Each sim second until the episode ends | −0.005 |
| Each failed / rejected / stuck / timed-out goal | −0.2 |

Every weight set ends up mapping about the same total area, because the arena saturates. So a higher score means **finishing the map sooner with fewer wasted trips**. This is the coverage-increase objective of Active Neural SLAM (Chaplot et al. 2020).

## 3. How the weights are learned (CMA-ES)

[CMA-ES](https://arxiv.org/abs/1604.00772) is a standard black-box optimiser for a handful of continuous parameters. It needs no gradients and handles noisy scores well.

**Each generation:**
1. **Sample:** CMA-ES samples 8 weight sets (the learned weights are searched in log10 space, 0.01–1000).
2. **Score on shared worlds:** each set runs on the **same 6 random worlds**. Those worlds are shared by all candidates and are new every generation. Comparing candidates on identical worlds removes most of the world-to-world noise. That noise (±40 per episode) is what made the earlier PPO attempt unable to learn.
3. **Progress check:** the current best estimate (the CMA-ES mean) and the heuristic also run on those worlds. `mean_weights_score − heuristic_score` is your progress indicator.
4. **Update:** CMA-ES moves towards the better sets. Its step size `sigma` shrinks as it converges. The run stops at 20 generations or when `sigma < 0.05`, i.e. steps below ~12 % weight changes.

**Why this converges** where the network didn't:
- It searches 5 numbers instead of ~70,000.
- The starting point is the heuristic's own weights (plus 1.0 for the new terms), so it begins near a known-good solution.
- The candidates are compared on identical worlds.

On a noisy 5-D test problem, the included CMA-ES cuts the distance to the optimum about 10× within 20 generations.

Implementation: [`tuning/tune.py`](../multi_robot_exploration/tuning/tune.py). It's a self-contained CMA-ES following Hansen's tutorial, with no extra packages; pycma is avoided because it pulls in SciPy, which conflicts with numpy 2 on Humble.

## 4. Files

| Piece | File |
|---|---|
| Formula, goal nudge, masking | [`tuning/features.py`](../multi_robot_exploration/tuning/features.py) |
| Runs one scored episode | [`tuning/episode.py`](../multi_robot_exploration/tuning/episode.py) |
| CMA-ES loop (`tune_weights`) | [`tuning/tune.py`](../multi_robot_exploration/tuning/tune.py) |
| Fixed-world comparison (`evaluate_weights`) | [`tuning/evaluate.py`](../multi_robot_exploration/tuning/evaluate.py) |
| Deployment node (`tuned_frontier_coordinator`) | [`tuning/coordinator.py`](../multi_robot_exploration/tuning/coordinator.py) |
| ROS side (map, TF, Nav2 goals, events) | [`tuning/ros_interface.py`](../multi_robot_exploration/tuning/ros_interface.py) |
| Sim start / stop per slot | [`tuning/sim_manager.py`](../multi_robot_exploration/tuning/sim_manager.py) |
| Headless stack launch | [`launch/headless_stack.launch.py`](../launch/headless_stack.launch.py) |

## 5. Install (server; Ubuntu 22.04 + ROS 2 Humble)

```bash
sudo apt update && sudo apt install -y ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description \
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox ros-humble-gazebo-ros-pkgs \
  ros-humble-rmw-cyclonedds-cpp python3-numpy python3-opencv python3-colcon-common-extensions
# copy/clone this repo to ~/swarm
cd ~/swarm && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```
No pip packages are needed.

Every terminal (or add these to `~/.bashrc`):
```bash
source /opt/ros/humble/setup.bash && source ~/swarm/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## 6. Run it

### 6.1 Smoke test (~3 min)
```bash
ros2 run multi_robot_exploration evaluate_weights --weights heuristic --seeds 42 --max-episode-sim-s 120
```
**Pass:** it prints `seed=42 end=... score=...`, and afterwards `ps -eo cmd | grep gzserver` is empty.

### 6.2 Optional: A/B test the goal nudge (~1 h; the two commands can run at the same time)
```bash
ros2 run multi_robot_exploration evaluate_weights --weights heuristic --instance-id 9
ros2 run multi_robot_exploration evaluate_weights --weights heuristic --instance-id 10 --no-goal-fix
column -s, -t < ~/swarm_tuning_runs/eval.csv
```
Keep the nudge (the default) unless the `goal_fix=False` rows score higher. If you turn it off, add `--no-goal-fix` to **every** command below, and `goal_fix:=false` when deploying.

### 6.3 Tune
```bash
ros2 run multi_robot_exploration tune_weights --num-envs 8
```
Set `--num-envs` to about min(CPU cores / 2, RAM GB / 2). One generation is 8 × 6 + 2 × 6 = 60 episodes of about 4–5 min each:

| Envs | Per generation | 20 generations |
|---|---|---|
| 8 | ~35 min | ~12 h |
| 4 | ~70 min | ~1 day |

With limited time, `--worlds 4` is about ⅓ faster but noisier.

Output, in `~/swarm_tuning_runs/<time>/`:

| File | Contents |
|---|---|
| `best_weights.json` | the current best weights (the CMA-ES mean), rewritten every generation |
| `generations.csv` | one row per generation: `sigma`, candidate scores, `mean_weights_score`, `heuristic_score`, the weights |
| `evaluations.csv` | every episode |
| `cma_state.pkl` | optimiser state |
| `sim_logs/` | the latest episode of each slot |

**Watch progress:**
```bash
column -s, -t < ~/swarm_tuning_runs/<time>/generations.csv
```
Healthy signs:
- `mean_weights_score` rises above `heuristic_score` over generations. They share worlds, so the difference is meaningful. Judge the trend over several generations, not single rows: the same weights on the same world can score quite differently between runs (in a test, the heuristic scored −0.06 and then 1.81 on one world), because sim, SLAM and Nav2 timing isn't deterministic.
- `sigma` shrinks from 1.0.
- `failed_episodes` stays 0.

**If the run is interrupted**, continue with:
```bash
ros2 run multi_robot_exploration tune_weights --num-envs 8 --run-dir ~/swarm_tuning_runs/<time> --resume
```
The same command with a higher `--generations` continues a finished run.

| Flag | Default | Meaning |
|---|---|---|
| `--generations` | 20 | stop after this many (or earlier if `sigma < 0.05`) |
| `--popsize` | 8 | weight sets per generation |
| `--worlds` | 6 | worlds per weight set; more means less noise but slower |
| `--max-episode-sim-s` | 300 | episode limit |
| `--no-baseline` | off | skip the per-generation mean/heuristic runs (20 % faster, no progress signal) |
| `--first-instance` | 0 | sim slots used: ROS domain 40+slot, Gazebo port 11445+slot. Keep concurrent jobs apart. |
| `--seed` | 0 | makes the world sequence reproducible |

### 6.4 Confirm on held-out worlds (~40 min each)
```bash
ros2 run multi_robot_exploration evaluate_weights --weights heuristic
ros2 run multi_robot_exploration evaluate_weights --weights ~/swarm_tuning_runs/<time>/best_weights.json
column -s, -t < ~/swarm_tuning_runs/eval.csv
```
The default 8 seeds (101 … 808) are never used during tuning. The tuned weights win if their mean score is higher, which means a shorter time to finish mapping, fewer failed goals, or both.

### 6.5 Use the weights (normal 5-terminal pipeline)
Start terminals 1–4 as usual. Then, instead of `frontier_exploration.launch.py`:
```bash
ros2 launch multi_robot_exploration tuned_frontier_exploration.launch.py \
    weights_file:=$HOME/swarm_tuning_runs/<time>/best_weights.json
```
Without `weights_file` it uses the heuristic weights. Or copy the numbers from `best_weights.json` straight into a report: that's the whole result.

## 7. Troubleshooting

| Problem | What to do |
|---|---|
| `sim not ready (attempt N): {...}` | The dict shows what's missing (`nav2_active`, `map`, `tf`). See `sim_logs/sim_<slot>.log`. Failed episodes are retried once, then counted as the generation's worst score. |
| Sims left after a crash or `kill -9` | Run `ps -eo pid,pgid,cmd \| grep headless_stack`, then `kill -INT -<pgid>` for each. |
| Two jobs interfere | Use different `--first-instance` / `--instance-id` slots. |
| Weights hit a bound (0.01 or 1000) | That term is effectively off, or dominant. This is a legitimate result. |

## 8. History
- An earlier PPO attempt (a neural network choosing frontiers) didn't learn within 20k decisions. It's in git commit `2b15f0a` ("complex rl") if you ever want it back.
- The weight-tuning approach replaced it because it has far fewer unknowns and compares candidates on shared worlds.

### Sources
- N. Hansen, *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772.
- Chaplot et al., *Learning to Explore using Active Neural SLAM*, ICLR 2020 (coverage-increase objective).
