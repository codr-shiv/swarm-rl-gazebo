# RL Frontier Selection

Train an agent that decides **which frontier each robot explores next**, so the team maps the arena as fast as possible. It replaces only the coordinator's hand-tuned cost function. Everything else is the same stack you already run: Gazebo, per-robot SLAM, map merge and Nav2.

Training runs on **the real stack, headless** (gzserver only, no window). Every episode gets a fresh random world.

---

## How it works

```
 rl_train (MaskablePPO)                          one episode = one fresh sim
 ┌──────────────────────┐   action: slot k   ┌──────────────────────────────────────────┐
 │ policy  π(obs, mask) │ ─────────────────▶ │ GazeboExplorationEnv                     │
 └──────────────────────┘                    │  ├─ SimManager ─ ros2 launch             │
            ▲  obs, reward                   │  │     rl_sim_stack.launch.py (headless) │
            └─────────────────────────────── │  │     gzserver + SLAM + map merge + Nav2│
                                             │  └─ ExplorationInterface (ROS node)      │
                                             │        /map, TF, NavigateToPose x2       │
                                             └──────────────────────────────────────────┘
```

| Piece | File | Role |
|---|---|---|
| Headless stack | [`launch/rl_sim_stack.launch.py`](../launch/rl_sim_stack.launch.py) | Brings up Gazebo, SLAM, map merge and Nav2 from one launch. There's no exploration brain in it. |
| Sim manager | [`rl/sim_manager.py`](../multi_robot_exploration/rl/sim_manager.py) | Starts and stops that launch in its own process group, ROS domain and Gazebo port, so envs can run in parallel. |
| ROS interface | [`rl/ros_interface.py`](../multi_robot_exploration/rl/ros_interface.py) | Handles the map, poses, Nav2 goals, and the "robot needs a new goal" events. Training and deployment share it. |
| Features | [`rl/features.py`](../multi_robot_exploration/rl/features.py) | Builds the observation and the action mask, and holds the heuristic baseline. Pure numpy. |
| Env | [`rl/gazebo_env.py`](../multi_robot_exploration/rl/gazebo_env.py) | The Gymnasium env: episode control and reward. |
| Train / evaluate | [`rl/train.py`](../multi_robot_exploration/rl/train.py), [`rl/evaluate.py`](../multi_robot_exploration/rl/evaluate.py) | The CLIs: `rl_train` and `rl_evaluate`. |
| Deploy | [`rl/rl_coordinator.py`](../multi_robot_exploration/rl/rl_coordinator.py) | `rl_frontier_coordinator`, a replacement for terminal 5. |

Frontiers come from [`frontier_utils.py`](../multi_robot_exploration/frontier_utils.py), the same code `frontier_coordinator` uses, so the agent sees exactly the candidates the heuristic sees. Goal handling matches the coordinator too: reach radius, stuck detection, frontier-vanished check and the 30 s blacklist.

### MDP

- **Step = one decision.** Whenever a robot needs a goal (at the start, or when its goal succeeded, failed, got stuck, timed out or vanished), the agent picks a frontier for that robot. The sim then keeps running, with both robots driving, until the next decision. Steps therefore take variable sim time.
- **Action:** `Discrete(12)`, an index into the 12 frontiers nearest the deciding robot. An **action mask** hides empty slots, frontiers closer than 0.6 m, and the other robot's current goal. The algorithm is `MaskablePPO` from sb3-contrib.
- **Observation (125 floats):** 10 features for each of the 12 candidate slots, plus 5 global features:
  - **Per candidate:** valid flag; distance to the deciding robot; distance to the other robot; distance to the other robot's goal; frontier size; unknown fraction within 1 m (information gain); side of the region split; bearing (cos, sin); and the coordinator's own heuristic cost. That last one gives the agent a good prior to improve on.
  - **Global:** explored area; elapsed fraction of the episode; number of frontiers; whether the other robot is busy; distance between the robots.

  The features are ego-centric, so one shared policy drives both robots.
- **Reward:** the robots share it.

  | Term | Value |
  |---|---|
  | Newly mapped area | +1 per m² |
  | Time | −0.02 per sim second |
  | Failed / rejected / stuck / timed-out goal | −1 each |
  | Map fully explored | +10 |

  The constants are at the top of `gazebo_env.py`.
- **Episode end:**
  - **Terminated:** no frontiers left for 5 s.
  - **Truncated:** the sim-time limit is reached (`--max-episode-sim-s`, default 600), or the sim process dies or freezes.

---

## 0. Running on another (more powerful) PC

1. You need Ubuntu 22.04 with ROS 2 Humble, either natively or in a distrobox like this one. Install the ROS packages:
   ```bash
   sudo apt update && sudo apt install -y ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description \
     ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox ros-humble-gazebo-ros-pkgs \
     ros-humble-rmw-cyclonedds-cpp python3-numpy python3-opencv python3-pip python3-colcon-common-extensions
   ```
2. Copy the `swarm` folder to `~/swarm` on the new PC, then install the ML packages and build:
   ```bash
   pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
   pip3 install --user -r ~/swarm/requirements-rl.txt
   pip3 uninstall -y setuptools          # required, see step 1 below
   cd ~/swarm && source /opt/ros/humble/setup.bash && colcon build --symlink-install
   ```
3. Train. Every terminal needs this setup first (or add it to `~/.bashrc`):
   ```bash
   source /opt/ros/humble/setup.bash && source ~/swarm/install/setup.bash
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   ros2 run multi_robot_exploration rl_train --num-envs 6 --timesteps 20000
   ```
   Pick `--num-envs` so that each env has about 2 CPU cores and 2 GB RAM (for example, 6 on a 16-core, 32 GB PC). Let the run finish. The model is saved as `~/swarm_rl_runs/<run>/frontier_ppo_final.zip`, with checkpoints in `checkpoints/`.
4. Compare it with the heuristic (step 4 below), then run it in the normal pipeline (step 5 below).

## 1. Install (once, inside the distrobox)

```bash
distrobox enter ros-humble
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user -r ~/swarm/requirements-rl.txt
pip3 uninstall -y setuptools      # IMPORTANT: torch pulls a setuptools that breaks colcon on Humble
swarm_build
```

Check the install:
```bash
python3 -c "import torch, sb3_contrib, gymnasium; print('ok')"
ls ~/swarm/install/multi_robot_exploration/lib/multi_robot_exploration/ | grep rl_
```

## 2. Baseline and smoke test (about 4 minutes)

This runs one short episode with the existing heuristic. It shows that the headless sim, the decision loop and teardown all work:

```bash
ros2 run multi_robot_exploration rl_evaluate --policy heuristic --seeds 42 --max-episode-sim-s 180
```

You should get a line like `seed=42 end=time_limit explored=53.2 m2 time=180.0 s`. Results are appended to `~/swarm_rl_runs/eval.csv`, and sim logs go to `~/swarm_rl_runs/eval_sim_logs/`.

For the real baseline, use the 5 default evaluation worlds with full-length episodes (about 55 minutes):
```bash
ros2 run multi_robot_exploration rl_evaluate --policy heuristic
ros2 run multi_robot_exploration rl_evaluate --policy random      # optional lower bound
```

## 3. Train

```bash
ros2 run multi_robot_exploration rl_train --num-envs 2
```

| Flag | Default | Meaning |
|---|---|---|
| `--timesteps` | 20000 | Total **decisions** to train on. |
| `--num-envs` | 1 | Parallel sims. Each uses about 2 CPU cores and 1.5–2 GB RAM. |
| `--max-episode-sim-s` | 600 | Episode length in sim seconds. |
| `--n-steps` | 128 | Decisions per env between PPO updates. |
| `--rtf` | 1.0 | Requested Gazebo real-time factor. Values above 1 help only if the CPU keeps up; watch `explore/episode_sim_time_s`. |
| `--gui` | off | Show Gazebo for env 0, to watch what the agent is doing. |
| `--resume PATH.zip` | | Continue from a checkpoint. |
| `--run-dir` | `~/swarm_rl_runs/<timestamp>` | Where checkpoints, TensorBoard logs, the monitor CSV and sim logs go. |

Each episode costs about 35 s of startup plus the sim time (RTF 1). One 600 s episode is roughly 50–150 decisions, so plan on **about 10 minutes per episode per env**. 20k decisions is about 1–2 days with 2 envs. Treat this as an initial setup: prove the loop, then scale up.

**Stopping:** Ctrl+C saves `frontier_ppo_interrupted.zip` and shuts every sim down. Checkpoints are saved every `--checkpoint-every` decisions in `checkpoints/`.

**Monitoring** (in another terminal):
```bash
tensorboard --logdir ~/swarm_rl_runs --port 6006     # open http://localhost:6006
```

| Metric | What it means |
|---|---|
| `explore/explored_m2` | Main metric: mapped area at episode end. |
| `explore/fully_explored` | Fraction of episodes that finished exploring the whole map. |
| `explore/failed_goals` | Bad frontier choices (unreachable, stuck). |
| `rollout/ep_rew_mean` | Episode return. |

## 4. Evaluate against the heuristic

```bash
ros2 run multi_robot_exploration rl_evaluate --policy ~/swarm_rl_runs/<run>/frontier_ppo_final.zip
column -s, -t < ~/swarm_rl_runs/eval.csv
```

Every policy is evaluated on the same world seeds (`--seeds`, default `101 202 303 404 505`), so the rows are directly comparable. The policy is better if it maps more area in the same time, or finishes (`end_reason=explored`) sooner.

## 5. Run the trained policy in the normal pipeline

Start terminals 1–4 as usual. Then, in place of `frontier_exploration.launch.py`, run:
```bash
ros2 launch multi_robot_exploration rl_frontier_exploration.launch.py \
    model_path:=$HOME/swarm_rl_runs/<run>/frontier_ppo_final.zip
```

To watch the training stack itself with a Gazebo window:
```bash
ros2 launch multi_robot_exploration rl_sim_stack.launch.py gui:=true
```

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `sim not ready (attempt N): {...}` | The dict shows what is missing: `nav2_active`, `map` or `tf`. See `sim_logs/sim_<N>.log` in the run dir. After 3 failed attempts the env raises an error. |
| `colcon build` fails with `canonicalize_version() ... strip_trailing_zero` | Run `pip3 uninstall -y setuptools` (see Install). |
| Sims left running after a crash or kill -9 | List them with `ps -eo pid,pgid,cmd \| grep rl_sim_stack`, then `kill -INT -<pgid>` for each (the sim manager puts every sim in its own process group). Don't use a broad `pkill -f` pattern: distrobox shares the host's process list, so it can also hit a sim you're running by hand. |
| Training and your manual sim interfere | They can't: training uses ROS domains 40+ and Gazebo ports 11445+, and `rl_evaluate` uses slot 9 by default. Your manual sim uses the default domain. |

## Next steps (ideas)

- **Pretrain offline.** `features.build_observation` only needs an occupancy grid and poses, so a fast 2D simulator (like the earlier `rl_sim`) could produce identical observations. Pretrain there, then fine-tune here.
- **Behaviour cloning warm start.** Imitate `heuristic_action` for a few thousand decisions before PPO, so early episodes aren't wasted.
- **Tune the reward**, for example a larger time penalty to favour finishing early.
- **Add a local map crop** (a CNN input) next to the candidate features.
