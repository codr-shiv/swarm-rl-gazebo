# RL Frontier Selection

Train an agent that decides **which frontier each robot explores next**, so the team maps the arena as fast as possible. It replaces only the coordinator's hand-tuned frontier cost; Gazebo, per-robot SLAM, map merge and Nav2 stay exactly as they are.

Training runs on **the real stack, headless** (gzserver only). Every episode is a fresh random world.

> **Run training on a server, not a laptop.** One env is one full Gazebo + 2×SLAM + 2×Nav2 stack. The recommended pipeline takes roughly 1.5 days with 4 envs. Section 3 sizes it for your hardware.

---

## 1. Design

### 1.1 Components

```
 rl_pretrain / rl_train / rl_evaluate                   one episode = one fresh sim
 ┌──────────────────────┐   action: slot k   ┌──────────────────────────────────────────┐
 │ MaskablePPO policy   │ ─────────────────▶ │ GazeboExplorationEnv                     │
 └──────────────────────┘                    │  ├─ SimManager ─ ros2 launch             │
            ▲  obs, mask, reward             │  │     rl_sim_stack.launch.py (headless) │
            └─────────────────────────────── │  │     gzserver + SLAM + map merge + Nav2│
                                             │  └─ ExplorationInterface (ROS node)      │
                                             │        /map, TF, NavigateToPose x2       │
                                             └──────────────────────────────────────────┘
```

| Piece | File | Role |
|---|---|---|
| Headless stack | [`launch/rl_sim_stack.launch.py`](../launch/rl_sim_stack.launch.py) | Brings up Gazebo, SLAM, map merge and Nav2 from one launch, with no exploration brain. |
| Sim manager | [`rl/sim_manager.py`](../multi_robot_exploration/rl/sim_manager.py) | Gives each sim its own process group, ROS domain (40 + slot) and Gazebo port (11445 + slot). |
| ROS interface | [`rl/ros_interface.py`](../multi_robot_exploration/rl/ros_interface.py) | Map, poses, Nav2 goals, and "robot needs a new goal" events. Shared by training and deployment. |
| Features | [`rl/features.py`](../multi_robot_exploration/rl/features.py) | Candidate list, goal nudge, observation, action mask, heuristic teacher. Pure numpy. |
| Env | [`rl/gazebo_env.py`](../multi_robot_exploration/rl/gazebo_env.py) | The Gymnasium env: episode control and reward. |
| Warm start | [`rl/pretrain.py`](../multi_robot_exploration/rl/pretrain.py) | `rl_pretrain`: collects heuristic play and behaviour-clones it into the policy. |
| Train / evaluate | [`rl/train.py`](../multi_robot_exploration/rl/train.py), [`rl/evaluate.py`](../multi_robot_exploration/rl/evaluate.py) | `rl_train` (PPO fine-tuning) and `rl_evaluate` (fixed-seed comparison). |
| Deploy | [`rl/rl_coordinator.py`](../multi_robot_exploration/rl/rl_coordinator.py) | `rl_frontier_coordinator`, a replacement for terminal 5. |

Frontiers come from [`frontier_utils.py`](../multi_robot_exploration/frontier_utils.py), the same code `frontier_coordinator` uses. The goal lifecycle is also the coordinator's: reach radius 0.4 m, stuck after 9 s without 0.1 m of progress, frontier vanished, and a 30 s blacklist for failed goals.

### 1.2 Decision process (semi-MDP with macro actions)

- **Step = one frontier decision.** When a robot needs a goal, the agent picks a frontier for it. Both robots then keep driving (Nav2) until some robot needs a goal again, so a step lasts a variable amount of sim time. Decisions happen at the frontier level rather than per velocity command, following decentralised multi-robot exploration with macro actions (Tan et al. 2021).
- **Action:** `Discrete(12)`, an index into the 12 frontiers nearest the deciding robot. **Action masking** (`MaskablePPO`, sb3-contrib) hides:
  - empty slots
  - frontiers closer than 0.6 m
  - the other robot's current goal
  - frontiers with no reachable safe goal (see goal nudge)
- **Goal nudge (`safe_goals`):** Nav2 rejects goals inside inflated obstacles. The first training run logged 278 "start or goal pose are an obstacle" failures in a single episode. Each frontier centroid is therefore moved to the nearest cell within 0.6 m that is known-free and at least 0.25 m from any obstacle, or dropped if there is none. This moves goals *into known free space*. Snapping goals onto the frontier edge (next to unknown space) was measured earlier to halve exploration, and is not what this does. Disable it with `--no-goal-fix` (see the A/B test in 3.2).
- **Observation (137 floats):**
  - **11 features per candidate slot:** valid flag; distance to the deciding robot; distance to the other robot; distance to the other robot's goal; frontier size; unknown fraction within 1 m (information gain); side of the region split; bearing (cos, sin); `log(1 + heuristic cost)`; and a **heuristic-pick flag**.
  - **5 global features:** explored area; elapsed fraction; number of frontiers; whether the other robot is busy; distance between the robots.

  The features are ego-centric, so one policy drives both robots. The flag lets the policy reproduce the heuristic exactly, then learn when to deviate from it.
- **Reward** (shared by both robots):

  | Term | Value |
  |---|---|
  | Newly mapped area (merged map) | +0.1 per m² |
  | Time | −0.005 per sim second |
  | Failed / rejected / stuck / timed-out goal | −0.2 each |

  This is the coverage-increase reward of Active Neural SLAM (0.02 per m² there). The scale keeps episode returns around +2 to +5.
- **Episode end:**
  - **Terminated:** no frontiers for 5 s (`explored`), **or the map grew less than 0.5 m² in the last 60 s (`saturated`)**. Saturation ends episodes as soon as the map stops growing, so faster exploration means less accumulated time penalty. Without it, every policy collects the same area reward and the reward can't tell them apart.
  - **Truncated:** 300 sim-s limit (`time_limit`), or the sim died.

### 1.3 Why a warm start (and what went wrong without it)

The first training run (from scratch, old reward, 600 s episodes, 7 h, 155 episodes) learned nothing:
- the return stayed flat at 10–13 per 25-episode block, with a spread of −36 to +45 caused by the random worlds
- every episode hit the 600 s limit
- about 24 of ~42 decisions failed

PPO itself was healthy (KL ≈ 0.008, rising explained variance). The task was the problem:
1. The arena saturates at about 53 m², so every policy earned the same area reward.
2. Many offered goals were unreachable.
3. A random policy needs far more samples than we have. Active Neural SLAM trains its goal-selection policy with 72 parallel threads over millions of steps; we get about 20k decisions.

Fixes 1 and 2 are described in 1.2. For 3, the standard remedy is a **teacher warm start (knowledge distillation / behaviour cloning)**, as in multi-robot DRL exploration with knowledge distillation (MDPI Mathematics 2025). `rl_pretrain` clones the heuristic into the policy and pre-trains the critic on Monte-Carlo returns of the same episodes. A random critic is what usually destroys a cloned policy in the first PPO updates. PPO then fine-tunes with conservative settings.

### 1.4 PPO settings (`rl/train.py`, applied on `--resume` too)

| Setting | Value | Source / reason |
|---|---|---|
| algorithm | MaskablePPO (sb3-contrib) | invalid-action masking |
| policy | MLP, separate pi/vf heads [128,128] | small vector observation |
| learning rate | 1e-4, linear decay to 0 | fine-tuning a warm start; ANS uses 2.5e-5 from scratch at a much larger scale |
| n_steps per env | 64 (×4 envs = 256 per update) | frequent updates, slow samples |
| batch_size / n_epochs | 64 / 4 | ANS: 4 epochs |
| clip_range | 0.1 | stay near the cloned policy |
| target_kl | 0.015 | early-stop updates that drift too far |
| ent_coef | 0.001 | ANS |
| gamma / gae_lambda | 0.99 / 0.95 | ANS / SB3 defaults |
| vf_coef / max_grad_norm | 0.5 / 0.5 | ANS / SB3 defaults |

Behaviour cloning uses Adam 1e-3, batch 256, 30 epochs, with 10 % of *episodes* held out. Its loss is masked cross-entropy plus 0.5 × value MSE to discounted returns (γ 0.99).

---

## 2. Install (server or distrobox, Ubuntu 22.04 + ROS 2 Humble)

```bash
sudo apt update && sudo apt install -y ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description \
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox ros-humble-gazebo-ros-pkgs \
  ros-humble-rmw-cyclonedds-cpp python3-numpy python3-opencv python3-pip python3-colcon-common-extensions
# copy/clone this repo to ~/swarm, then:
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user -r ~/swarm/requirements-rl.txt
pip3 uninstall -y setuptools     # REQUIRED: torch pulls setuptools>=77, which breaks colcon build on Humble
cd ~/swarm && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

Every terminal (or add these to `~/.bashrc`):
```bash
source /opt/ros/humble/setup.bash && source ~/swarm/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Check it:
```bash
python3 -c "import torch, sb3_contrib, gymnasium; print('ok')"
ls ~/swarm/install/multi_robot_exploration/lib/multi_robot_exploration/ | grep rl_
# -> rl_evaluate  rl_frontier_coordinator  rl_pretrain  rl_train
```

A GPU isn't needed: the network is tiny, and the simulation is the bottleneck.

---

## 3. Training pipeline (run on the server)

Each step has a **gate**. Don't continue until it passes.

### 3.0 Size the run
Each env needs about 2 CPU cores and 2 GB RAM. Set `N = min(cores / 2, RAM_GB / 2)`, for example 8 on a 16-core / 32 GB server. Wall time scales about 1/N. Timings below are for N = 4 at real-time speed.

### 3.1 Smoke test (~5 min)
```bash
ros2 run multi_robot_exploration rl_evaluate --policy heuristic --seeds 42 --max-episode-sim-s 120
```
**Gate:** it prints `seed=42 end=... explored=... m2`, and afterwards `ps -eo cmd | grep gzserver` shows nothing.

### 3.2 A/B test the goal nudge (~1 h; the two can run at the same time in two terminals)
```bash
ros2 run multi_robot_exploration rl_evaluate --policy heuristic --seeds 101 202 303 404 505 --instance-id 9
ros2 run multi_robot_exploration rl_evaluate --policy heuristic --seeds 101 202 303 404 505 --instance-id 10 --no-goal-fix
column -s, -t < ~/swarm_rl_runs/eval.csv
```
**Gate:** keep the nudge (the default) if the `goal_fix=True` rows have mean `explored_m2` ≥ and `failed_goals` ≤ the `False` rows. Otherwise add `--no-goal-fix` to **every** command below. The same setting must be used for pretrain, train, evaluate and deploy (`goal_fix:=false`). These rows are also your heuristic baseline.

### 3.3 Warm start: collect + clone (~4–6 h at N = 4)
```bash
ros2 run multi_robot_exploration rl_pretrain --num-envs 4 --decisions 3000
```
This writes `~/swarm_rl_runs/pretrain_<time>/`:

| File | Contents |
|---|---|
| `dataset.npz` | the heuristic decisions |
| `heuristic_episodes.csv` | the heuristic baseline on training-distribution worlds |
| `bc_init.zip` | the warm-started policy |

**Gate:** the final line says `held-out action accuracy ≥ 0.85: OK`, and the value `R2` is > 0. If accuracy is low, collect more data. To re-clone without re-collecting:
```bash
ros2 run multi_robot_exploration rl_pretrain --dataset ~/swarm_rl_runs/pretrain_<time>/dataset.npz --epochs 60 --run-dir ~/swarm_rl_runs/pretrain_<time>
```

### 3.4 Check the warm start (~1 h)
```bash
ros2 run multi_robot_exploration rl_evaluate --policy ~/swarm_rl_runs/pretrain_<time>/bc_init.zip
```
**Gate:** mean `explored_m2` and episode time are within about 5 % of the heuristic rows from 3.2. It's a clone, so it should match. If it doesn't, don't start PPO.

### 3.5 PPO fine-tuning (~1–1.5 days at N = 4 for 20k decisions)
```bash
ros2 run multi_robot_exploration rl_train --resume ~/swarm_rl_runs/pretrain_<time>/bc_init.zip \
    --num-envs 4 --timesteps 20000
```
Let it finish. It writes `~/swarm_rl_runs/<time>/`:
- `frontier_ppo_final.zip`
- `checkpoints/frontier_ppo_<N>_steps.zip` (every 1000 decisions)
- `tb/` (TensorBoard logs)
- `monitor.monitor.csv`
- `sim_logs/sim_<slot>.log` (the latest episode of each env)

To run several jobs on one machine, give them non-overlapping slots with `--first-instance`, for example `--first-instance 20`.

Monitor it with:
```bash
tensorboard --logdir ~/swarm_rl_runs --port 6006
```

What healthy curves look like:

| Metric | Expected behaviour |
|---|---|
| `explore/episode_sim_time_s` | should **decrease**: saturating the map sooner is the objective |
| `explore/explored_m2` | stays about constant (the arena saturates) |
| `explore/failed_goals` | should decrease |
| `rollout/ep_rew_mean` | should increase slowly from the warm-start level |
| `train/approx_kl` | ≤ 0.015; `target_kl` stops updates above that |
| `train/clip_fraction` | < 0.2 |
| `train/explained_variance` | > 0.3 early (warm critic) |

**Warning signs:**
- `ep_rew_mean` drops well below its first values and stays there for more than 2000 decisions.
- `entropy_loss` heads towards 0 very early.

Either way, the policy is drifting from the warm start. Stop, and resume from the last good checkpoint with a lower learning rate (edit `LEARNING_RATE` in `rl/train.py`).

### 3.6 Pick the best model (~1 h per model)
```bash
for m in ~/swarm_rl_runs/<time>/checkpoints/frontier_ppo_{10000,15000}_steps.zip ~/swarm_rl_runs/<time>/frontier_ppo_final.zip; do
  ros2 run multi_robot_exploration rl_evaluate --policy $m
done
column -s, -t < ~/swarm_rl_runs/eval.csv
```
Keep the model with the shortest mean episode time (the fastest saturation) at equal or higher `explored_m2`, compared with the heuristic rows.

### Realistic expectations
- **After 3.4:** heuristic-level performance, by construction.
- **After 3.5:** a modest improvement is the realistic target: fewer failed goals, better splits between the robots, earlier saturation. No setup can guarantee a large jump over a well-tuned heuristic within 20k decisions. The warm start makes it very unlikely you end up *worse*, because you can always keep `bc_init.zip`.
- **To go further:** more decisions (`--timesteps 50000+` with more envs), or the ideas in section 6.

---

## 4. Deploy the trained policy (normal 5-terminal pipeline)

Start terminals 1–4 as usual. Then, in place of `frontier_exploration.launch.py`:
```bash
ros2 launch multi_robot_exploration rl_frontier_exploration.launch.py \
    model_path:=$HOME/swarm_rl_runs/<time>/frontier_ppo_final.zip
```
The node's parameters must match training: `goal_fix` (default true) and `max_episode_sim_s` (default 300). To watch the training stack with a Gazebo window: `ros2 launch multi_robot_exploration rl_sim_stack.launch.py gui:=true`.

---

## 5. Command reference

| Command | Key flags (defaults) |
|---|---|
| `rl_pretrain` | `--decisions 3000`, `--num-envs 1`, `--max-episode-sim-s 300`, `--no-goal-fix`, `--dataset`, `--epochs 30`, `--first-instance 0`, `--run-dir` |
| `rl_train` | `--resume`, `--timesteps 20000`, `--num-envs 1`, `--n-steps 64`, `--max-episode-sim-s 300`, `--no-goal-fix`, `--checkpoint-every 1000`, `--first-instance 0`, `--rtf 1.0`, `--gui`, `--run-dir` |
| `rl_evaluate` | `--policy heuristic\|random\|<zip>`, `--seeds 101 202 303 404 505`, `--max-episode-sim-s 300`, `--no-goal-fix`, `--instance-id 9`, `--out ~/swarm_rl_runs/eval.csv` |

Constants:
- **Reward and episode end:** top of `rl/gazebo_env.py`.
- **Goal nudge and features:** top of `rl/features.py`.
- **PPO:** top of `rl/train.py`.

Changing observation features makes old models and datasets incompatible, so re-run from 3.3.

## 6. Troubleshooting

| Problem | What to do |
|---|---|
| `sim not ready (attempt N): {...}` | The dict shows what's missing (`nav2_active`, `map`, `tf`). See `sim_logs/sim_<slot>.log`. After 3 failed attempts the env raises an error. |
| `colcon build` fails with `canonicalize_version() ... strip_trailing_zero` | Run `pip3 uninstall -y setuptools`. |
| Sims left after a crash or `kill -9` | Run `ps -eo pid,pgid,cmd \| grep rl_sim_stack`, then `kill -INT -<pgid>` for each. Avoid broad `pkill -f`: in a distrobox it also hits sims you run by hand. |
| Two jobs interfere | Give them different `--first-instance` / `--instance-id` slots. |

## 7. Next steps (ideas)
- **More data, more envs:** simplest and most reliable.
- **Offline pretraining:** `features.build_observation` needs only an occupancy grid and poses, so a fast 2D simulator (like the earlier `rl_sim`) could generate cloning data or pre-train PPO.
- **Local map crop:** add a CNN input next to the candidate features (Active Neural SLAM style).

### Sources
- Chaplot et al., *Learning to Explore using Active Neural SLAM*, ICLR 2020, https://arxiv.org/abs/2004.05155 (coverage reward, PPO settings for goal selection)
- Tan et al., *Deep RL for Decentralized Multi-Robot Exploration With Macro Actions*, https://arxiv.org/abs/2110.02181 (frontier-level macro actions)
- *A Multi-Robot Collaborative Exploration Method Based on DRL and Knowledge Distillation*, Mathematics 13(1):173, 2025, https://www.mdpi.com/2227-7390/13/1/173 (teacher warm start)
- Huang & Ontañón, *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, 2020, and sb3-contrib `MaskablePPO` (action masking)
