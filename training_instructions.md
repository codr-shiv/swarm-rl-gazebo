# Training Instructions (fresh PC / server)

These steps take a clean machine to trained frontier weights. Training means tuning the 5 weights of the frontier formula with CMA-ES on the headless Gazebo stack (see [docs/weight_tuning.md](docs/weight_tuning.md) for the details).

**Requirements**
- Ubuntu 22.04. On another Linux, use a distrobox with Ubuntu 22.04 (step 0).
- At least 8 CPU cores and 16 GB RAM. More cores means more parallel sims, so training finishes faster.
- No GPU and no display needed.
- About 10 GB of disk.

**Before you start (on your laptop):** commit and push this repo, so the PC can clone the current version:
```bash
cd ~/swarm && git add -A && git commit -m "frontier weight tuning" && git push
```

---

## 0. (Only if the PC is not Ubuntu 22.04) Create an Ubuntu 22.04 distrobox
```bash
distrobox create --name ros-humble --image ubuntu:22.04
distrobox enter ros-humble
```
Run every following step inside the distrobox.

## 1. Install ROS 2 Humble
```bash
sudo apt update && sudo apt install -y software-properties-common curl git
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install -y ros-humble-ros-base
```

## 2. Install the simulation stack
```bash
sudo apt install -y \
  ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description \
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
  ros-humble-rmw-cyclonedds-cpp \
  python3-numpy python3-opencv python3-colcon-common-extensions tmux
```
No pip packages are needed.

## 3. Get the code and build
```bash
git clone git@github.com:codr-shiv/swarm-rl-gazebo.git ~/swarm
cd ~/swarm
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```
The build ends with `Summary: 1 package finished`.

> If the build fails with `canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'`, a pip-installed setuptools is shadowing the system one. Run `pip3 uninstall -y setuptools` and build again.

## 4. Environment (every new terminal)
Add these lines to `~/.bashrc` once, so every terminal has them:
```bash
cat >> ~/.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
[ -f /usr/share/gazebo/setup.sh ] && source /usr/share/gazebo/setup.sh
source ~/swarm/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export TURTLEBOT3_MODEL=burger
EOF
source ~/.bashrc
```
Check that the commands exist:
```bash
ls ~/swarm/install/multi_robot_exploration/lib/multi_robot_exploration/
# must include: tune_weights  evaluate_weights  tuned_frontier_coordinator
```

## 5. Smoke test (~3 min)
```bash
ros2 run multi_robot_exploration evaluate_weights --weights heuristic --seeds 42 --max-episode-sim-s 120
```
**Pass:**
- It prints a line like `seed=42 end=... score=... explored=... m2`, then a summary.
- `ps -eo cmd | grep gzserver | grep -v grep` prints nothing afterwards.

If it prints `sim not ready ...`, look at `~/swarm_tuning_runs/eval_sim_logs/sim_9.log`.

## 6. Pick the number of parallel sims
Each sim needs about 2 CPU cores and 2 GB RAM:
```bash
echo "cores: $(nproc), RAM GB: $(free -g | awk '/Mem:/{print $2}')"
```
`NUM_ENVS` is the smaller of cores ÷ 2 and RAM GB ÷ 2, e.g. 16 cores / 32 GB → 8. Leave a core or two free if the machine does other work.

| NUM_ENVS | Time per generation | Full run (20 generations) |
|---|---|---|
| 4 | ~70 min | ~1 day |
| 8 | ~35 min | ~12 h |
| 12 | ~25 min | ~8 h |

## 7. Train (the final command)
Run it inside `tmux`, so it keeps going if your SSH session drops:
```bash
tmux new -s tuning
```
Then, inside tmux, with `8` replaced by your `NUM_ENVS`:
```bash
ros2 run multi_robot_exploration tune_weights --num-envs 8 --run-dir ~/swarm_tuning_runs/main 2>&1 | tee ~/swarm_tuning_runs_main.log
```
- **Detach:** `Ctrl+B`, then `D`.
- **Reattach later:** `tmux attach -t tuning`.

Let it run to the end. It stops by itself after 20 generations, or earlier when the weights stop changing, and prints `Done (...)`.

**Check progress** (from any terminal):
```bash
column -s, -t < ~/swarm_tuning_runs/main/generations.csv
```
Good signs:
- `mean_weights_score` sits above `heuristic_score` for most generations. Judge the trend over several generations, because single rows are noisy.
- `sigma` shrinks from about 1.0 towards 0.05.
- `failed_episodes` stays at 0.

**If the PC reboots or the run dies,** resume from the last finished generation:
```bash
ros2 run multi_robot_exploration tune_weights --num-envs 8 --run-dir ~/swarm_tuning_runs/main --resume
```

## 8. Result
```bash
cat ~/swarm_tuning_runs/main/best_weights.json
```
The whole trained result is these 6 numbers (distance is always 1).

## 9. Confirm against the heuristic on held-out worlds (~40 min each)
The two commands can run at the same time, because they use different sim slots:
```bash
ros2 run multi_robot_exploration evaluate_weights --weights heuristic --instance-id 9
ros2 run multi_robot_exploration evaluate_weights --weights ~/swarm_tuning_runs/main/best_weights.json --instance-id 10
column -s, -t < ~/swarm_tuning_runs/eval.csv
```
The tuned weights are better if their mean `score` in the summary line is higher: a shorter time to finish mapping, fewer failed goals, or both.

## 10. Use the weights
Copy `best_weights.json` to your laptop, then run the normal pipeline (terminals 1–4). In terminal 5, run:
```bash
ros2 launch multi_robot_exploration tuned_frontier_exploration.launch.py weights_file:=$HOME/best_weights.json
```

---

## Everything in one block
After steps 0–4 (install, clone, build, `.bashrc`) are done:
```bash
ros2 run multi_robot_exploration evaluate_weights --weights heuristic --seeds 42 --max-episode-sim-s 120   # smoke test
tmux new -s tuning
ros2 run multi_robot_exploration tune_weights --num-envs 8 --run-dir ~/swarm_tuning_runs/main 2>&1 | tee ~/swarm_tuning_runs_main.log
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `Package 'multi_robot_exploration' not found` | Run `source ~/swarm/install/setup.bash`, or open a new terminal after step 4. |
| `sim not ready (attempt N): {...}` | The dict shows what's missing (`nav2_active`, `map`, `tf`). See `~/swarm_tuning_runs/main/sim_logs/sim_<slot>.log`. Occasional ones are retried automatically. |
| Lots of `failed_episodes`, or a machine slows to a crawl | Too many sims for the hardware. Stop the run (`Ctrl+C` in tmux), then resume with a smaller `--num-envs`. |
| Sims still running after a crash | Run `ps -eo pid,pgid,cmd \| grep headless_stack`, then `kill -INT -<pgid>` for each. |
| Another tuning or evaluation job is already running | Give each job its own slots: `--first-instance 20` for `tune_weights`, `--instance-id 30` for `evaluate_weights`. |
