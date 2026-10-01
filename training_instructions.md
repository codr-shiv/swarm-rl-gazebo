# Training Instructions (fresh PC / server)

These steps take a clean machine to trained frontier weights. Training means tuning the 5 weights of the frontier formula with CMA-ES on the headless Gazebo stack (see [docs/weight_tuning.md](docs/weight_tuning.md) for the details).

**Requirements**
- Ubuntu 22.04. On another Linux, use a distrobox with Ubuntu 22.04 (step 0).
- At least 8 CPU cores and 16 GB RAM. More cores means more parallel sims, so training finishes faster.
- No GPU and no display needed.
- About 10 GB of disk.

---

## Quick start: your laptop (8 cores, 32 GB RAM, distrobox `ros-humble`)

Everything is already installed, and `~/.bashrc` sources `~/swarm/install` and sets `TURTLEBOT3_MODEL=burger`. So it's just:

1. **Host (Fedora) terminal:** stop the laptop from sleeping, and enable loopback multicast (needed for ROS discovery when Wi-Fi is off; repeat after each reboot):
   ```bash
   sudo ip link set lo multicast on
   systemd-inhibit --what=sleep:idle:handle-lid-switch sleep 24h &
   ```
2. **Close heavy apps** (browser, IDE), plug in the charger, and keep it ventilated.
3. **Build and smoke-test** (~3 min):
   ```bash
   distrobox enter ros-humble
   cd ~/swarm && rm -rf build install log && colcon build --symlink-install && source install/setup.bash
   ros2 run multi_robot_exploration evaluate_weights --weights heuristic --seeds 42 --max-episode-sim-s 120
   ```
   **Pass:** it prints `seed=42 end=... score=...`.
4. **Train.** This is the ideal command for 8 cores / 32 GB:
   ```bash
   tmux new -s tuning
   ros2 run multi_robot_exploration tune_weights --num-envs 6 --run-dir ~/swarm_tuning_runs/main 2>&1 | tee -a ~/swarm_tuning_runs_main.log
   ```
   - **Why 6 sims:** each needs about 1.3 physical cores and 2 GB RAM. 6 leaves headroom for the tuner and OS; 7–8 would oversubscribe the 8 cores.
   - **Worlds:** keep the default 6 worlds per candidate, which gives the least noisy result.
   - **Time:** about 45–50 min per generation, so 20 generations take about 15–17 h, i.e. 2 nights. Detach with `Ctrl+B` then `D`.
5. **Stop and continue across nights.** `Ctrl+C` in tmux stops it (only the current generation is lost). Continue the next night with:
   ```bash
   ros2 run multi_robot_exploration tune_weights --num-envs 6 --run-dir ~/swarm_tuning_runs/main --resume 2>&1 | tee -a ~/swarm_tuning_runs_main.log
   ```
6. **Result:** `cat ~/swarm_tuning_runs/main/best_weights.json`. Then do step 9 below (compare on held-out worlds) and step 10 (use the weights).

If the first generation shows `failed_episodes` above 0, or `episode exceeded 3x real time`, the laptop is overloaded. `Ctrl+C` and `--resume` with `--num-envs 5`.

---

## Fresh PC / server

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

**What you'll see:** one line per finished episode, then a summary line per generation:
```
gen 0: running 60 episodes on 8 sims (one line per finished episode)...
  [slot 3] cand5     world 1826701614: score  1.583, saturated after 212 s, 2 failed goals
  ...
gen 0: candidates 1.124 (best 1.583) | current mean 1.131 vs heuristic 0.960 on the same worlds | sigma 0.857 | 34 min | weights {...}
```
The first episode lines appear about 5 minutes after start. The sims start 5 s apart, and each episode is ~35 s of startup plus up to 300 s of exploring. Warnings like `Publisher already registered` or `interface lo is not multicast-capable` are harmless if episode lines keep appearing.

**What it recovers from automatically:**
- A failed episode (sim didn't start, froze, or crashed) is retried once.
- A worker process that dies (e.g. out of memory) is detected: its sim is killed, the workers restart, and the lost episodes are redone.
- If more than half of a generation's episodes fail, the generation is re-run once. If it fails again, the tuner stops **without** changing the weights, prints the likely cause, and you continue with `--resume` after fixing it.

**Check progress** (from any terminal):
```bash
column -s, -t < ~/swarm_tuning_runs/main/generations.csv
```
Good signs:
- `mean_weights_score` sits above `heuristic_score` for most generations. Judge the trend over several generations, because single rows are noisy.
- `sigma` shrinks from about 1.0 towards 0.05.
- `failed_episodes` stays at 0.

**If the PC reboots, the run dies, or you stopped it with Ctrl+C,** resume from the last finished generation. Only the generation in progress is lost:
```bash
ros2 run multi_robot_exploration tune_weights --num-envs 8 --run-dir ~/swarm_tuning_runs/main --resume 2>&1 | tee -a ~/swarm_tuning_runs_main.log
```
Without `--resume`, the tuner refuses to start in a run directory that already has a state, so you can't overwrite a run by accident. Leftover sims from a crash are killed automatically at start. Keep `--worlds` and `--max-episode-sim-s` the same when resuming.

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
| Every episode fails with `nav2_active: False` / `map: False` | ROS nodes can't discover each other, usually because there's no network connection. Run `sudo ip link set lo multicast on` (again after each reboot), then `--resume`. |
| `episode exceeded 3x real time`, lots of `failed_episodes`, or the machine slows to a crawl | Too many sims for the hardware. Stop the run (`Ctrl+C` in tmux), then resume with a smaller `--num-envs`. |
| `already has a tuning state` | You restarted without `--resume`. Add it, or choose a new `--run-dir`. |
| Sims still running after the tuner was killed with `kill -9` | Starting the tuner again (with `--resume`) kills them automatically. To do it by hand: `ps -eo pid,pgid,cmd \| grep headless_stack`, then `kill -INT -<pgid>`. |
| Another tuning or evaluation job is already running | Give each job its own slots: `--first-instance 20` for `tune_weights`, `--instance-id 30` for `evaluate_weights`. |
