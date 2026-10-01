"""
Starts / stops one headless simulation stack (headless_stack.launch.py) in
its own process group, ROS domain and Gazebo port, so several can run in
parallel and each can be torn down completely without touching anything
else on the machine (pkill-style cleanup would also hit a sim you are
running by hand, since distrobox shares the host's process list).
"""
import glob
import os
import shutil
import signal
import subprocess
import time

BASE_DOMAIN_ID = 40
BASE_GAZEBO_PORT = 11445


class SimManager:

    def __init__(self, instance_id=0, gui=False, rtf=1.0, log_dir='/tmp'):
        self.instance_id = instance_id
        self.domain_id = BASE_DOMAIN_ID + instance_id
        self.gazebo_port = BASE_GAZEBO_PORT + instance_id
        self.gui = gui
        self.rtf = rtf
        self.log_dir = log_dir
        self.proc = None
        self._log = None

    def start(self, world_seed):
        self.stop()
        env = dict(os.environ)
        env.update({
            'ROS_DOMAIN_ID': str(self.domain_id),
            'GAZEBO_MASTER_URI': f'http://127.0.0.1:{self.gazebo_port}',
            'GAZEBO_WORLD_SEED': str(world_seed),
            'GAZEBO_RTF': str(self.rtf),
            'TURTLEBOT3_MODEL': 'burger',
            'GAZEBO_MODEL_PATH': env.get('GAZEBO_MODEL_PATH', '') +
                ':/opt/ros/humble/share/turtlebot3_gazebo/models',
        })
        for k in ('USE_HOUSE', 'USE_TB3_WORLD'):
            env.pop(k, None)
        if not self.gui:
            env.pop('DISPLAY', None)
            env.pop('WAYLAND_DISPLAY', None)

        # Per-slot ROS log dir, wiped every episode: ~/.ros/log would otherwise
        # grow by ~1 MB and dozens of folders per episode.
        ros_logs = os.path.join(self.log_dir, f'ros_logs_slot{self.instance_id}')
        shutil.rmtree(ros_logs, ignore_errors=True)
        env['ROS_LOG_DIR'] = ros_logs

        os.makedirs(self.log_dir, exist_ok=True)
        self._log = open(os.path.join(self.log_dir, f'sim_{self.instance_id}.log'), 'w')
        self.proc = subprocess.Popen(
            ['ros2', 'launch', 'multi_robot_exploration', 'headless_stack.launch.py',
             f'gui:={"true" if self.gui else "false"}'],
            env=env, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True)   # own process group -> clean killpg
        # Record the process group so a crashed worker's sim can still be killed
        with open(self._pgid_file(), 'w') as fh:
            fh.write(str(self.proc.pid))

    def _pgid_file(self):
        return os.path.join(self.log_dir, f'sim_{self.instance_id}.pgid')

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        pgid = self.proc.pid   # session leader: pgid == pid
        # SIGINT to `ros2 launch` only (it shuts its nodes down in order);
        # then escalate on the whole group for anything left behind.
        for sig, wait_s, kill in ((signal.SIGINT, 20.0, os.kill),
                                  (signal.SIGTERM, 5.0, os.killpg),
                                  (signal.SIGKILL, 5.0, os.killpg)):
            try:
                kill(pgid, sig)
            except ProcessLookupError:
                if kill is os.kill and self._group_alive(pgid):
                    continue   # launch already gone but children remain
                break
            deadline = time.time() + wait_s
            while time.time() < deadline and self._group_alive(pgid):
                time.sleep(0.2)
            if not self._group_alive(pgid):
                break
        try:
            self.proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None
        try:
            os.remove(self._pgid_file())
        except OSError:
            pass

    def _group_alive(self, pgid):
        self.proc.poll()   # reap the leader, or its zombie keeps the group "alive"
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _group_members(pgid):
    """PIDs whose process group is *pgid* (reads /proc; works across distrobox)."""
    pids = []
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as fh:
                fields = fh.read().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                pids.append(int(d))
        except (OSError, IndexError, ValueError):
            continue
    return pids


def kill_stale_sims(log_dir):
    """
    Kill sims recorded in log_dir/sim_*.pgid whose worker died without
    cleaning up (crash, OOM kill, kill -9). Only process groups that still
    contain ROS/Gazebo processes are touched.
    """
    killed = 0
    for path in glob.glob(os.path.join(log_dir, 'sim_*.pgid')):
        try:
            with open(path) as fh:
                pgid = int(fh.read().strip())
        except (OSError, ValueError):
            pgid = None
        members = _group_members(pgid) if pgid else []
        cmdlines = []
        for pid in members:
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as fh:
                    cmdlines.append(fh.read().decode(errors='ignore'))
            except OSError:
                pass
        if any(k in c for c in cmdlines for k in ('ros', 'gzserver', 'gazebo')):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                deadline = time.time() + 5.0
                while time.time() < deadline and _group_members(pgid):
                    time.sleep(0.2)
                if not _group_members(pgid):
                    break
            killed += 1
        try:
            os.remove(path)
        except OSError:
            pass
    return killed
