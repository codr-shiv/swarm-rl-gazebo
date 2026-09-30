"""
Starts / stops one headless simulation stack (headless_stack.launch.py) in
its own process group, ROS domain and Gazebo port, so several can run in
parallel and each can be torn down completely without touching anything
else on the machine (pkill-style cleanup would also hit a sim you are
running by hand, since distrobox shares the host's process list).
"""
import os
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

        os.makedirs(self.log_dir, exist_ok=True)
        self._log = open(os.path.join(self.log_dir, f'sim_{self.instance_id}.log'), 'w')
        self.proc = subprocess.Popen(
            ['ros2', 'launch', 'multi_robot_exploration', 'headless_stack.launch.py',
             f'gui:={"true" if self.gui else "false"}'],
            env=env, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True)   # own process group -> clean killpg

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

    def _group_alive(self, pgid):
        self.proc.poll()   # reap the leader, or its zombie keeps the group "alive"
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
