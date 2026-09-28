import math
import random
import tempfile
import os

def generate_random_world(
    arena_size_x=8.0,
    arena_size_y=8.0,
    num_obstacles_min=8,
    num_obstacles_max=15,
    wall_height=0.5,
    wall_thickness=0.15,
    obstacle_clearance=0.8,
    robot_clearance=0.8,
    robot_separation=2.0,
    seed=None,
    real_time_factor=1.0
):
    """
    Generates a random Gazebo world with walls, obstacles, and valid spawn poses.
    real_time_factor > 1 asks Gazebo to simulate faster than real time.
    
    Returns:
        tuple: (path_to_world_file, [(x1, y1), (x2, y2)])
    """
    if seed is not None:
        random.seed(seed)
        
    num_obstacles = random.randint(num_obstacles_min, num_obstacles_max)
    obstacles = []
    
    def is_valid_obstacle_pos(x, y, radius):
        # Check against walls
        if (x - radius < -arena_size_x/2 + wall_thickness or
            x + radius > arena_size_x/2 - wall_thickness or
            y - radius < -arena_size_y/2 + wall_thickness or
            y + radius > arena_size_y/2 - wall_thickness):
            return False
            
        # Check against other obstacles
        for ox, oy, oradius, _ in obstacles:
            dist = math.hypot(x - ox, y - oy)
            if dist < (radius + oradius + obstacle_clearance):
                return False
        return True

    # Generate obstacles
    for i in range(num_obstacles):
        for _ in range(50):  # max attempts per obstacle
            is_box = random.choice([True, False])
            
            if is_box:
                sx = random.uniform(0.3, 0.8)
                sy = random.uniform(0.3, 0.8)
                radius = math.hypot(sx/2, sy/2)
                type_data = ('box', sx, sy)
            else:
                r = random.uniform(0.15, 0.4)
                radius = r
                type_data = ('cylinder', r)
                
            x = random.uniform(-arena_size_x/2, arena_size_x/2)
            y = random.uniform(-arena_size_y/2, arena_size_y/2)
            
            if is_valid_obstacle_pos(x, y, radius):
                obstacles.append((x, y, radius, type_data))
                break

    # Generate robot spawns
    spawns = []
    for r_idx in range(2):
        for _ in range(1000):
            x = random.uniform(-arena_size_x/2 + wall_thickness + robot_clearance, 
                               arena_size_x/2 - wall_thickness - robot_clearance)
            y = random.uniform(-arena_size_y/2 + wall_thickness + robot_clearance, 
                               arena_size_y/2 - wall_thickness - robot_clearance)
            
            # Check against obstacles
            valid = True
            for ox, oy, oradius, _ in obstacles:
                dist = math.hypot(x - ox, y - oy)
                if dist < (oradius + robot_clearance):
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check against other spawns
            if spawns:
                for sx, sy in spawns:
                    if math.hypot(x - sx, y - sy) < robot_separation:
                        valid = False
                        break
            
            if valid:
                spawns.append((x, y))
                break
                
        if len(spawns) <= r_idx: # fallback if failed
            if r_idx == 0:
                spawns.append((-arena_size_x/2 + 1.0, -arena_size_y/2 + 1.0))
            else:
                spawns.append((arena_size_x/2 - 1.0, arena_size_y/2 - 1.0))

    # Build SDF
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
    <scene>
      <ambient>0.4 0.4 0.4 1.0</ambient>
      <background>0.12 0.14 0.18 1.0</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>
    <physics type="ode">
      <real_time_update_rate>{1000.0 * real_time_factor}</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>{real_time_factor:g}</real_time_factor>
    </physics>
"""

    # Walls
    sdf += f"""
    <model name="arena_walls">
      <static>true</static>
      <!-- North Wall -->
      <link name="wall_n">
        <pose>0 {arena_size_y/2} {wall_height/2} 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>{arena_size_x + wall_thickness*2} {wall_thickness} {wall_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{arena_size_x + wall_thickness*2} {wall_thickness} {wall_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/DarkGrey</name></script></material>
        </visual>
      </link>
      <!-- South Wall -->
      <link name="wall_s">
        <pose>0 {-arena_size_y/2} {wall_height/2} 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>{arena_size_x + wall_thickness*2} {wall_thickness} {wall_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{arena_size_x + wall_thickness*2} {wall_thickness} {wall_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/DarkGrey</name></script></material>
        </visual>
      </link>
      <!-- East Wall -->
      <link name="wall_e">
        <pose>{arena_size_x/2} 0 {wall_height/2} 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>{wall_thickness} {arena_size_y} {wall_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{wall_thickness} {arena_size_y} {wall_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/DarkGrey</name></script></material>
        </visual>
      </link>
      <!-- West Wall -->
      <link name="wall_w">
        <pose>{-arena_size_x/2} 0 {wall_height/2} 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>{wall_thickness} {arena_size_y} {wall_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{wall_thickness} {arena_size_y} {wall_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/DarkGrey</name></script></material>
        </visual>
      </link>
    </model>
"""

    # Add obstacles
    for i, (ox, oy, oradius, type_data) in enumerate(obstacles):
        sdf += f"""
    <model name="obs_{i}">
      <static>true</static>
      <pose>{ox} {oy} {0.5/2} 0 0 0</pose>
      <link name="link">
        <collision name="collision">"""
        if type_data[0] == 'box':
            geom = f"<box><size>{type_data[1]} {type_data[2]} 0.5</size></box>"
        else:
            geom = f"<cylinder><radius>{type_data[1]}</radius><length>0.5</length></cylinder>"
            
        sdf += f"""
          <geometry>{geom}</geometry>
        </collision>
        <visual name="visual">
          <geometry>{geom}</geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/White</name></script></material>
        </visual>
      </link>
    </model>"""

    sdf += """
  </world>
</sdf>
"""

    # Write to a named temporary file
    fd, path = tempfile.mkstemp(suffix='.world', prefix='random_world_')
    with os.fdopen(fd, 'w') as f:
        f.write(sdf)

    return path, spawns
