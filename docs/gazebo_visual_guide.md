# Gazebo Visual & Cinematic Enhancement Guide

A step-by-step setup to turn standard Gazebo Classic simulations into high-contrast, broadcast-ready 3D demo captures for multi-agent swarm projects.

---

## 1. Remove LiDAR Ray Fan Overlay

The solid blue LiDAR fans obscure robots, obstacles, and navigation paths.

* **GUI Method**: In the top menu, go to `View` and uncheck `Ray Clips` (or `Laser Scans`).
* **URDF / Xacro Method**: Locate the sensor definition block and disable visual rendering:
  ```xml
  <sensor name="lidar" type="ray">
    <visualize>false</visualize> <!-- Set from true to false -->
    <ray>
      <!-- scan properties -->
    </ray>
  </sensor>
  ```
