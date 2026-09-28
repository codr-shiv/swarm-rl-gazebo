# RViz Visual Enhancement Guide: Multi-Robot Swarm Demos

This guide outlines practical techniques to transform default, academic-looking RViz windows into high-contrast, cinematic, and professional visualization viewports tailored for multi-robot exploration and frontier navigation demos.

---

## 1. Global Viewport & Canvas Styling

Default RViz configurations use flat medium-grey backgrounds that wash out laser scans, trajectories, and costmaps. Shifting to an OLED-level dark canvas makes neon colors pop and gives the interface a modern "mission control" aesthetic.

### Background & Grid Configuration

In the **Displays** panel under **Global Options**:

* **Background Color**: Change from default `(48, 48, 48)` to deep slate/black:
  * **RGB**: `5, 7, 14` (Hex: `#05070E`)
* **Fixed Frame**: Set to your global frame (typically `map` or `odom`).

In the **Grid** display:

* **Plane Cell Count**: `50` to `100` (large enough to span the environment).
* **Color**: Set to subtle dark slate:
  * **RGB**: `30, 35, 48` (Hex: `#1E2330`)
* **Alpha**: Reduce to `0.15` – `0.20`. *(The grid should provide subtle depth without visually competing with occupancy cells).*

---

## 2. Multi-Robot Identity & Trajectory Ribbons

Avoid using generic thin red lines for all robots. Assign high-contrast complementary color schemes to each agent across their paths, footprints, and markers.

### Agent Color Palette

| Entity | Robot 1 (Lead Scout) | Robot 2 (Support / Split Scout) | Shared / Conflict |
|---|---|---|---|
| **Theme Color** | **Electric Cyan** | **Neon Orange / Amber** | **Vibrant Magenta / Gold** |
| **Hex Code** | `#00F0FF` | `#FF5500` | `#E024C3` / `#FFD700` |
| **RGB (0–255)** | `(0, 240, 255)` | `(255, 85, 0)` | `(224, 36, 195)` |
| **Normalized Float** | `(0.0, 0.94, 1.0)` | `(1.0, 0.33, 0.0)` | `(0.88, 0.14, 0.76)` |

### Path Display (`nav_msgs/msg/Path`)

1. Add a **Path** display for each robot's planned trajectory:
   * **Topic**: `/robot1/plan` and `/robot2/plan`
   * **Color**: Set to corresponding theme color.
   * **Alpha**: `1.0`
   * **Line Style**: Change from `Lines` to `Billboards` or increase line width parameter if using custom marker trajectories (`0.06 m` to `0.10 m`).

### Persistent Path History (Trails)

To show the area covered over time, publish a history trail of past poses or use a **PoseArray** / **MarkerArray** with a line strip:

* **Line Width**: `0.04 m`
* **Alpha**: `0.45` (subtle trailing ribbon behind the robot).

---

## 3. Map & Costmap Visualization

### Global Occupancy Grid (`nav_msgs/msg/OccupancyGrid`)

* **Topic**: `/map`
* **Color Scheme**:
  * Default `map` gives flat grey/black/white.
  * Switch to `costmap` or `raw` depending on preference.
  * If using custom color schemes: Set **Occupied Space** to high-contrast white/light-cyan, and **Free Space** to deep translucent blue-black (`Alpha = 0.7`).
* **Unexplored Area**: Ensure unexplored space (`-1`) matches or seamlessly blends into the global canvas background (`Alpha = 0.0` or background color).

### Local Costmaps (`nav_msgs/msg/OccupancyGrid`)

* **Alpha**: `0.35` – `0.50`
* **Color Scheme**: `costmap` (creates a vibrant heatmap bubble around each robot showing collision margins in real-time).

---

## 4. Frontier Marker Optimization (`visualization_msgs/msg/MarkerArray`)

Frontiers should look like active tactical objectives rather than raw point clouds.
