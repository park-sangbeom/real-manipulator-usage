# 🤖 real-manipulator-usage

This package provides a minimal working example for controlling a **UR5e robot arm** using ROS.  
It is intended for real-world manipulator experiments, leveraging a [customized-kinematics package](https://github.com/park-sangbeom/kinematics-usage) and ROS control interfaces.

## 🛠 Prerequisites

Before using this package, ensure the following software and dependencies are properly installed and configured:

### ✅ 1. System Requirements
- Ubuntu 22.04
- ROS 2 Humble Hawksbill or later
- Python 3.8+
- Mujoco 2.3.3 (or compatible version)
- NVIDIA GPU (optional, for faster simulation)

---

### ✅ 2. Python Dependencies

Install via `pip`:

```bash
pip install numpy gym mujoco matplotlib transforms3d
```

### ✅ 3. ROS 2 Dependencies (UR5e Control)

| Package                             | Purpose                          |
|-------------------------------------|----------------------------------|
| `ur_description`                    | URDF model for UR5e              |
| `ur_gazebo` / `ur5e_moveit_config`  | Simulation and MoveIt config     |
| `ur_robot_driver`                   | Real-time hardware interface     |
| `controller_manager`                | Controller loading & management  |
| `ros_control` / `ros_controllers`  | Low-level joint control          |

You can install these via `rosdep` or clone official packages from:

- https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
- https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver
- https://github.com/ros-controls/ros2_control
- https://github.com/ros-controls/ros2_controllers

---


## 📁 Demo

A basic test script is available in the `demo/` folder:

- `demo/test_move.py`  
  → Sends target joint trajectories to the UR5e robot using the ROS action interface.

---

## ▶️ How to Run

To execute a full motion planning for a real robot, run:

```bash
python3 main.py
```

