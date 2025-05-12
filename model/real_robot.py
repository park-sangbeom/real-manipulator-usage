import math
import numpy as np
from math import pi

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState


class RealRobot(Node):
    def __init__(self):
        super().__init__('real_robot_controller')

        self.JOINT_NAMES = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.client = ActionClient(self, FollowJointTrajectory, '/follow_joint_trajectory')
        self.joint_states_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joint_positions = None

    def joint_state_callback(self, msg):
        self.current_joint_positions = msg.position

    def wait_for_joint_state(self):
        while rclpy.ok() and self.current_joint_positions is None:
            self.get_logger().info('Waiting for joint_states...')
            rclpy.spin_once(self)
        return self.current_joint_positions

    def build_trajectory(self, positions_list, time_list):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.JOINT_NAMES
        points = []
        for pos, t in zip(positions_list, time_list):
            pt = JointTrajectoryPoint()
            pt.positions = pos
            pt.velocities = [0.0] * 6
            pt.time_from_start.sec = t
            points.append(pt)
        goal.trajectory.points = points
        return goal

    def capture_pose(self):
        q = [
            -90.0 / 180 * math.pi,
            -132.46 / 180 * math.pi,
            122.85 / 180 * math.pi,
            99.65 / 180 * math.pi,
            45.0 / 180 * math.pi,
            -90.02 / 180 * math.pi
        ]
        current_pos = self.wait_for_joint_state()
        goal = self.build_trajectory([list(current_pos), q], [0, 3])

        self.get_logger().info("Sending capture pose goal...")
        self.client.wait_for_server()
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Capture pose executed.")

    def execute_trajectory(self, joint_list):
        positions = []
        durations = []

        current_pos = self.wait_for_joint_state()
        positions.append(list(current_pos))
        durations.append(0)

        d = 3
        for q in joint_list:
            positions.append(q)
            durations.append(d)
            d += 1  # or use fixed small step like 0.01 if smoother is needed

        goal = self.build_trajectory(positions, durations)
        self.get_logger().info("Sending trajectory goal...")
        self.client.wait_for_server()
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Trajectory finished.")

    def move_capture_pose(self):
        user_input = input("Continue to capture pose? (y/n): ")
        if user_input.lower() == 'y':
            self.capture_pose()
        else:
            self.get_logger().info("Capture pose canceled.")

    def move_arm(self, joint_list):
        user_input = input("Continue to move arm? (y/n): ")
        if user_input.lower() == 'y':
            self.execute_trajectory(joint_list)
        else:
            self.get_logger().info("Trajectory execution canceled.")


def main():
    rclpy.init()
    robot = RealRobot()
    robot.move_capture_pose()
    robot.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()