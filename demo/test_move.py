import time
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]

Q1 = [0, 0, -1.57, 0, 0, 0]
Q2 = [0, 0, -1.57, 0, 0, 0]
Q3 = [0, -0.2, -1.57, 0, 0, 0]

class UR5eCommander(Node):
    def __init__(self):
        super().__init__('ur5e_commander')
        self.client = ActionClient(self, FollowJointTrajectory, '/follow_joint_trajectory')
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joint_positions = None

    def joint_state_callback(self, msg):
        self.current_joint_positions = msg.position

    def wait_for_joint_state(self):
        while rclpy.ok() and self.current_joint_positions is None:
            self.get_logger().info('Waiting for joint_states...')
            rclpy.spin_once(self)
        return self.current_joint_positions

    def build_goal(self, poses, durations):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = JOINT_NAMES

        points = []
        for i, q in enumerate(poses):
            pt = JointTrajectoryPoint()
            pt.positions = q
            pt.velocities = [0.0] * 6
            pt.time_from_start.sec = durations[i]
            points.append(pt)

        goal_msg.trajectory.points = points
        return goal_msg

    def move_repeated(self):
        self.get_logger().info("Waiting for FollowJointTrajectory action server...")
        self.client.wait_for_server()

        current_pos = self.wait_for_joint_state()
        poses = [list(current_pos)]
        durations = [0]

        d = 2
        for _ in range(10):
            poses += [Q1, Q2, Q3]
            durations += [d, d + 1, d + 2]
            d += 3

        goal_msg = self.build_goal(poses, durations)
        self.get_logger().info("Sending repeated trajectory goal...")
        future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Trajectory execution finished.")

def main():
    rclpy.init()
    node = UR5eCommander()

    print("This program makes the robot move between the following three poses:")
    print(str([q * 180. / math.pi for q in Q1]))
    print(str([q * 180. / math.pi for q in Q2]))
    print(str([q * 180. / math.pi for q in Q3]))
    inp = input("Continue? y/n: ")
    if inp.strip().lower() == 'y':
        node.move_repeated()
    else:
        print("Halting program.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()