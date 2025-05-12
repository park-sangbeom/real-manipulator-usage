import rclpy
from rclpy.node import Node
import numpy as np
import math
import sys
sys.path.append('..')

from model.real_robot import RealRobot
from model.sim_robot import SimRobot

xml_file = './urdf/UR5e.xml'

class RobotCommander(Node):
    def __init__(self):
        super().__init__('move_node')
        self.sim_robot = SimRobot(xml_path=xml_file)
        self.real_robot = RealRobot()

    def capture_pose(self):
        """Single trajectory"""
        self.real_robot.move_capture_pose()
        self.get_logger().info("Capture pose done.")

    def move_to_pose(self, desired_position, desired_rotation):
        """Multiple trajectories"""
        interp_q = self.sim_robot.simple_move(
            desired_position=desired_position,
            desired_rotation=desired_rotation,
            freq=500,
            desired_time=8
        )
        self.real_robot.move_arm(joint_list=interp_q)
        self.get_logger().info("Trajectory finished.")

def main(args=None):
    rclpy.init(args=args)
    node = RobotCommander()

    desired_position = np.array([0.9, 0.0, 1.0], dtype=object)
    desired_rotation = np.array([-math.pi/2, 0.0, -math.pi/2], dtype=object)

    node.move_to_pose(desired_position, desired_rotation)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()