import sys 
sys.path.append('..')
import roslib; roslib.load_manifest('ur_driver')
import rospy  
import numpy as np 
import math 
from model.real_robot import RealRobot
from model.sim_robot import SimRobot

xml_file   = './urdf/UR5e.xml'
sim_robot  = SimRobot(xml_path = xml_file)
real_robot = RealRobot()

def capture_pose():
    """ 
    Example: Single trajectory 
    """ 
    real_robot.move_capture_pose()
    print("Capture pose done.")

def main(desired_position=np.array([0.9, 0, 1.0], dtype=object), 
        desired_rotation=np.array([-math.pi/2, 0, -math.pi/2], dtype=object)):
    """
    Example: Multiple trajectories
    """
    interp_q = sim_robot.simple_move(desired_position=desired_position, 
                                    desired_rotation=desired_rotation,
                                    freq=500, 
                                    desired_time=8)
    real_robot.move_arm(joint_list=interp_q)
    print("Trajectory finished.")

if __name__ == "__main__":
    rospy.init_node("move")
    desired_position=np.array([0.9, 0, 1.0], dtype=object), 
    desired_rotation=np.array([-math.pi/2, 0, -math.pi/2], dtype=object)
    main(desired_position=np.array([0.9, 0, 1.0], dtype=object), 
         desired_rotation=np.array([-math.pi/2, 0, -math.pi/2], dtype=object))