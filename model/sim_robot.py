import xml.etree.ElementTree as ET
from gym.envs.mujoco import mujoco_env
from gym import utils
import math
import os 
import numpy as np 

class SimRobot(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 name       = 'UR5e',
                 xml_path   = '../urdf/UR5e.xml',
                 frame_skip = 5):
        self.name       = name
        self.xml_path   = os.path.abspath(xml_path)
        self.frame_skip = frame_skip
        # Open xml
        self.xml        = open(xml_path, 'rt', encoding='UTF8')
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
        )
        utils.EzPickle.__init__(self) 

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]
        # Reset
        self.reset()
        self.set_config()
        
    def step(self,a):
        """
            Step forward
        """
        # Run sim
        self.do_simulation(a, self.frame_skip)
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = 0.0
        self.d    = False
        self.info = dict()
        return self.o,self.r,self.d,self.info

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[0],q[1],q[2],q[3],q[4],q[5]]
            )*180.0/np.pi

    def _get_obs(self):
        """
            Get observation
        """
        o = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext,-1,1).flat
        ])
        return o

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time

    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def set_config(self, VERBOSE=False):
        """
            Print a robot configuration from Unity
        """
        dt                   = 0.02
        self.HZ              = int(1/dt)
        self.body_names      = list(self.model.body_names)
        self.eef_name        = 'wrist_3_link'
        self.n_joint         = self.model.njnt
        self.joint_names     = [self.model.joint_id2name(x) for x in range(self.n_joint)]
        self.joint_types     = self.model.jnt_type # 0:free, 1:ball, 2:slide, 3:hinge
        self.rev_joint_idxs  = np.where(self.joint_types==3)[0].astype(np.int32) # revolute joint
        self.rev_joint_names = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.rev_qvel_idxs   = [self.model.get_joint_qvel_addr(x) for x in self.rev_joint_names]
        self.n_rev_joint     = len(self.rev_joint_idxs)
        self.pri_joint_idxs  = np.where(self.joint_types==2)[0].astype(np.int32) # gripper joint
        self.pri_joint_names = [self.joint_names[x] for x in self.pri_joint_idxs]
        self.joint_range     = self.model.jnt_range
        self.torque_range    = self.model.actuator_ctrlrange
        if VERBOSE:
            # Print-out useful information
            print ("dt:[%.3f] HZ:[%d]"%(dt,self.HZ))
            print ("body_names      :\n%s"%(self.body_names))
            print ("eef_name        :\n%s"%(self.eef_name))
            print ("")
            print ("n_joint         : [%d]"%(self.n_joint))
            print ("joint_names     : %s"%(self.joint_names))
            print ("joint_types     : %s"%(self.joint_types))
            print ("n_rev_joint     : [%d]"%(self.n_rev_joint))
            print ("rev_joint_idxs  : %s"%(self.rev_joint_idxs))
            print ("rev_joint_names : %s"%(self.rev_joint_names))
            print ("pri_joint_idxs  : %s"%(self.pri_joint_idxs))
            print ("pri_joint_names : %s"%(self.pri_joint_names))
            print ("")
            print ("joint_range     :\n%s"%(self.joint_range))
            print ("torque_range    :\n%s"%(self.torque_range))
            print ("torque_range    :\n%s"%(self.torque_range)) 
            
    def solve_ik(self, target_name = ['wrist_3_link', "tcp_link"],
                       P_EE_des=np.array([0.5, 0, 1.1]), 
                       R_EE_des=np.array([-math.pi, 0, math.pi]), 
                       max_ik_tick=1e3):
        """
            Simple IK Solve 
        """
        ik_tick = 0
        # Convert rpy to rotation matrix
        R_mat_des = self.rpy2r(R_EE_des)
        # Env reset 
        # self.reset()
        # Constraint
        self.sim.data.qpos[self.rev_joint_idxs[1]] = np.array([-1])
        self.step(np.zeros(self.adim))
        while True:
            self.q_curr    = self.sim.data.qpos[self.rev_joint_idxs]
            self.P_EE_curr = np.array(self.data.body_xpos[self.model.body_name2id(target_name[1])])
            self.R_EE_curr = np.array(self.data.body_xmat[self.model.body_name2id(target_name[0])].reshape([3, 3]))
            self.T_EE_curr = self.pr2T(p=self.P_EE_curr,R=self.R_EE_curr)
            self.J_p_EE    = np.array(self.data.get_body_jacp(target_name[1]).reshape((3, -1))[:,self.rev_qvel_idxs])
            self.J_R_EE    = np.array(self.data.get_body_jacr(target_name[0]).reshape((3, -1))[:,self.rev_qvel_idxs])
            self.J_full_EE = np.array(np.vstack([self.J_p_EE,self.J_R_EE]))
            # Solve IK 
            self.P_EE_err  = P_EE_des - self.P_EE_curr
            # Compute Error 
            curr_joint = [self.P_EE_curr.reshape(-1,1), self.R_EE_curr]
            trgt_joint = [self.column_v(P_EE_des[0], P_EE_des[1], P_EE_des[2]), R_mat_des]
            err        = self.Cal_VWerr(trgt_joint, curr_joint).reshape(-1)
            # Jacobian 
            J          = self.J_full_EE 
            # Pseudo Inverse
            dq         = np.linalg.inv((J.T@J)+1e-6*np.eye(J.shape[1]))@J.T@err
            q_des      = self.q_curr+0.01*dq
            # Move Joint
            self.sim.data.qpos[self.rev_joint_idxs[:-1]] = q_des[:-1]
            self.sim.forward()
            # Max solve iter 
            ik_tick +=1
            if np.linalg.norm(err)<1e-12:
                break 
            if ik_tick>max_ik_tick:
                break
        q_list = np.array([0]) # Including world joint 
        q_list = np.append(q_list, self.sim.data.qpos[self.rev_joint_idxs][:-1])
        return q_list

    def pr2T(self, p,R):
        """ convert pose to transformation matrix """
        p0 = p.ravel()
        T = np.block([
            [R, p0[:, np.newaxis]],
            [np.zeros(3), 1]
        ])
        return T

    def r2w(self, R):
        el = np.array([
                [R[2,1] - R[1,2]],
                [R[0,2] - R[2,0]], 
                [R[1,0] - R[0,1]]
            ])
        norm_el = np.linalg.norm(el)
        if norm_el > 1e-10:
            w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
        elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
            w = np.array([[0, 0, 0]]).T
        else:
            w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
        return w

    def Cal_VWerr(self, cref, cnow):
        perr = (cref[0] - cnow[0]).astype(float)
        Rerr = np.matmul(np.linalg.inv(cnow[1].astype(float)), cref[1].astype(float))
        werr = np.matmul(cnow[1].astype(float), self.r2w(Rerr))
        err =  np.concatenate([perr, werr], axis=0)
        assert err.shape == (6, 1)
        return err

    def rpy2r(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
        Cphi = np.math.cos(roll)
        Sphi = np.math.sin(roll)
        Cthe = np.math.cos(pitch)
        Sthe = np.math.sin(pitch)
        Cpsi = np.math.cos(yaw)
        Spsi = np.math.sin(yaw)

        rot = np.array([
            [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
            [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
            [-Sthe, Cthe * Sphi, Cthe * Cphi]
        ])
        assert rot.shape == (3, 3)
        return rot


    def column_v(self, x,y,z):
        return np.array([[x, y, z]], dtype=object).T
    
    def q_interpolation(self, joint_val_seq, desired_time=10, freq = 500): # UR5e Hz: 500
        joint_seq_arr    = np.array(joint_val_seq, dtype=object)
        joint_seq        = joint_seq_arr.T
        new_q_list = []
        for idx in range(len(joint_seq)):
            for i in range(len(joint_seq[idx])):
                if i ==(len(joint_seq[idx])-1): break 
                if i == 0:
                    pre_q, after_q = joint_seq[idx][i:i+2]
                    interp_init_q    = np.linspace(pre_q, after_q, int(freq*(desired_time)))
                    interp_q_arr      = interp_init_q 
                else:
                    pre_q, after_q = joint_seq[idx][i:i+2]
                    interp_q       = np.linspace(pre_q, after_q, int(freq*(desired_time)))
                    interp_q_arr   = np.append(interp_q_arr, interp_q)
            new_q_list.append(interp_q_arr)
        np_q = np.array(new_q_list, dtype=object)
        np_q_trans =np_q.T 
        return np_q_trans
    
    def simple_move(self, desired_position=np.array([0.9, 0, 1.0], dtype=object), 
                          desired_rotation=np.array([-math.pi/2, 0, -math.pi/2], dtype=object),
                          desired_time=10, freq=500):
        init_pose = np.array([np.deg2rad(-90), np.deg2rad(-132.46), np.deg2rad(122.85), np.deg2rad(99.65), np.deg2rad(45), np.deg2rad(-90.02)])
        q_list = []; q_list.append(init_pose)
        # Init pose 
        self.reset()
        self.sim.data.qpos[self.rev_joint_idxs[:-1]] = init_pose
        self.sim.forward()

        q=self.solve_ik(target_name = ['wrist_3_link', "tcp_link"], 
                        P_EE_des    = desired_position, 
                        R_EE_des    = desired_rotation)
        ctrl_q=q[1:]
        q_list.append(ctrl_q)
        interpoled_q =self.q_interpolation(q_list,desired_time,freq)
        return interpoled_q