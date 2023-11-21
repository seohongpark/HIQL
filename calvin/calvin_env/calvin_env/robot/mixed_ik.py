import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from calvin_env.utils.utils import angle_between_quaternions


class MixedIK:
    def __init__(
        self,
        robot_uid,
        cid,
        ll_real,
        ul_real,
        base_position,
        base_orientation,
        tcp_link_id,
        ll,
        ul,
        jr,
        rp,
        use_ik_fast,
        use_nullspace=True,
        threshold_pos=0.01,
        threshold_orn=0.05,
        weights=(1, 1, 1, 1, 1, 1, 1),
        num_angles=50,
    ):
        self.robot_uid = robot_uid
        self.cid = cid
        self.use_nullspace = use_nullspace
        self.use_ik_fast = use_ik_fast
        self.ik_fast = None
        if self.use_ik_fast:
            from ikfast_franka_panda import get_fk

            from calvin_env.robot.IKfast import IKfast

            self.get_fk = get_fk
            self.ik_fast = IKfast(
                robot_uid, cid, rp, ll_real, ul_real, base_position, base_orientation, weights, num_angles
            )
        self.tcp_link_id = tcp_link_id
        self.ll = ll
        self.ul = ul
        self.jr = jr
        self.ll_real = ll_real
        self.ul_real = ul_real
        self.rp = rp
        self.num_dof = len(self.ll_real)
        self.threshold_pos = threshold_pos
        self.threshold_orn = threshold_orn
        self.is_using_IK_fast = False

    def get_bullet_ik(self, desired_ee_pos, desired_ee_orn):
        if self.use_nullspace:
            jnt_ps = p.calculateInverseKinematics(
                self.robot_uid,
                self.tcp_link_id,
                desired_ee_pos,
                desired_ee_orn,
                self.ll,
                self.ul,
                self.jr,
                self.rp,
                physicsClientId=self.cid,
            )
        else:
            jnt_ps = p.calculateInverseKinematics(
                self.robot_uid, self.tcp_link_id, desired_ee_pos, desired_ee_orn, physicsClientId=self.cid
            )
        # clip joint positions outside the joint ranges
        jnt_ps = np.clip(jnt_ps[: self.num_dof], self.ll_real, self.ul_real)
        return jnt_ps

    def robot_to_world(self, pos_r, orn_r):
        """
        pos, Rot -> pos, quat
        """
        pose_r = np.eye(4)
        pose_r[:3, 3] = pos_r
        pose_r[:3, :3] = orn_r
        pose_w = np.linalg.inv(self.ik_fast.T_robot_world) @ pose_r
        pos_r = pose_w[:3, 3]
        orn_r = R.from_matrix(pose_w[:3, :3]).as_quat()
        return pos_r, orn_r

    def pose_within_threshold(self, target_pos, target_orn, q):
        pos, orn = self.get_fk(q)
        pos, orn = self.robot_to_world(pos, orn)
        angular_diff = angle_between_quaternions(orn, target_orn)
        threshold_pos_exceeded = np.linalg.norm(target_pos - pos) > self.threshold_pos
        threshold_orn_exceeded = angular_diff > self.threshold_orn
        return not (threshold_pos_exceeded or threshold_orn_exceeded)

    def get_joint_states(self):
        return list(zip(*p.getJointStates(self.robot_uid, range(self.num_dof))))[0]

    def get_ik(self, target_pos, target_orn):
        if self.is_using_IK_fast and not self.pose_within_threshold(target_pos, target_orn, self.get_joint_states()):
            q_ik_fast = self.ik_fast.get_ik_solution(target_pos, target_orn)
            if q_ik_fast is not None:
                self.is_using_IK_fast = True
                return q_ik_fast
            else:
                self.is_using_IK_fast = False
                q_bullet = self.get_bullet_ik(target_pos, target_orn)
                return q_bullet
        self.is_using_IK_fast = False
        q_bullet = self.get_bullet_ik(target_pos, target_orn)
        if self.use_ik_fast and not self.pose_within_threshold(target_pos, target_orn, q_bullet):
            q_ik_fast = self.ik_fast.get_ik_solution(target_pos, target_orn)
            if q_ik_fast is not None:
                self.is_using_IK_fast = True
                return q_ik_fast
            else:
                return q_bullet
        return q_bullet
