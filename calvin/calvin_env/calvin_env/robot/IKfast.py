from ikfast_franka_panda import get_ik
import numpy as np
from scipy.spatial.transform import Rotation as R


class IKfast:
    def __init__(
        self,
        robot_uid,
        cid,
        rp,
        ll_real,
        ul_real,
        base_position,
        base_orientation,
        weights=[1, 1, 1, 1, 1, 1, 1],
        num_angles=50,
    ):
        self.robot_uid = robot_uid
        self.cid = cid
        self.ll_real = ll_real
        self.ul_real = ul_real
        self.rp = rp
        self.num_dof = len(self.ll_real)
        self.weights = weights
        self.num_angles = num_angles
        T_world_robot = np.eye(4)
        T_world_robot[:3, 3] = base_position
        T_world_robot[:3, :3] = R.from_quat(base_orientation).as_matrix()
        self.T_robot_world = np.linalg.inv(T_world_robot)

    def world_to_robot(self, pos_w, orn_w):
        """
        pos, quat -> pos, Rot
        """
        pose_w = np.eye(4)
        pose_w[:3, 3] = pos_w
        pose_w[:3, :3] = R.from_quat(orn_w).as_matrix()
        pose_r = self.T_robot_world @ pose_w
        pos_r = list(pose_r[:3, 3])
        orn_r = pose_r[:3, :3].tolist()
        return pos_r, orn_r

    def filter_solutions(self, sol):
        test_sol = np.ones(self.num_dof) * 9999.0
        for i in range(self.num_dof):
            for add_ang in [-2.0 * np.pi, 0, 2.0 * np.pi]:
                test_ang = sol[i] + add_ang
                if self.ul_real[i] >= test_ang >= self.ll_real[i]:
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.0):
            return test_sol
        return None

    def take_closest_sol(self, sols, last_q, weights):
        best_sol_ind = np.argmin(np.sum((weights * (sols - np.array(last_q))) ** 2, 1))
        return sols[best_sol_ind]

    def get_ik_solution(self, target_pos, target_orn):
        target_pos_robot, target_orn_robot = self.world_to_robot(target_pos, target_orn)
        sols = []
        feasible_sols = []
        for q_6 in np.linspace(self.ll_real[-1], self.ul_real[-1], self.num_angles):
            sols += get_ik(target_pos_robot, target_orn_robot, [q_6])
        for sol in sols:
            sol = self.filter_solutions(sol)
            if sol is not None:
                feasible_sols.append(sol)
        if len(feasible_sols) < 1:
            return None
        best_sol = self.take_closest_sol(feasible_sols, self.rp[:7], self.weights)
        return best_sol
