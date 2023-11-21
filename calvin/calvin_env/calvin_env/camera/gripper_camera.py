import numpy as np
import pybullet as p

from calvin_env.camera.camera import Camera


class GripperCamera(Camera):
    def __init__(self, fov, aspect, nearval, farval, width, height, robot_id, cid, name, objects=None):
        self.cid = cid
        self.robot_uid = robot_id
        links = {
            p.getJointInfo(self.robot_uid, i, physicsClientId=self.cid)[12].decode("utf-8"): i
            for i in range(p.getNumJoints(self.robot_uid, physicsClientId=self.cid))
        }
        self.gripper_cam_link = links["gripper_cam"]
        self.fov = fov
        self.aspect = aspect
        self.nearval = nearval
        self.farval = farval
        self.width = width
        self.height = height

        self.name = name

    def render(self):
        camera_ls = p.getLinkState(
            bodyUniqueId=self.robot_uid, linkIndex=self.gripper_cam_link, physicsClientId=self.cid
        )
        camera_pos, camera_orn = camera_ls[:2]
        cam_rot = p.getMatrixFromQuaternion(camera_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
        # camera: eye position, target position, up vector
        self.view_matrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self.nearval, farVal=self.farval
        )
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            physicsClientId=self.cid,
        )
        rgb_img, depth_img = self.process_rgbd(image, self.nearval, self.farval)
        return rgb_img, depth_img
