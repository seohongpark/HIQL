import os

import numpy as np

from calvin_env.camera.camera import Camera
import tacto

REPO_BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class TactileSensor(Camera):
    def __init__(
        self, width, height, digit_link_ids, visualize_gui, cid, name, config_path, robot_id=None, objects=None
    ):
        """
        Initialize the camera
        Args:
            argument_group: initialize the camera and add needed arguments to argparse

        Returns:
            None
        """
        self.cid = cid
        self.name = name
        self.robot_uid = robot_id
        self.digits = tacto.Sensor(
            width=width, height=height, visualize_gui=visualize_gui, config_path=os.path.join(REPO_BASE, config_path)
        )
        self.digits.add_camera(robot_id, digit_link_ids)  # env.robot.digit_links()
        for obj in objects:
            # self.digits.add_body(obj)
            self.digits.add_object(obj.file.as_posix(), obj.uid, obj.global_scaling)
        self.visualize_gui = visualize_gui

    def render(self):
        rgb, depth = self.digits.render()
        if self.visualize_gui:
            self.digits.updateGUI(rgb, depth)
        rgb = np.concatenate(rgb, axis=2)
        depth = np.stack(depth, axis=2)
        return rgb, depth
