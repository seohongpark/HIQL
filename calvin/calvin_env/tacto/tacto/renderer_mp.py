import logging
from threading import Thread

import numpy as np
from omegaconf import OmegaConf
import multiprocessing
from multiprocessing import Process, Queue
from .renderer import Renderer as _Renderer
logger = logging.getLogger(__name__)


def worker(parent_to_child_q, child_to_parent_q, width, height, background, config_path):
    renderer = _Renderer(width, height, background, config_path)
    while True:
        try:
            cmd, data = parent_to_child_q.get()
            if cmd == "add_object":
                renderer.add_object(*data)
            elif cmd == "render":
                child_to_parent_q.put(renderer.render(*data))
            elif cmd == "update_camera_pose":
                renderer.update_camera_pose(*data)
            elif cmd == "depth0":
                child_to_parent_q.put(renderer.depth0)
            elif cmd == "close":
                break
        except EOFError:
            break


class Renderer:
    def __init__(self, width, height, background, config_path):
        self.closed = False
        self._width = width
        self._height = height
        self.conf = OmegaConf.load(config_path)
        self.parent_to_child_q = Queue()
        self.child_to_parent_q = Queue()
        # It is not clear to me why this works with a thread, but not with a process
        self._renderer = Thread(target=worker, args=(self.parent_to_child_q, self.child_to_parent_q, width, height, background, config_path), daemon=True)
        self._renderer.start()
        self.parent_to_child_q.put(("depth0", None))
        self.depth0 = self.child_to_parent_q.get()
        print("received depth0")

    def __del__(self):
        self.close()

    def add_object(self, objTrimesh, obj_name, position=[0, 0, 0], orientation=[0, 0, 0]):
        self.parent_to_child_q.put(("add_object", (objTrimesh, obj_name, position, orientation)))

    def render(self, object_poses=None, normal_forces=None, noise=True, calibration=True):
        self.parent_to_child_q.put(("render", (object_poses, normal_forces, noise, calibration)))
        return self.child_to_parent_q.get()

    def update_camera_pose(self, position, orientation):
        self.parent_to_child_q.put(("update_camera_pose", (position, orientation)))

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def _add_noise(self, color):
        """
        Add Gaussian noise to the RGB image
        :param color:
        :return:
        """
        # Add noise to the RGB image
        mean = self.conf.sensor.noise.color.mean
        std = self.conf.sensor.noise.color.std

        if mean != 0 or std != 0:
            noise = np.random.normal(mean, std, color.shape)  # Gaussian noise
            color = np.clip(color + noise, 0, 255).astype(
                np.uint8
            )  # Add noise and clip

        return color

    def close(self):
        if self.closed:
            return
        self.parent_to_child_q.put(('close', None))
        self._renderer.join()
        self.closed = True
