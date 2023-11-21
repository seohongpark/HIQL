import logging

import numpy as np
from omegaconf.errors import ConfigKeyError

from calvin_env.scene.objects.base_object import BaseObject

log = logging.getLogger(__name__)


class MovableObject(BaseObject):
    def __init__(self, name, obj_cfg, p, cid, data_path, global_scaling, euler_obs, surfaces, np_random):
        super().__init__(name, obj_cfg, p, cid, data_path, global_scaling)
        self.initial_pos = obj_cfg["initial_pos"]
        self.initial_orn = obj_cfg["initial_orn"]
        if isinstance(self.initial_pos, list):
            self.initial_pos = np.array(self.initial_pos)
        if isinstance(self.initial_orn, list):
            self.initial_orn = np.array(p.getQuaternionFromEuler(self.initial_orn))
        self.euler_obs = euler_obs
        self.surfaces = surfaces
        self.np_random = np_random

        initial_pos, initial_orn = self.sample_initial_pose()
        self.uid = self.p.loadURDF(
            self.file.as_posix(),
            initial_pos,
            initial_orn,
            globalScaling=global_scaling,
            physicsClientId=self.cid,
        )

    def reset(self, state=None):
        if state is None:
            initial_pos, initial_orn = self.sample_initial_pose()
        else:
            initial_pos, initial_orn = np.split(state, [3])
            if len(initial_orn) == 3:
                initial_orn = self.p.getQuaternionFromEuler(initial_orn)
        self.p.resetBasePositionAndOrientation(
            self.uid,
            initial_pos,
            initial_orn,
            physicsClientId=self.cid,
        )

    def sample_initial_pose(self):
        initial_pos = self.initial_pos
        if isinstance(self.initial_pos, str):
            if self.initial_pos == "any":
                surface = self.np_random.choice(list(self.surfaces.keys()))
                sampling_range = np.array(self.surfaces[surface])
            else:
                try:
                    sampling_range = np.array(self.surfaces[self.initial_pos])
                except ConfigKeyError:
                    log.error(f"surface {self.initial_pos} not specified in scene config")
                    raise KeyError
            initial_pos = self.np_random.uniform(sampling_range[0], sampling_range[1])
        initial_orn = self.initial_orn
        if isinstance(self.initial_orn, str):
            if self.initial_orn == "any":
                initial_orn = np.array(
                    self.p.getQuaternionFromEuler(self.np_random.uniform([0, 0, -np.pi], [0, 0, np.pi]))
                )
            else:
                log.error("Only keyword 'any' supported at the moment")
                raise ValueError
        return initial_pos, initial_orn

    def get_state(self):
        pos, orn = self.p.getBasePositionAndOrientation(self.uid, physicsClientId=self.cid)
        if self.euler_obs:
            orn = self.p.getEulerFromQuaternion(orn)
        return np.concatenate([pos, orn])

    def get_info(self):
        pos, orn = self.p.getBasePositionAndOrientation(self.uid, physicsClientId=self.cid)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.uid, physicsClientId=self.cid)
        obj_info = {
            "current_pos": pos,
            "current_orn": orn,
            "current_lin_vel": lin_vel,
            "current_ang_vel": ang_vel,
            "contacts": self.p.getContactPoints(bodyA=self.uid, physicsClientId=self.cid),
            "uid": self.uid,
        }
        return obj_info

    def serialize(self):
        return {
            "uid": self.uid,
            "info": self.p.getBodyInfo(self.uid, physicsClientId=self.cid),
            "pose": self.p.getBasePositionAndOrientation(self.uid, physicsClientId=self.cid),
        }
