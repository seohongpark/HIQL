from calvin_env.scene.objects.base_object import BaseObject


class FixedObject(BaseObject):
    def __init__(self, name, obj_cfg, p, cid, data_path, global_scaling):
        super().__init__(name, obj_cfg, p, cid, data_path, global_scaling)
        self.initial_pos = obj_cfg["initial_pos"]
        self.initial_orn = self.p.getQuaternionFromEuler(obj_cfg["initial_orn"])

        self.uid = self.p.loadURDF(
            self.file.as_posix(),
            self.initial_pos,
            self.initial_orn,
            globalScaling=global_scaling,
            physicsClientId=self.cid,
        )
        self.info_dict = {"uid": self.uid}
        self.num_joints = self.p.getNumJoints(self.uid, physicsClientId=self.cid)
        if self.num_joints > 0:
            # save link names and ids in dictionary
            links = {
                self.p.getJointInfo(self.uid, i, physicsClientId=self.cid)[12].decode("utf-8"): i
                for i in range(self.num_joints)
            }
            links["base_link"] = -1
            self.info_dict["links"] = links

    def reset(self, state=None):
        pass

    def get_info(self):
        obj_info = {**self.info_dict, "contacts": self.p.getContactPoints(bodyA=self.uid, physicsClientId=self.cid)}
        return obj_info

    def serialize(self):
        joints = (
            self.p.getJointStates(self.uid, list(range(self.num_joints)), physicsClientId=self.cid)
            if self.num_joints > 0
            else ()
        )
        return {"uid": self.uid, "info": self.p.getBodyInfo(self.uid, physicsClientId=self.cid), "joints": joints}
