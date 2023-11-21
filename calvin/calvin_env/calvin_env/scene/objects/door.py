MAX_FORCE = 4


class Door:
    def __init__(self, name, cfg, uid, p, cid):
        self.name = name
        self.p = p
        self.cid = cid
        # get joint_index by name (to prevent index errors when additional joints are added)
        joint_index = next(
            i
            for i in range(self.p.getNumJoints(uid, physicsClientId=self.cid))
            if self.p.getJointInfo(uid, i, physicsClientId=self.cid)[1].decode("utf-8") == name
        )
        self.joint_index = joint_index
        self.uid = uid
        self.initial_state = cfg["initial_state"]
        self.p.setJointMotorControl2(
            self.uid,
            self.joint_index,
            controlMode=p.VELOCITY_CONTROL,
            force=MAX_FORCE,
            physicsClientId=self.cid,
        )

    def reset(self, state=None):
        _state = self.initial_state if state is None else state
        self.p.resetJointState(
            self.uid,
            self.joint_index,
            _state,
            physicsClientId=self.cid,
        )

    def get_state(self):
        joint_state = self.p.getJointState(self.uid, self.joint_index, physicsClientId=self.cid)
        return float(joint_state[0])

    def get_info(self):
        return {"current_state": self.get_state()}
