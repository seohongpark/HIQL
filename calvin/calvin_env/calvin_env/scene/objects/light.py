from enum import Enum


class LightState(Enum):
    ON = 1
    OFF = 0


class Light:
    def __init__(self, name, cfg, uid, p, cid):
        self.name = name
        self.uid = uid
        self.p = p
        self.cid = cid
        self.link = cfg["link"]
        self.link_id = next(
            i
            for i in range(self.p.getNumJoints(uid, physicsClientId=self.cid))
            if self.p.getJointInfo(uid, i, physicsClientId=self.cid)[12].decode("utf-8") == self.link
        )
        self.color_on = cfg["color"]
        self.color_off = [1, 1, 1, 1]
        self.state = LightState.OFF

    def reset(self, state=None):
        if state is None:
            self.turn_off()
        else:
            if state == LightState.ON.value:
                self.turn_on()
            elif state == LightState.OFF.value:
                self.turn_off()
            else:
                print("Light state can be only 0 or 1.")
                raise ValueError

    def get_state(self):
        return self.state.value

    def get_info(self):
        return {"logical_state": self.get_state()}

    def turn_on(self):
        self.state = LightState.ON
        self.p.changeVisualShape(self.uid, self.link_id, rgbaColor=self.color_on, physicsClientId=self.cid)

    def turn_off(self):
        self.state = LightState.OFF
        self.p.changeVisualShape(self.uid, self.link_id, rgbaColor=self.color_off, physicsClientId=self.cid)

    def serialize(self):
        return self.get_info()
