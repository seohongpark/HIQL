class BaseObject:
    def __init__(self, name, obj_cfg, p, cid, data_path, global_scaling):
        self.p = p
        self.cid = cid
        self.name = name
        self.file = data_path / obj_cfg["file"]
        self.global_scaling = global_scaling

    def reset(self, state):
        pass

    def get_info(self):
        pass
