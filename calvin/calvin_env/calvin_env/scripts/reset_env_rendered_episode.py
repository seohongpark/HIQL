from copy import deepcopy
import glob
import os
from pathlib import Path
import time

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from calvin_env.envs.tasks import Tasks
from calvin_env.utils import utils

"""
This script loads a rendered episode and replays it using the recorded actions.
Optionally, gaussian noise can be added to the actions.
"""


def noise(action, pos_std=0.01, rot_std=1):
    """
    adds gaussian noise to position and orientation.
    units are m for pos and degree for rot
    """
    pos, orn, gripper = action
    rot_std = np.radians(rot_std)
    pos_noise = np.random.normal(0, pos_std, 3)
    rot_noise = p.getQuaternionFromEuler(np.random.normal(0, rot_std, 3))
    pos, orn = p.multiplyTransforms(pos, orn, pos_noise, rot_noise)
    return pos, orn, gripper


@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    root_dir = Path("/tmp/test_render/2021-10-05/16-51-11")

    ep_start_end_ids = np.sort(np.load(root_dir / "ep_start_end_ids.npy"), axis=0)
    rel_actions = []
    tasks = hydra.utils.instantiate(cfg.tasks)
    prev_info = None
    t1 = time.time()
    for s, e in ep_start_end_ids:
        print("new_episode")
        for i in range(s, e + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            data = np.load(file)
            img = data["rgb_static"]
            cv2.imshow("win2", cv2.resize(img[:, :, ::-1], (500, 500)))
            cv2.waitKey(1)

            if (i - s) % 32 == 0:
                print(f"reset {i}")
                env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
            action = data["rel_actions"]
            # action = np.split(data["actions"], [3, 6])
            # action = noise(action)

            # rel_actions.append(create_relative_action(data["actions"], data["robot_obs"][:6]))
            # action = utils.to_relative_action(data["actions"], data["robot_obs"], max_pos=0.04, max_orn=0.1)
            # tcp_pos, tcp_orn = p.getLinkState(env.robot.robot_uid, env.robot.tcp_link_id, physicsClientId=env.cid)[:2]
            # tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            # action2 = create_relative_action(data["actions"], np.concatenate([tcp_pos, tcp_orn]))
            o, _, _, info = env.step(action)
            print(info["scene_info"]["lights"]["led"]["logical_state"])
            if (i - s) % 32 != 0:
                print(tasks.get_task_info(prev_info, info))
            else:
                prev_info = deepcopy(info)
            time.sleep(0.01)
    print(time.time() - t1)
    # rel_actions = np.array(rel_actions)
    # for j in range(rel_actions.shape[1]):
    #     plt.figure(j)
    #     plt.hist(rel_actions[:, j], bins=10)
    #     plt.plot()
    #     plt.show()


if __name__ == "__main__":
    run_env()
