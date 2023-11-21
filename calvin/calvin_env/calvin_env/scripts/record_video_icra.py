from copy import deepcopy
from pathlib import Path
import time

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

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


def depth2rgb(img, minval=0, maxval=1):
    img -= minval
    img /= maxval - minval
    img *= 255
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)
    img = np.tile(np.expand_dims(img, axis=2), 3)
    return img


@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    root_dir = Path("/home/hermannl/data/calvin_abcd_example")

    ep_start_end_ids = [[100000, 104999]]
    rel_actions = []
    tasks = hydra.utils.instantiate(cfg.tasks)
    prev_info = None
    t1 = time.time()

    video = cv2.VideoWriter(
        "/home/hermannl/Documents/calvin/env_A_tactile.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (240, 160)
    )
    video_static = cv2.VideoWriter(
        "/home/hermannl/Documents/calvin/env_A_static_depth.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (200, 200)
    )
    video_gripper = cv2.VideoWriter(
        "/home/hermannl/Documents/calvin/env_A_gripper_depth.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (100, 100)
    )
    for s, e in ep_start_end_ids:
        print("new_episode")
        for i in range(s, e + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            data = np.load(file)
            # img = data["rgb_static"]
            # cv2.imshow("win2", cv2.resize(img[:, :, ::-1], (500, 500)))
            # cv2.waitKey(0)

            # if (i - s) % 32 == 0:
            #     print(f"reset {i}")
            obs = env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])

            im = obs["rgb_obs"][0]
            im = np.concatenate([im[:, :, :3], im[:, :, 3:6]], axis=1)
            cv2.imshow("im", im)
            cv2.waitKey(1)
            video.write(im[:, :, ::-1])
            video_gripper.write(depth2rgb(obs["depth_obs"][1], minval=0.1, maxval=0.5)[:, :, ::-1])
            video_static.write(depth2rgb(obs["depth_obs"][0], minval=3.5, maxval=5)[:, :, ::-1])
            # action = data["rel_actions"]
            action = np.split(data["actions"], [3, 6])
            action = noise(action)

            rel_actions.append(utils.to_relative_action(data["actions"], data["robot_obs"][:6]))
            # action = utils.to_relative_action(data["actions"], data["robot_obs"], max_pos=0.04, max_orn=0.1)
            # tcp_pos, tcp_orn = p.getLinkState(env.robot.robot_uid, env.robot.tcp_link_id, physicsClientId=env.cid)[:2]
            # tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            # action2 = utils.to_relative_action(data["actions"], np.concatenate([tcp_pos, tcp_orn]))
            o, _, _, info = env.step(action)
            print(info["scene_info"]["lights"]["led"]["logical_state"])
            if (i - s) % 32 != 0:
                print(tasks.get_task_info(prev_info, info))
            else:
                prev_info = deepcopy(info)
            time.sleep(0.01)
    video.release()
    video_static.release()
    video_gripper.release()
    print(time.time() - t1)
    rel_actions = np.array(rel_actions)
    for j in range(rel_actions.shape[1]):
        plt.figure(j)
        plt.hist(rel_actions[:, j], bins=10)
        plt.plot()
        plt.show()


if __name__ == "__main__":
    run_env()
