from pathlib import Path

import cv2
import hydra
import numpy as np
import pybullet as p

"""
This script loads a rendered episode and replays it using the recorded actions.
Optionally, gaussian noise can be added to the actions.
"""


def scale_depth(img):
    img = np.clip(img, 0, 1.5)
    img -= img.min()
    img /= img.max()
    return img


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
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)

    root_dir = Path(__file__).parents[3] / "dataset/task_D_D/training"

    ep_start_end_ids = np.sort(np.load(root_dir / "ep_start_end_ids.npy"), axis=0)

    seq_len = 32
    tasks = hydra.utils.instantiate(cfg.tasks)
    for s, e in ep_start_end_ids:
        i = s
        while 1:

            file = root_dir / f"episode_{i:07}.npz"
            data = np.load(file)
            # gripper_img = data["rgb_gripper"]
            # cv2.imshow("gripper", gripper_img[:, :, ::-1])
            # static_img = data["rgb_static"]
            # cv2.imshow("static", static_img[:, :, ::-1])
            # print(data["robot_obs"])
            # env.render()
            env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
            start_info = env.get_info()
            cv2.imshow("keylistener", np.zeros((300, 300)))
            k = cv2.waitKey(0) % 256
            if k == ord("a"):
                i -= 1
                i = np.clip(i, s, e - seq_len)
            if k == ord("d"):
                i += 1
                i = np.clip(i, s, e - seq_len)
            if k == ord("q"):
                i -= 100
                i = np.clip(i, s, e - seq_len)
            if k == ord("z"):
                i -= seq_len
                i = np.clip(i, s, e - seq_len)
            if k == ord("e"):
                i += 100
                i = np.clip(i, s, e - seq_len)
            if k == ord("r"):
                for j in range(i + 1, i + seq_len):
                    file = root_dir / f"episode_{j:07d}.npz"
                    data = np.load(file)
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    gripper_img = data["rgb_gripper"]
                    cv2.imshow("gripper", gripper_img[:, :, ::-1])
                    static_img = data["rgb_static"]
                    cv2.imshow("static", static_img[:, :, ::-1])
                    cv2.waitKey(100)
                end_info = env.get_info()
                task_info = tasks.get_task_info(start_info, end_info)
                if len(task_info):
                    print()
                    print()
                    print()
                    print(task_info)
                else:
                    print()
                    print()
                    print()
                    print("No task detected.")
                i = j
            if k == ord("n"):  # ESC
                break


if __name__ == "__main__":
    run_env()
