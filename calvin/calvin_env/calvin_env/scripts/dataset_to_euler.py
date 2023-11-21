from pathlib import Path
from shutil import copyfile, copytree

import numpy as np
import pybullet as p
from tqdm import tqdm

load_path = Path("/home/hermannl/phd/data/banana_dataset_01_29/validation")

save_path = Path("/home/hermannl/phd/data/banana_dataset_01_29_euler/validation")
save_path.mkdir(parents=True, exist_ok=True)

for file in tqdm(load_path.glob("*.npz")):
    data = np.load(file)
    robot_obs = data["robot_obs"]
    robot_obs_euler = np.concatenate([robot_obs[:3], p.getEulerFromQuaternion(robot_obs[3:7]), robot_obs[7:]])
    scene_obs = data["scene_obs"]
    scene_obs_euler = scene_obs[:3]
    for i in range(6):
        scene_obs_euler = np.append(scene_obs_euler, scene_obs[3 + i * 7 : 3 + i * 7 + 3])
        scene_obs_euler = np.append(scene_obs_euler, p.getEulerFromQuaternion(scene_obs[3 + i * 7 + 3 : 3 + i * 7 + 7]))
    actions = data["actions"]
    actions_euler = np.concatenate([actions[:3], p.getEulerFromQuaternion(actions[3:7]), actions[7:]])
    data_euler = dict(data.items())
    data_euler["robot_obs"] = robot_obs_euler
    data_euler["scene_obs"] = scene_obs_euler
    data_euler["actions"] = actions_euler
    np.savez(save_path / file.name, **data_euler)

for file in set(load_path.glob("*")) - set(load_path.glob("*.npz")):
    if file.is_dir():
        copytree(file, save_path / file.name)
    else:
        copyfile(file, save_path / file.name)
