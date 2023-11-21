from pathlib import Path
from shutil import copyfile, copytree

import numpy as np
from tqdm import tqdm

load_path = Path("/home/meeso/expert_demos_03_10/training")

save_path = Path("/home/meeso/expert_demos_03_10/training_unnormalized_depth")
save_path.mkdir(parents=True, exist_ok=True)

for file in tqdm(load_path.glob("*.npz")):
    data = np.load(file)
    corrected_data = dict(data.items())
    corrected_data["depth_static"] = data["depth_static"] * 2.0
    corrected_data["depth_gripper"] = data["depth_gripper"] * 2.0
    np.savez(save_path / file.name, **corrected_data)


for file in set(load_path.glob("*")) - set(load_path.glob("*.npz")):
    if file.is_dir():
        copytree(file, save_path / file.name)
    else:
        copyfile(file, save_path / file.name)
