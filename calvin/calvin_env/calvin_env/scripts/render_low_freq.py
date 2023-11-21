import argparse
import itertools
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from calvin_env.utils import utils


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


parser = argparse.ArgumentParser(description="convert dataset to 15 hz (leave one step out)")
parser.add_argument("src", type=str)
parser.add_argument("dest", type=str)
parser.add_argument("--max_rel_pos", type=float, default=0.04)
parser.add_argument("--max_rel_orn", type=float, default=0.1)
args = parser.parse_args()
src_path = Path(args.src)
dest_path = Path(args.dest)

os.mkdir(dest_path)
os.mkdir(dest_path / "training")
os.mkdir(dest_path / "validation")

new_i = 0

for subdir in ["training", "validation"]:
    ep_lens = np.load(src_path / subdir / "ep_lens.npy")
    ep_start_end_ids = np.load(src_path / subdir / "ep_start_end_ids.npy")

    new_ep_lens = []
    new_ep_start_end_ids = []

    for start, end in tqdm(ep_start_end_ids):

        ep_len = end - start + 1
        for offset in (0, 1):
            new_start = new_i
            for old_i in range(start + offset, end + 1, 2):
                old_data = np.load(src_path / subdir / f"episode_{old_i:06d}.npz")
                data = dict(old_data)
                if old_i < end:
                    next_old_data = np.load(src_path / subdir / f"episode_{old_i + 1:06d}.npz")
                    next_data = dict(next_old_data)
                    data["actions"] = next_data["actions"]
                    data["rel_actions"] = utils.to_relative_action(
                        data["actions"], data["robot_obs"], max_pos=args.max_rel_pos, max_orn=args.max_rel_orn
                    )
                np.savez(dest_path / subdir / f"episode_{new_i:06d}.npz", **data)
                new_i += 1
            new_end = new_i - 1
            new_ep_len = new_end - new_start + 1
            new_ep_start_end_ids.append((new_start, new_end))
            new_ep_lens.append(new_ep_len)
    np.save(dest_path / subdir / "ep_lens.npy", new_ep_lens)
    np.save(dest_path / subdir / "ep_start_end_ids.npy", new_ep_start_end_ids)
    shutil.copy(src_path / subdir / "statistics.yaml", dest_path / subdir)
    os.makedirs(dest_path / subdir / ".hydra")
    shutil.copytree(src_path / subdir / ".hydra", dest_path / subdir / ".hydra", dirs_exist_ok=True)
    cfg = OmegaConf.load(dest_path / subdir / ".hydra/merged_config.yaml")
    cfg.robot.max_rel_pos = args.max_rel_pos
    cfg.robot.max_rel_orn = args.max_rel_orn
    cfg.env.control_freq = 15
    OmegaConf.save(cfg, dest_path / subdir / ".hydra/merged_config.yaml")
