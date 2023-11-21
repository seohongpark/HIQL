from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

path = Path(sys.argv[-1])

for subdir in ["training", "validation"]:
    for file in tqdm((path / subdir).glob("*.npz")):
        data = np.load(file)
        if data["rel_actions"][-1] == 0:
            data = dict(data)
            data["rel_actions"][-1] = -1
            data["actions"][-1] = -1
            np.savez(file, **data)
