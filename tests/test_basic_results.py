import numpy as np
import os
from pathlib import Path


base_path = Path("~/hpc_xyce_results/optim_basic/32/").expanduser()


subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

if not subdirs:
    raise FileNotFoundError("Not found any subdirectory in {}".format(base_path))


latest_dir = subdirs[-1]
file_path = latest_dir / "aco_optim.npz"

print(f"Data path: {file_path}")


data = np.load(file_path)


print("Files in data:", data.files)
diff_first = np.linalg.norm(data["target_conductance_matrix"] - data["effective_conductance_matrix_list"][0])
diff_last = np.linalg.norm(data["target_conductance_matrix"] - data["effective_conductance_matrix_list"][-1])

print(f"Diff (First): {diff_first}")
print(f"Diff (Last): {diff_last}")