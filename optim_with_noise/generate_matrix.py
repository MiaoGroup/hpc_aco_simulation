import numpy as np
import os
import json
from lib.constraint import target_conductance_max_dict
current_file_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_file_dir, "config.json"), "r") as f:
    config = json.load(f)

m = config.get("scale")
n = m


# conductance_lsb = 99e-6 / 255

low_bound_conductance = 1e-6
single_cell_conductance_range = 100e-6
target_conductance_max = target_conductance_max_dict[m]

target_conductance_matrix = (
    np.random.random((m, n)) * (target_conductance_max - low_bound_conductance) # type: ignore[operator]
    + low_bound_conductance
)


print("Generated target conductance matrix with shape:", target_conductance_matrix.shape)
print("Sample values from target conductance matrix:", target_conductance_matrix[:5, :5])

data_root = os.path.join(current_file_dir, "data", f"{m}")
# 确保 data 目录存在
os.makedirs(data_root, exist_ok=True)

file_path = os.path.join(
    current_file_dir,
    "data",
    f"{m}",
    "generated_target_conductance_matrix.npz",
)

np.savez(
    file_path,
    target_conductance_matrix=target_conductance_matrix,
    low_bound_conductance=low_bound_conductance,
    single_cell_conductance_range=single_cell_conductance_range,
    target_conductance_max=target_conductance_max,
)