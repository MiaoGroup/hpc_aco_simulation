"""
CxDNN: Hardware-soware Compensation Methods for Deep Neural Networks on Resistive Crossbar Systems
"""

import numpy as np
import os
from lib.simulation import mem_spice
from lib.path_util import get_results_data_dir
from lib.constraint import target_conductance_max_dict

current_file_dir = os.path.dirname(os.path.abspath(__file__))


m = 256
n = m
print(f"Matrix size: {m}x{n}")


if "LSB_JOBID" not in os.environ:
    os.environ["LSB_JOBID"] = "local_test_jobid"
jobid = os.environ["LSB_JOBID"]
print(f"Job ID: {jobid}")  # Print the job ID to verify it's being captured correctly


tmp_file_dir = os.path.join(os.getcwd(), ".tmp", jobid)
os.makedirs(tmp_file_dir, exist_ok=True)


low_bound_conductance = 1e-6
single_cell_conductance_range = 100e-6
target_conductance_max = target_conductance_max_dict[m]

target_conductance_matrix = (
    np.random.random((m, n)) * (target_conductance_max - low_bound_conductance) # type: ignore[operator]
    + low_bound_conductance
)


wire_resistance_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

for wire_resistance in wire_resistance_list:
    m = target_conductance_matrix.shape[0]
    n = target_conductance_matrix.shape[1]


    effective_conductance_matrix = mem_spice.get_effective_conductance_matrix(
        target_conductance_matrix,
        wire_resistance=wire_resistance,
        tmp_file_dir=tmp_file_dir,
        spice_backend="xyce",
    )
    # effective_conductance_matrix = np.random.random(target_conductance_matrix.shape) * 0.1 + target_conductance_matrix * 0.9

    v_input = np.random.random((1, effective_conductance_matrix.shape[0])) * 0.2

    refer_outputs = mem_spice.parallel_sim_vmm(
        v_input,
        np.round(1 / target_conductance_matrix, 0),
        wire_resistance=wire_resistance,
        spice_backend="xyce",
    )

    # ideal_output = v_input @ target_conductance_matrix
    model_outputs = v_input @ effective_conductance_matrix

    results_data_dir = get_results_data_dir()
    output_file_path = os.path.join(
        results_data_dir,
        "reference_model",
        f"{m}",
        f"{wire_resistance}.npz",
    )

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    np.savez(
        output_file_path,
        effective_conductance_matrix=effective_conductance_matrix,
        target_conductance_matrix=target_conductance_matrix,
        v_input=v_input,
        refer_outputs=refer_outputs,
        model_outputs=model_outputs,
    )
    print("Saved effective conductance matrix to:", output_file_path)
# print("Effective conductance matrix shape:", effective_conductance_matrix.shape)
# print("Sample values from effective conductance matrix:", effective_conductance_matrix[:5, :5])
# print("Saved programmed conductance matrix to:", output_file_path)
