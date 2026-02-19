import datetime
import os
import time
from lib.simulation import mem_spice
import numpy as np
import logging
from lib.path_util import get_results_data_dir
import sys
from lib.constraint import target_conductance_max_dict

# Define the size of the matrix
# _logger = logging.getLogger(__name__)
# _logger.setLevel(logging.INFO)
# logging.basicConfig(filename="./logs/convergence.log", level=logging.INFO)

if "LSB_JOBID" not in os.environ:
    os.environ["LSB_JOBID"] = "local_test_jobid"
jobid = os.environ["LSB_JOBID"]
print(f"Job ID: {jobid}")  # Print the job ID to verify it's being captured correctly

tmp_file_dir = os.path.join(os.getcwd(), ".tmp", jobid)
os.makedirs(tmp_file_dir, exist_ok=True)


def optimization(
    target_conductance_matrix: np.ndarray,
    _single_cell_conductance_range,
    _low_bound_conductance,
    _iteration_num,
    _beta,
    file_path: str,
):
    m = target_conductance_matrix.shape[0]
    n = target_conductance_matrix.shape[1]

    effective_conductance_matrix_list = []

    # include the initial conductance matrix
    new_written_conductance_matrix = np.copy(target_conductance_matrix)
    new_written_conductance_matrix_list = [new_written_conductance_matrix]
    # spice_output_current_vectors_list = []

    # voltage range is [0, 1) V, no different from [0, 0.2) V in simulation, just scale it

    # Eta matrix, H
    eta_matrix: np.ndarray = np.load(
        os.path.join(
            os.getcwd(),
            "lib",
            "simulation",
            "eta",
            f"eta_matrix_{m}x{n}_G_15_wire_{wire_resistance}.npy",
        )
    )

    effective_conductance_matrix = mem_spice.get_effective_conductance_matrix(
        new_written_conductance_matrix,
        wire_resistance=wire_resistance,
        tmp_file_dir=tmp_file_dir,
        spice_backend="xyce",
    )
    effective_conductance_matrix_list.append(effective_conductance_matrix)

    # start
    print(f"Iteration 1/{_iteration_num}")
    # 第一步写入
    new_written_conductance_matrix = mem_spice.clip_conductance_matrix(
        new_written_conductance_matrix * eta_matrix,
        single_cell_conductance_range=_single_cell_conductance_range,
        low_bound_conductance=_low_bound_conductance,
    )

    new_written_conductance_matrix_list.append(new_written_conductance_matrix)

    for __ in range(1, _iteration_num):
        # get_spice_vmm_output_results

        # get effective conductance matrix
        effective_conductance_matrix = mem_spice.get_effective_conductance_matrix(
            new_written_conductance_matrix,
            wire_resistance=wire_resistance,
            tmp_file_dir=tmp_file_dir,
            spice_backend="xyce",
        )
        effective_conductance_matrix_list.append(effective_conductance_matrix)

        # update written conductance matrix using pseudo-gradient descent
        differ_matrix = effective_conductance_matrix - target_conductance_matrix

        # decay eta
        eta_matrix = eta_matrix * _beta
        # update written conductance matrix
        print(f"Iteration {__ + 1}/{_iteration_num}")
        new_written_conductance_matrix = mem_spice.clip_conductance_matrix(
            new_written_conductance_matrix - differ_matrix * eta_matrix,
            single_cell_conductance_range=_single_cell_conductance_range,
            low_bound_conductance=_low_bound_conductance,
        )
        # save the new written conductance matrix
        new_written_conductance_matrix_list.append(new_written_conductance_matrix)

        # program, no need in simulation
        # program_memristor(new_written_conductance_matrix)

    effective_conductance_matrix = mem_spice.get_effective_conductance_matrix(
        new_written_conductance_matrix,
        wire_resistance=wire_resistance,
        tmp_file_dir=tmp_file_dir,
        spice_backend="xyce",
    )
    effective_conductance_matrix_list.append(effective_conductance_matrix)

    np.savez(
        file_path,
        target_conductance_matrix=target_conductance_matrix,
        effective_conductance_matrix_list=effective_conductance_matrix_list,
        new_written_conductance_matrix_list=new_written_conductance_matrix_list,
        # random_input_voltage_vectors=random_input_voltage_vectors,
        # ideal_output_current_vectors=ideal_output_current_vectors,
        # spice_output_current_vectors_list=spice_output_current_vectors_list,
        # eta_matrix=eta_matrix,
    )


print(sys.argv)
args = sys.argv
data_index = 1
if len(args) == 2:
    print("Usage: python aco_optim.py <data_index>")
    data_index = int(args[1])

m = 512
n = m

wire_resistance = 0.35

iteration_num = 20


# beta_list = [1.5, 1.2, 1, 0.99, 0.95, 0.9, 0.8, 0.7]
# beta_list = [1.2, 1.5, 2.0, 3.0]


current_file_dir = os.path.dirname(os.path.abspath(__file__))



# iteration_num = 20
# beta = 0.99

# conductance_lsb = 99e-6 / 255

low_bound_conductance = 1e-6
single_cell_conductance_range = 150e-6
target_conductance_max = target_conductance_max_dict[m]

target_conductance_matrix = (
    np.random.random((m, n)) * (target_conductance_max - low_bound_conductance) # type: ignore[operator]
    + low_bound_conductance
)


# matrix_type = args[3]

# data = mapping.load_memristor_mapping()  # target
# for beta in beta_list:
# conductance_matrix = data[i][j][matrix_type]
date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Current date and time: {date}")
# exit(0)
results_file_path = os.path.join(
    get_results_data_dir(),
    "optim_basic",
    f"{m}",
    f"{date}",
    "aco_optim.npz",
)

os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

optimization(
    target_conductance_matrix=target_conductance_matrix,
    _single_cell_conductance_range=single_cell_conductance_range,
    _low_bound_conductance=low_bound_conductance,
    _iteration_num=iteration_num,
    _beta=1,
    file_path=results_file_path,
)
