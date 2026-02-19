import os
from lib.simulation import mem_spice
import numpy as np
from lib.path_util import get_results_data_dir
import sys
import json

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

def add_uniform_noise_to_conductance_matrix(
    conductance_matrix: np.ndarray,
    g_min: float,
    g_max: float,
    noise_coeff: float,
) -> np.ndarray:
    """
    Uniform noise model after feedback tuning (write-verify)

    noise ~ U(-noise_coeff * g_max, +noise_coeff * g_max)
    """
    noise_amplitude = noise_coeff * g_max

    noise = np.random.uniform(
        low=-noise_amplitude,
        high=noise_amplitude,
        size=conductance_matrix.shape,
    )

    noisy_conductance_matrix = conductance_matrix + noise

    noisy_conductance_matrix = np.clip(
        noisy_conductance_matrix,
        a_min=g_min,
        a_max=g_max,
    )

    return noisy_conductance_matrix

# def add_noise_to_conductance_matrix(
#     conductance_matrix: np.ndarray,
#     _low_bound_conductance: float,
#     noise_coeff: float,
# ) -> np.ndarray:
#     noise = np.random.normal(
#         loc=0.0,
#         scale=noise_coeff * conductance_matrix,
#         size=conductance_matrix.shape,
#     )
#     noisy_conductance_matrix = conductance_matrix + noise
#     noisy_conductance_matrix = np.clip(noisy_conductance_matrix, a_min=_low_bound_conductance, a_max=None)
#     return noisy_conductance_matrix

def optimization(
    target_conductance_matrix: np.ndarray,
    _single_cell_conductance_range,
    _low_bound_conductance,
    _iteration_num,
    noise_coeff: float,
    file_path: str,
):
    m = target_conductance_matrix.shape[0]
    n = target_conductance_matrix.shape[1]

    effective_conductance_matrix_list = []
    

    # include the initial conductance matrix
    new_written_conductance_matrix = np.copy(target_conductance_matrix)
    new_written_conductance_matrix = add_uniform_noise_to_conductance_matrix(
        new_written_conductance_matrix,
        _low_bound_conductance,
        single_cell_conductance_range,
        noise_coeff,
    )
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
    effective_conductance_matrix = add_uniform_noise_to_conductance_matrix(
        effective_conductance_matrix,
        _low_bound_conductance,
        single_cell_conductance_range,
        noise_coeff,
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

    new_written_conductance_matrix = add_uniform_noise_to_conductance_matrix(
        new_written_conductance_matrix,
        _low_bound_conductance,
        single_cell_conductance_range,
        noise_coeff,
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

        effective_conductance_matrix = add_uniform_noise_to_conductance_matrix(
            effective_conductance_matrix,
            _low_bound_conductance,
            single_cell_conductance_range,
            noise_coeff,
        )
        effective_conductance_matrix_list.append(effective_conductance_matrix)

        # update written conductance matrix using pseudo-gradient descent
        differ_matrix = effective_conductance_matrix - target_conductance_matrix

        # decay eta
        eta_matrix = eta_matrix * beta
        # update written conductance matrix
        print(f"Iteration {__ + 1}/{_iteration_num}")
        new_written_conductance_matrix = mem_spice.clip_conductance_matrix(
            new_written_conductance_matrix - differ_matrix * eta_matrix,
            single_cell_conductance_range=_single_cell_conductance_range,
            low_bound_conductance=_low_bound_conductance,
        )
        new_written_conductance_matrix = add_uniform_noise_to_conductance_matrix(
            new_written_conductance_matrix,
            _low_bound_conductance,
            single_cell_conductance_range,
            noise_coeff,
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

    effective_conductance_matrix = add_uniform_noise_to_conductance_matrix(
        effective_conductance_matrix,
        _low_bound_conductance,
        single_cell_conductance_range,
        noise_coeff,
    )
    effective_conductance_matrix_list.append(effective_conductance_matrix)

    np.savez(
        file_path,
        target_conductance_matrix=target_conductance_matrix,
        effective_conductance_matrix_list=effective_conductance_matrix_list,
        new_written_conductance_matrix_list=new_written_conductance_matrix_list,
        noise_coeff=noise_coeff,
        # random_input_voltage_vectors=random_input_voltage_vectors,
        # ideal_output_current_vectors=ideal_output_current_vectors,
        # spice_output_current_vectors_list=spice_output_current_vectors_list,
        # eta_matrix=eta_matrix,
    )


current_file_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_file_dir, "config.json"), "r") as f:
    config = json.load(f)

m = config.get("scale")
n = m
print(f"Matrix size: {m}x{n}")

wire_resistance = 0.35

iteration_num = 20

beta = 0.99

# conductance_lsb = 99e-6 / 255

# low_bound_conductance = 10e-6
# single_cell_conductance_range = 100e-6
# target_conductance_max = 15e-06

# target_conductance_matrix = (
#     np.random.random((m, n)) * (target_conductance_max - low_bound_conductance) # type: ignore[operator]
#     + low_bound_conductance
# )

data = np.load(
    os.path.join(
        current_file_dir,
        "data",
        f"{m}",
        "generated_target_conductance_matrix.npz",
    )
)

target_conductance_matrix = data["target_conductance_matrix"]
low_bound_conductance = data["low_bound_conductance"].item()
single_cell_conductance_range = data["single_cell_conductance_range"].item()
target_conductance_max = data["target_conductance_max"].item()

noise_coeff = 0.05

print(sys.argv)
args = sys.argv

# matrix_type = args[3]

# data = mapping.load_memristor_mapping()  # target

# conductance_matrix = data[i][j][matrix_type]
results_file_path = os.path.join(
    get_results_data_dir(),
    "optim_with_noise",
    "uniform_noise",
    f"{m}",
    f"{noise_coeff}",
    "aco_optim.npz",
)

os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
print(os.path.dirname(results_file_path))
# exit(0)
optimization(
    target_conductance_matrix=target_conductance_matrix,
    _single_cell_conductance_range=single_cell_conductance_range,
    _low_bound_conductance=low_bound_conductance,
    _iteration_num=iteration_num,
    noise_coeff=noise_coeff,
    file_path=results_file_path,
)
