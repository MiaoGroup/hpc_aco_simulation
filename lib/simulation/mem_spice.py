import inspect
import os
from typing import Any, Literal
from venv import logger
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict
import subprocess
from multiprocessing import Process
from lib.simulation.memory_plan import get_available_memory_linux
from lib.simulation.compatibility import check_command_availability

# logger.setLevel("INFO")

threads_num = 48

# # Ensure these checks are only performed once at module load
# _virtuoso_checked = False
# _xyce_checked = False

# if not _virtuoso_checked:
#     virtuoso_exist, virtuoso_path = check_command_availability("virtuoso")
#     _virtuoso_checked = True

# if not _xyce_checked:
#     xyce_exist, xyce_path = check_command_availability("Xyce")
#     _xyce_checked = True


def clip_conductance_matrix(
    conductance_matrix: NDArray,
    single_cell_conductance_range: float,
    low_bound_conductance: float,
):
    conductance_matrix[conductance_matrix > single_cell_conductance_range] = (
        single_cell_conductance_range
    )
    conductance_matrix[conductance_matrix < low_bound_conductance] = (
        low_bound_conductance
    )
    return conductance_matrix


def get_tmp_file_dir() -> str:
    # caller_frame = inspect.stack()[-1]
    # # 获取文件路径
    # caller_file_path = caller_frame.filename
    # current_file_dir = os.path.dirname(os.path.abspath(caller_file_path))

    return os.path.join(os.getcwd(), ".tmp")


def get_effective_conductance_matrix(
    target_conductance_matrix: np.ndarray,
    wire_resistance,
    tmp_file_dir: str | None,
    spice_backend: Literal["xyce", "virtuoso"] = "xyce",
):
    if tmp_file_dir is None:
        tmp_file_dir = get_tmp_file_dir()

    return 1 / parallel_sim_effective_resistance_matrix(
        (1 / target_conductance_matrix).round(0),
        wire_resistance,
        spice_backend=spice_backend,
        std_out=subprocess.PIPE,
        tmp_file_dir=tmp_file_dir,
    )


def build_memristor_two_terminal_node(memristor_name: str):
    return f"{memristor_name}_L", f"{memristor_name}_R"


def single_memristor_cell(
    input_cell_node_name: str,
    output_cell_node_name: str,
    memristor_name: str,
    row_wire_resistance: float,
    col_wire_resistance: float,
    memristor_resistance: float,
):
    assert isinstance(input_cell_node_name, str)
    assert isinstance(output_cell_node_name, str)
    assert isinstance(memristor_name, str)
    memristor_left_node, memristor_right_node = build_memristor_two_terminal_node(
        memristor_name
    )

    r_row = f"R_ROW_{memristor_name} {input_cell_node_name} {memristor_left_node} {row_wire_resistance}\n"

    r_mem = f"R_MEM_{memristor_name} {memristor_left_node} {memristor_right_node} {memristor_resistance:.2f}\n"
    r_col = f"R_COL_{memristor_name} {memristor_right_node} {output_cell_node_name} {col_wire_resistance}\n"
    return r_row + r_mem + r_col


def input_dc_voltage(
    v_in_value,
    input_nodes,
):
    v_source = ""
    v_in_value = v_in_value.flatten()

    for i in range(v_in_value.size):
        v_source += f"V{input_nodes[i]} {input_nodes[i]} 0 DC {v_in_value[i]}\n"

    return v_source


def build_output_nodes(col_number: int):
    output_nodes = [f"ROUT_{i}" for i in range(col_number)]
    return output_nodes


def sim_effective_resistance_matrix(
    memristor_array_values,
    wire_resistance: float = 50,
    std_out=subprocess.PIPE,
    spice_backend: Literal["xyce", "virtuoso"] = "xyce",
    tmp_file_dir=None,
):
    conductance_matrix = np.zeros_like(memristor_array_values)
    m = memristor_array_values.shape[0]
    for i in range(m):
        print(i)
        input_quad_vector = np.zeros(m)
        input_quad_vector[i] = 1
        spice_output_current_vector = sim_memristor_vector_matrix_mul(
            input_quad_vector,
            memristor_array_values,
            wire_resistance=wire_resistance,
            std_out=std_out,
            spice_backend=spice_backend,
            tmp_file_dir=tmp_file_dir,
        )
        conductance_matrix[i, :] = spice_output_current_vector

    resistance_matrix = 1 / conductance_matrix
    return resistance_matrix


def parallel_sim_effective_resistance_matrix(
    memristor_array_res_values,
    wire_resistance: float | tuple[float, float] = 50,
    std_out=subprocess.PIPE,
    spice_backend: Literal["virtuoso", "xyce"] = "virtuoso",
    tmp_file_dir: str = get_tmp_file_dir(),
):
    """Calculate effective conductance matrix from target resistance matrix.
    This function computes the effective conductance matrix by simulating current outputs
    for unit voltage inputs using SPICE simulation. For each row, it applies a unit voltage
    to that row and measures the resulting currents to get the conductance values.
    Args:
        spice_backend:
        std_out: specify the console stdout
        wire_resistance: wire resistance of array
        memristor_array_res_values (numpy.ndarray): Target resistance matrix with shape (m, n)
            containing resistance values for the crossbar array.
    Returns:
        numpy.ndarray: Effective conductance matrix with shape (m, n) where each element
            represents the measured conductance between corresponding input-output pairs.
    Notes:
        - Uses SPICE simulation with wire resistance effects included
        - Processes row by row, applying unit voltage to one row at a time
        - Final conductance matrix represents the actual input-output relationship
          accounting for wire resistance effects

    """

    row_nums = memristor_array_res_values.shape[0]
    # tmp_file_dir = get_tmp_file_dir()

    process_list: list[Process] = []

    for i in range(row_nums):
        logger.debug(f"Parallel sim effective {i}")
        input_quad_vector = np.zeros(row_nums)
        input_quad_vector[i] = 1
        process_list.append(
            Process(
                target=sim_memristor_vector_matrix_mul,
                args=(
                    input_quad_vector,
                    memristor_array_res_values,
                    wire_resistance,
                    std_out,
                    spice_backend,
                    tmp_file_dir,
                    str(i),
                ),
            )
        )

        process_list[-1].start()
        if i % 128 == 0:
            print("Parallel sim effective {}/{}".format(i, row_nums))

        if len(process_list) >= threads_num:
            # assert row_nums < 600, "内存占用可能过多"
            # logger.error('memory available' + str(get_available_memory_linux()) + 'GB')
            process_list[0].join()
            # logger.error('process 0 complete, memory available' + str(get_available_memory_linux()) + 'GB')
            process_list.pop(0)

            print("Joined one")
            # logger.error('process 0 pop, memory available' + str(get_available_memory_linux()) + 'GB')
    print(f"All queqe: {len(process_list)}")
    done_nums = 0
    for p in process_list:
        p.join()
        done_nums += 1
        # if done_nums % 64 == 0:
        print("Joined {}".format(done_nums))

    resistance_matrix = 1.0 / read_spice_output_results(
        memristor_array_res_values.shape, tmp_file_dir, spice_backend=spice_backend
    )
    return resistance_matrix


def read_spice_output_results(
    shape, file_dir, spice_backend: Literal["virtuoso", "xyce"] = "virtuoso"
):
    result_matrix = np.zeros(shape)
    if spice_backend == "virtuoso":
        for i in range(shape[0]):
            with open(os.path.join(file_dir, f"result_{i}.out"), "r") as f:
                lines = f.readlines()
            data_line = lines[7]
            data_line.strip()
            data_str = data_line.split()
            result_matrix[i, :] = np.array(data_str[1:], dtype=float)

    elif spice_backend == "xyce":
        for i in range(shape[0]):
            values_np = np.loadtxt(
                os.path.join(file_dir, f"result_{i}.csv"),
                delimiter=",",
                skiprows=1,
                dtype=float,
            )
            result_matrix[i, :] = values_np.flatten()

    else:
        raise ValueError("spice backend not supported")

    return result_matrix


def parallel_sim_vmm(
    input_voltages_vector,
    target_res_matrix: NDArray,
    wire_resistance: float | tuple[float, float],
    spice_backend: Literal["virtuoso", "xyce"] = "virtuoso",
):
    """
    Simulates vector-matrix multiplication using a memristor crossbar array with SPICE.
    This function performs a vector-matrix multiplication simulation using memristors,
    taking into account wire resistance effects in the crossbar array.
    Parameters:
        wire_resistance:
            wire
        target_res_matrix: ndarray:
            2D array representing the target resistance matrix of memristors
        input_voltages_vector: ndarray:
            1D array containing input voltages to be applied
    Returns:
        ndarray:
            (number, cols) The output current vector resulting from the vector-matrix multiplication
    Notes:
        Uses virtuoso as the SPICE backend simulator for accurate electrical simulation
        including wire resistance effects.
    """

    row_nums = target_res_matrix.shape[0]
    tmp_file_dir = get_tmp_file_dir()

    process_list: list[Process] = []
    input_voltages_vector = input_voltages_vector.reshape((-1, row_nums))
    for i in range(input_voltages_vector.shape[0]):
        logger.debug(f"Parallel VMM {i}")
        input_voltage = input_voltages_vector[i, :]
        process_list.append(
            Process(
                target=sim_memristor_vector_matrix_mul,
                args=(
                    input_voltage,
                    target_res_matrix,
                    wire_resistance,
                    subprocess.PIPE,
                    spice_backend,
                    tmp_file_dir,
                    str(i),
                ),
            )
        )

        process_list[-1].start()

        if len(process_list) >= threads_num:
            assert row_nums < 600, "内存占用可能过多"
            # logger.error('memory available' + str(get_available_memory_linux()) + 'GB')
            process_list[0].join()
            # logger.error('process 0 complete, memory available' + str(get_available_memory_linux()) + 'GB')
            process_list.pop(0)
            # logger.error('process 0 pop, memory available' + str(get_available_memory_linux()) + 'GB')

    for p in process_list:
        p.join()

    return read_spice_output_results(
        (input_voltages_vector.shape[0], target_res_matrix.shape[1]),
        tmp_file_dir,
        spice_backend=spice_backend,
    )


def sim_memristor_vector_matrix_mul(
    input_voltage_values,
    memristor_resistance_array: NDArray[Any],
    wire_resistance: float | tuple[float, float] = 50,
    std_out=subprocess.PIPE,
    spice_backend: Literal["xyce", "virtuoso"] = "xyce",
    tmp_file_dir=None,
    file_suffix: str = "0",
):
    """生成并模拟一个包含电阻和忆阻器的电路网络。

    Args:
        file_suffix (str):
        tmp_file_dir:
        spice_backend:
        std_out:
        input_voltage_values (np.ndarray): 输入电压值数组，大小应为 (row_number,)，单位为伏特 (V)。
        memristor_resistance_array (np.ndarray): 忆阻器电阻值数组，大小应为 (row_number, col_number)，单位为欧姆 (Ω)。
        wire_resistance (float, optional): 导线电阻值，单位为欧姆 (Ω)。默认为 50 Ω。

    Returns:
        ndarray[tuple[int, ...], dtype[Any]] | None:
    """

    row_number, col_number = memristor_resistance_array.shape
    assert input_voltage_values.size == row_number
    assert memristor_resistance_array.shape == (row_number, col_number)

    if type(wire_resistance) is float:
        row_wire_resistance = wire_resistance
        col_wire_resistance = wire_resistance
    else:
        assert type(wire_resistance) is tuple
        row_wire_resistance = wire_resistance[0]
        col_wire_resistance = wire_resistance[1]

        # print(f"row wire: {row_wire_resistance}")

    input_voltage_nodes = [f"IN_{i}" for i in range(row_number)]

    memristor_names = defaultdict(dict)
    for i in range(row_number):
        for j in range(col_number):
            memristor_names[i][j] = f"M_{i}_{j}"
    if spice_backend == "ngspice":
        netlist = [".title mem\n"]
    elif spice_backend == "xyce":
        netlist = ["MEM\n*\n"]
    else:
        netlist = ["MEM\n"]
    # first row
    netlist.append(input_dc_voltage(input_voltage_values, input_voltage_nodes))

    first_row = str()
    for col in range(col_number):

        if col == 0:
            first_row += single_memristor_cell(
                input_cell_node_name=input_voltage_nodes[0],
                output_cell_node_name=memristor_names[1][col] + "_R",
                memristor_name=memristor_names[0][col],
                row_wire_resistance=row_wire_resistance,
                col_wire_resistance=col_wire_resistance,
                memristor_resistance=memristor_resistance_array[0][col].item(),
            )
        else:
            first_row += single_memristor_cell(
                input_cell_node_name=memristor_names[0][col - 1] + "_L",
                output_cell_node_name=memristor_names[1][col] + "_R",
                memristor_name=memristor_names[0][col],
                row_wire_resistance=row_wire_resistance,
                col_wire_resistance=col_wire_resistance,
                memristor_resistance=memristor_resistance_array[0][col].item(),
            )
    netlist.append(first_row)

    # middle rows
    for row in range(1, row_number - 1):
        for col in range(col_number):
            if col == 0:
                netlist.append(
                    single_memristor_cell(
                        input_cell_node_name=input_voltage_nodes[row],
                        output_cell_node_name=memristor_names[row + 1][col] + "_R",
                        memristor_name=memristor_names[row][col],
                        row_wire_resistance=row_wire_resistance,
                        col_wire_resistance=col_wire_resistance,
                        memristor_resistance=memristor_resistance_array[row][
                            col
                        ].item(),
                    )
                )
            else:
                netlist.append(
                    single_memristor_cell(
                        input_cell_node_name=memristor_names[row][col - 1] + "_L",
                        output_cell_node_name=memristor_names[row + 1][col] + "_R",
                        memristor_name=memristor_names[row][col],
                        row_wire_resistance=row_wire_resistance,
                        col_wire_resistance=col_wire_resistance,
                        memristor_resistance=memristor_resistance_array[row][
                            col
                        ].item(),
                    )
                )
    # last row
    for col in range(col_number):
        if col == 0:
            netlist.append(
                single_memristor_cell(
                    input_cell_node_name=input_voltage_nodes[row_number - 1],
                    output_cell_node_name="0",
                    memristor_name=memristor_names[row_number - 1][col],
                    row_wire_resistance=row_wire_resistance,
                    col_wire_resistance=col_wire_resistance,
                    memristor_resistance=memristor_resistance_array[row_number - 1][
                        col
                    ].item(),
                )
            )
        else:
            netlist.append(
                single_memristor_cell(
                    input_cell_node_name=memristor_names[row_number - 1][col - 1]
                    + "_L",
                    output_cell_node_name="0",
                    memristor_name=memristor_names[row_number - 1][col],
                    row_wire_resistance=row_wire_resistance,
                    col_wire_resistance=col_wire_resistance,
                    memristor_resistance=memristor_resistance_array[row_number - 1][
                        col
                    ].item(),
                )
            )

    # caller_frame = inspect.stack()[-1]
    # # 获取文件路径
    # caller_file_path = caller_frame.filename
    # caller_file_dir = os.path.dirname(os.path.abspath(caller_file_path))
    if not tmp_file_dir:
        # tmp_file_dir = os.path.join(caller_file_dir, ".tmp")
        tmp_file_dir = get_tmp_file_dir()
    os.makedirs(tmp_file_dir, exist_ok=True)

    if spice_backend == "ngspice":
        tmp = str()
        tmp += ".op \n"

        tmp += ".control\n"
        tmp += "run\n"
        # netlist += f"write memristor_array_out.txt "
        for col in range(col_number):
            tmp += f"save @R_COL_{memristor_names[row_number - 1][col]}[i]\n"

        tmp += "\n"
        tmp += ".endc\n"

        tmp += ".END\n"

        netlist.append(tmp)

        with open("memristor_array_netlist.cir", "w") as f:
            for line in netlist:
                f.write(line)

        # print(env["PATH"])
        # subprocess.run(["pwsh", "echo", "%PATH%"], env=env, text=True)
        subprocess.run(
            [
                "powershell",
                "ngspice_con.exe",
                "-b",
                "-r",
                "mem.raw",
                "memristor_array_netlist.cir",
            ],
            # text=True,
            # stdin=subprocess.PIPE,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )

        # with open("mem.raw", "r") as f:
        #     text = f.read()

        #     values_str = re.findall(r"[\d.]+e[-+]\d+", text)

        #     # 将字符串值列表转换为 NumPy 浮点数数组
        #     values_np = np.array(values_str, dtype=float)
    elif spice_backend == "virtuoso":

        tmp = str()
        tmp += "simulator lang=spectre \n"
        tmp += "dc1 dc\n"
        tmp += f"print "
        for col in range(col_number):
            tmp += f"I(R_COL_{memristor_names[row_number - 1][col]}),"
        tmp += f' name=dc1 to="/home/zhaoyichen/wire_resistance_simulation/.tmp/result_{file_suffix}.out" precision="%15g"\n'
        tmp += "o1 options try_fast_op=no\n"
        tmp += "simulator lang=spice \n"
        tmp += ".END\n"
        netlist.append(tmp)

        file_path = os.path.join(tmp_file_dir, f"tmp_{file_suffix}.cir")
        with open(file_path, "w") as f:
            for line in netlist:
                f.write(line)

        subprocess.run(
            [
                f"spectre ++aps -log {file_path}",
            ],
            shell=True,
            stdout=std_out,
        )
        return 0
    else:
        # Xyce
        tmp = str()
        tmp += ".OP \n"
        tmp += f".PRINT DC FILE={os.path.join(tmp_file_dir, f'result_{file_suffix}.csv')} FORMAT=CSV "
        for col in range(col_number):
            tmp += f"I(R_COL_{memristor_names[row_number - 1][col]}) "
        tmp += "\n"

        tmp += ".END\n"
        netlist.append(tmp)

        file_path = os.path.join(tmp_file_dir, f"tmp_{file_suffix}.cir")

        with open(file_path, "w") as f:
            for line in netlist:
                f.write(line)

        # xyce_exist, xyce_path = check_command_availability("Xyce")
        subprocess.run(
            [
                # "powershell",
                # "/usr/local/Xyce_7.9/bin/Xyce",
                # xyce_path if xyce_exist else "Xyce",
                "Xyce",
                "-quiet",
                file_path,
                # "-l",
                # "memristor_array_netlist.cir.log",
            ],
            stdout=std_out,
        )
        # values_np = np.loadtxt(
        #     os.path.join(tmp_file_dir, "memristor_array_netlist.cir.csv"),
        #     delimiter=",",
        #     skiprows=1,
        #     dtype=float,
        # )

    # return values_np
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    check_command_availability("Xyce")
