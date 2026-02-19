import numpy as np
from lib.simulation.mem_spice import parallel_sim_vmm, check_command_availability
import time

if __name__ == "__main__":
    check_command_availability("Xyce")
    m = 512
    n = 512
    wire_resistance = 0.35

    weight = 24
    res_matrix = np.ones((m, n)) * (512000 / weight)
    start_time = time.time()
    for i in range(1):
        random_inputs = np.random.random((2, m))

        results = parallel_sim_vmm(random_inputs, res_matrix, wire_resistance=wire_resistance, spice_backend="xyce")

        ideal_results = random_inputs @ (1 / res_matrix)
    print("All time {} s".format((time.time() - start_time)))
    # plt.scatter(ideal_results.flatten(), results.flatten())
    # plt.show()
