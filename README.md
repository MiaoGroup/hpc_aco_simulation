# ACO: Adaptive Conductance Optimization Simulation

This repository contains the simulation framework for the **Adaptive Conductance Optimization (ACO)** algorithm. The program is specifically optimized for high-performance computing (**HPC**) environments.

## 📋 Prerequisites

To run these simulations, your environment must meet the following requirements:

* **HPC Cluster:** Access to a High-Performance Computing cluster.
* **Job Scheduler:** The cluster must support **IBM Spectrum LSF** (Load Sharing Facility).
* **Container Runtime:** **Apptainer** (formerly Singularity) must be installed to manage the simulation environment.

## 🚀 Getting Started

### 1. Environment
The simulation runs inside an Apptainer container to ensure consistency across different HPC nodes. 

While the framework supports multiple backends, only the **Xyce** environment (`xyce.sif`) is provided in this open-source repository.

| Backend | Status | File Provided | Tested on HPC |
| :---:| :---: | :---: | :--: |
| Xyce | ✅ Supported | xyce.sif | ✅ |
| Ngspice | ✅ Supported | No | ❌ |
| Virtuoso | ✅ Supported | No | ❌ |

### 2. Job Submission
> We use `bsub` to submit tasks to the LSF scheduler. 

We provide a pre-configured `bsub` submission script `run_bsub.sh`. Depending on your specific HPC environment, you may need to adjust the node parameters within the script.

The current configuration uses the following command:

```bash
bsub -q 6330ib -n 56 -o logs/output%J.log apptainer exec --env PYTHONPATH="${PYTHONPATH_VAL}" "${APPTAINER_IMAGE}" python3 "${TEST_FILE}" "${@:2}"

```

In this example, tasks are submitted to the **6330ib** queue using **56 CPU cores**. Please modify these values to match your cluster's requirements.

#### 1. Set Permissions

After modifying the script, ensure it has the necessary execution permissions:

```bash
chmod +x run_bsub.sh
```

#### 2. Submit the Job

To submit a task to the HPC node, execute the script followed by the target Python file:

```bash
./run_bsub.sh optim_basic/aco_optim.py
```

### 3. Find Results

By default, the results are stored in `~/hpc_xyce_results`

You can configure the results directory by modifying `get_results_data_dir()` in `lib/path_util.py`.

---

## Simulation Tasks

| Name | Description |
| :--- | :--- |
| [optim_basic](./optim_basic/aco_optim.py) | Simulation code of Fig. 3  |
| [optim_with_noise](./optim_with_noise/aco_optim.py) | Simulation under realistic non-ideal device conditions|
| [reference+model](./reference_model/refer.py) | Simulation of effective transfer matrix representing circuit-level input-output relationship.

