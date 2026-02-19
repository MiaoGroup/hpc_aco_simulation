import os


def get_results_data_dir():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = os.path.expanduser("~")
    if os.access(home_dir, os.W_OK):
        data_dir = os.path.join(home_dir, "hpc_xyce_results")
        # os.makedirs(data_dir, exist_ok=True)
        # check if exists and writable
        if os.path.exists(data_dir) and os.access(data_dir, os.W_OK):

            return data_dir
        else:
            print(
                f"Warning: Cannot write to {data_dir}. Falling back to local data directory."
            )
            raise PermissionError

    data_dir = os.path.join(current_file_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir
