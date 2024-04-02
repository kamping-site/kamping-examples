import pandas as pd
from pathlib import Path
import json
import os
import argparse
import sys


def get_json_path(data, value_path):
    current = data
    for key in value_path:
        if key not in current:
            return None
        current = current[key]
    return current


def chunks(lst, n):
    """Yields successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_logs_from_directory(directory, mpi_type, value_paths):
    """
    Read all the json files in the directory and return a pandas dataframe
    """
    glob_pattern = '*timer.json'
    log_path = Path(directory) / "output"
    df_entries = []
    for file in log_path.glob(glob_pattern):
        with open(file) as log:
            if os.stat(file).st_size == 0:
                continue
            output = json.load(log)
            base_entry = {}
            to_expand = {}
            for name, path in value_paths.items():
                value = get_json_path(output, path)
                if not isinstance(value, list):
                    base_entry[name] = value
                else:
                    to_expand[name] = value
            iterations = get_json_path(output, ["info", "iterations"])
            if iterations is None:
                # if no iterations are specified, we assume that the length of the longest list is the number of iterations
                max_iterations = max([len(value) for value in to_expand.values()])
                min_iterations = min([len(value) for value in to_expand.values()])
                assert max_iterations == min_iterations
                iterations = max_iterations

            else:
                iterations = int(iterations)
            for iteration in range(0, iterations):
                entry = base_entry.copy()
                entry["iteration"] = iteration
                entry["mpi_type"] = mpi_type
                for key, value in to_expand.items():
                    window_size = len(value) // iterations
                    window = value[iteration * window_size:(iteration + 1) *
                                    window_size]
                    if len(window) == 1:
                        window = window[0]
                        entry[key] = window
                        df_entries.append(entry)
    return pd.DataFrame(df_entries)
