import pandas as pd
from pathlib import Path
import json
import os
import argparse
import sys
import logparser

def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",help = "path to base directory where the experiments are located.")
    parser.add_argument("--experiments", nargs="*", help = "name of the experiments")
    args = parser.parse_args()

    value_paths = {
        "total_time": ["root", "total_time", "statistics", "max"],
        "alltoall": ["root", "total_time", "alltoall", "statistics", "max"],
        "local_frontier_processing": [
            "root", "total_time", "local_frontier_processing", "statistics", "max"
        ],
        "algorithm": ["info", "algorithm"],
        "p": ["info", "p"],
        "graph": ["info", "graph"],
    }

    df = []
    for exp in args.experiments:
        experiment_path = Path(args.path) / exp
        if not experiment_path.exists():
            print("The experiment directory {} doesn't exist".format(experiment_path))
            sys.exit(1)
        df.append(logparser.read_logs_from_directory(experiment_path, "intel", value_paths))

    df = pd.concat(df)
    df = df.query('iteration > 0 and algorithm != "kamping"')
    def process_graph_str(entry):
        graphtype = entry[0].split("=")[1]
        permute = "permute:false"
        if len(entry)>3 and graphtype == "rgg2d":
            permute = "permute:" + entry[3].split("=")[1]
        return graphtype + "_" + permute

    df.loc[:, "graph"] = df["graph"].str.split(";").apply(
        lambda x: process_graph_str(x))
    cols = ['graph','p','algorithm','mpi_type','iteration','total_time']
    df = df.sort_values(cols)
    df = df[cols]
    df.to_csv(sys.stdout, index = False)

if __name__ == "__main__":
    main()
