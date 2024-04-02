#!/usr/bin/env python
import logparser
import sys
import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",help = "path to base directory where the experiments are located.")
    parser.add_argument("--experiments", nargs="*", help = "name of the experiments")
    args = parser.parse_args()

    value_paths = {
        "p": ["info", "p"],
        "algorithm": ["info", "algorithm"],
        "n_local": ["info", "n_local"],
        "correct": ["info", "correct"],
        'time': ['root', 'total_time', 'statistics', 'max']
    }

    df = []
    for exp in args.experiments:
        experiment_path = Path(args.path) / exp
        if not experiment_path.exists():
            print("The experiment directory {} doesn't exist".format(experiment_path))
            sys.exit(1)
        df.append(logparser.read_logs_from_directory(experiment_path, "intel", value_paths))

    df = pd.concat(df)
    df_wide = df.query('iteration > 0').pivot_table(index="p", columns=["algorithm"] , values="time", aggfunc='mean')
    df_wide.to_csv(sys.stdout, index = True)

if __name__ == "__main__":
    main()
