#!/usr/bin/env python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help="Input file")
    parser.add_argument("--output_name", help="Output file name", required=True)
    parser.add_argument(
        "--output_format", choices=["pdf", "pgf", "both"], default="pdf"
    )

    args = parser.parse_args()
    mpl.use("pgf")
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 7,
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage{xspace}",
                    r"\newcommand{\kamping}{\textsc{KaMPIng}\xspace}",
                ]
            ),
        }
    )

    # remove extension from output_name
    output_name = Path(args.output_name).stem
    output_path = Path(args.output_name).parent
    if not Path(args.INPUT).exists():
        print(f"Input file {args.INPUT} does not exist.")
        sys.exit(1)

    # plt.style.use('tableau-colorblind10')
    # plt.style.use('seaborn-v0_8-colorblind')

    style = {
        "mpi": {"label": r"MPI~\cite{Forum2023}", "color": "tab:blue", "marker": "s"},
        # "boost": {
        #    "label": r"Boost.MPI~\cite{Gregor2007}",
        #    "color": "tab:orange",
        #    "marker": "o",
        # },
        "mpl": {"label": r"MPL~\cite{Bauke2015}", "color": "tab:green", "marker": "v"},
        "rwth": {
            "label": r"RWTH-MPI~\cite{Demiralp2023}",
            "color": "tab:purple",
            "marker": "<",
        },
        "kamping": {"label": r"\kamping", "color": "tab:red", "marker": ">"},
    }

    pt = 1 / 72
    fullwidth = 251 * pt
    df = pd.read_csv(args.INPUT, sep=",", header=0)
    df = df[df.iteration > 0].sort_values("p")
    hue_kws = {
        "color": [args["color"] for args in style.values()],
        "marker": [args["marker"] for args in style.values()],
        "markeredgecolor": [args["color"] for args in style.values()],
    }
    # fig, ax = plt.subplots(figsize=(fullwidth, fullwidth / 2))
    print(df)
    fg = sns.FacetGrid(df, hue="algorithm", hue_order=style.keys(), hue_kws=hue_kws)
    fg.map(
        sns.lineplot,
        "p",
        "time",
        markersize=3,
        fillstyle="none",
        markeredgewidth=0.5,
        linestyle="--",
        linewidth=0.5,
        err_style="band",
        err_kws={"edgecolor": None},
        errorbar="sd",
    )
    fig = fg.figure
    ax = fig.axes[0]
    fig.set_size_inches(0.85 * fullwidth, fullwidth / 3)
    ax.set_ylabel("Total time (s)")
    ax.set_xlabel(r"number of processors")
    ax.set_xscale("log", base=2)
    xticks = df.p.unique()
    ax.set_xticks(xticks, labels=[str(int(x // 76)) for x in xticks])
    new_labels = [style[algo]["label"] for algo in style.keys()]
    fig.legend(
        handles=ax.lines,
        labels=new_labels,
        bbox_to_anchor=(0.2, 0.98),
        loc="upper left",
        handletextpad=0.5,
    )
    [i.set_linewidth(0.1) for i in ax.spines.values()]

    ax.tick_params(axis="y", direction="in", which="both", width=0.1)
    ax.tick_params(axis="x", direction="out", which="both", width=0.1)
    ax.grid("both", linewidth=0.5, alpha=0.5, linestyle="dotted")
    ax.annotate(
        r"$\times 76$", xy=(0.93, 0.15), xycoords="figure fraction", fontsize="small"
    )

    output_format = args.output_format
    if output_format == "pdf" or output_format == "both":
        fig.savefig(output_path / f"{output_name}.pdf", bbox_inches="tight")
    if output_format == "pgf" or output_format == "both":
        fig.savefig(output_path / f"{output_name}.pgf", bbox_inches="tight")


if __name__ == "__main__":
    main()
