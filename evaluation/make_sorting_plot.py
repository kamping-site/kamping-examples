#!/usr/bin/env python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 7,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.preamble': "\n".join([
        r"\usepackage{xspace}",
        r"\newcommand{\kamping}{\textsc{KaMPIng}\xspace}",
    ])
})

# plt.style.use('tableau-colorblind10')
# plt.style.use('seaborn-v0_8-colorblind')

style = {
    'mpi': {
        'label': 'MPI~\cite{Forum2023}',
        'color': 'tab:blue',
        'marker': 's'
    },
    'boost': {
        'label': 'Boost.MPI~\cite{Gregor2007}',
        'color': 'tab:orange',
        'marker': 'o'
    },
    'mpl': {
        'label': 'MPL~\cite{Bauke2015}',
        'color': 'tab:green',
        'marker': 'v'
    },
    'rwth': {
        'label': 'RWTH-MPI~\cite{Demiralp2023}',
        'color': 'tab:purple',
        'marker': '<'
    },
    'kamping': {
        'label': r"\kamping",
        'color': 'tab:red',
        'marker': '>'
    },
}
hue_kws = {
    'color': [args['color'] for args in style.values()],
    'marker': [args['marker'] for args in style.values()],
    'markeredgecolor': [args['color'] for args in style.values()],
}

pt = 1 / 72
fullwidth = 251 * pt
df = pd.read_csv('sorting_data.csv')
df = df.melt(id_vars='p',
             var_name='mpi_type',
             value_name='time')
print(df)
# fig, ax = plt.subplots(figsize=(fullwidth, fullwidth / 2))
fg = sns.FacetGrid(df, hue='mpi_type',
                   hue_order=style.keys(),
                   hue_kws=hue_kws)
fg.map(sns.lineplot, 'p', 'time',
       markersize=3,
       fillstyle='none',
       markeredgewidth=.5,
       linestyle='--',
       linewidth=.5,
       err_style='band',
       err_kws={'edgecolor': None},
       errorbar='sd')
fig = fg.figure
ax = fig.axes[0]
fig.set_size_inches(.8 * fullwidth, fullwidth / 3)
ax.set_ylabel('time (s)')
ax.set_xlabel('\# compute nodes (48 ranks each)')
ax.set_xscale('log', base=2)
xticks = df.p.unique()
print(xticks)
ax.set_xticks(xticks, labels=[str(int(x // 48)) for x in xticks])
new_labels = [style[algo]['label'] for algo in style.keys()]
fig.legend(handles=ax.lines, labels=new_labels,
           bbox_to_anchor=(.2, 0.98), loc='upper left',
           handletextpad=.5)
[i.set_linewidth(.1) for i in ax.spines.values()]

ax.tick_params(axis="y",direction="in", which='both', width=.1)
ax.tick_params(axis="x",direction="out", which='both', width=.1)
ax.grid('both', linewidth=.5, alpha=.5, linestyle='dotted')
fig.savefig('sorting.pdf', bbox_inches='tight')
fig.savefig('sorting.pgf', bbox_inches='tight')

# df = df[(df.mpi_type == 'intel') & (df.iteration > 0)].sort_values('p')


# # df['algorithm'] = df['algorithm'].apply(lambda x: style[x]['label'])

# hue_kws = {
#     'color': [args['color'] for args in style.values()],
#     'marker': [args['marker'] for args in style.values()],
#     'markeredgecolor': [args['color'] for args in style.values()],
# }

# fg = sns.FacetGrid(df, col='graph', hue='algorithm',
#                    col_order=['gnm-undirected_permute:false', 'rgg2d_permute:false',
#                               # 'rgg3d_permute:false',
#                               'rhg_permute:false'],
#                    col_wrap=2,
#                    hue_order=style.keys(),
#                    hue_kws=hue_kws)
# fg.map(sns.lineplot, 'p', 'total_time',
#        markersize=3,
#        fillstyle='none',
#        markeredgewidth=.5,
#        linestyle='--',
#        linewidth=.5,
#        err_style='band',
#        err_kws={'edgecolor': None},
#        errorbar='sd')
# figure = fg.figure
# for ax in figure.axes:
#     ax.set_yscale('log')
#     ax.set_xscale('log', base=2)
#     [i.set_linewidth(.1) for i in ax.spines.values()]
#     ylim = ax.get_ylim()
#     # print(ylim)
#     # ax.set_ylim(ylim[0], 60)
#     ax.set_ylabel('')
#     ax.set_xlabel('')

#     ax.tick_params(axis="y",direction="in", which='both', width=.1)
#     ax.tick_params(axis="x",direction="out", which='both', width=.1)
#     ax.grid('both', linewidth=.1, alpha=.5)
#     ax.xaxis.set_major_locator(plt.LogLocator(base=2, numticks=5))
# # figure.axes[0].set_ylabel('Total time (s)')
# figure.set_size_inches(.75 * fullwidth, .7 * fullwidth)
# figure.axes[0].set_title('GNM', y=.8)
# figure.axes[1].set_title('RGG-2D', y=.8)
# # figure.axes[2].set_title('RGG-3D')
# figure.axes[2].set_title('RHG', y=.8)
# figure.supylabel('Total time (s)', x=-0.1)
# figure.supxlabel('number of processors', y=-0.05)
# new_labels = [style[algo]['label'] for algo in style.keys()]
# figure.legend(handles=figure.axes[0].lines, labels=new_labels, bbox_to_anchor=(1, .1), loc='lower right', handletextpad=.5)
