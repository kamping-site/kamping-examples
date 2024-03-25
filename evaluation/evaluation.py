import pandas as pd
from pathlib import Path
import json
import os


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


def read_file(directory, mpi_type):
  glob_pattern = '*timer.json'
  log_path = Path(directory) / "output"
  value_paths = {}
  value_paths["total_time"] = ["root", "total_time", "statistics", "max"]
  value_paths["alltoall"] = ["root", "total_time", "alltoall", "statistics", "max"]
  value_paths["local_frontier_processing"] = [
      "root", "total_time", "local_frontier_processing", "statistics", "max"
  ]
  value_paths["algorithm"] = ["info", "algorithm"]
  value_paths["p"] = ["info", "p"]
  value_paths["graph"] = ["info", "graph"]
  
  df_entries = []
  for file in log_path.glob(glob_pattern):
      with open(file) as log:
          if os.stat(file).st_size == 0:
              continue
          output = json.load(log)
          iterations = get_json_path(output, ["info", "iterations"])
          if iterations is None:
              iterations = 5
          iterations = int(iterations)
          base_entry = {}
          to_expand = {}
          for name, path in value_paths.items():
              value = get_json_path(output, path)
              if not isinstance(value, list):
                  base_entry[name] = value
              else:
                  to_expand[name] = value
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
  df = pd.DataFrame(df_entries)
  def process_graph_str(entry):
      graphtype = entry[0].split("=")[1]
      permute = "permute:false"
      if len(entry)>3 and graphtype == "rgg2d":
          permute = "permute:" + entry[3].split("=")[1]
      return graphtype + "_" + permute

  df["graph"] = df["graph"].str.split(";").apply(
      lambda x: process_graph_str(x))
  df = df.sort_values('p')
  return df

directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_24_03_24/"
df_intel1 =read_file(directory, "intel")
directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_24_03_25/"
df_intel2 =read_file(directory, "intel")
directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_small_rank_configs_24_03_25/"
df_intel2_small =read_file(directory, "intel")
directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_rmat_24_03_25/"
df_intel2_rmat =read_file(directory, "intel")

directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_rgg3d_24_03_25/"
df_intel2_rgg3d =read_file(directory, "intel")

#directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_permute_24_03_24/"
#df_intel_permute =read_file(directory, "intel")
directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_ompi_24_03_24/"
df_ompi =read_file(directory, "ompi")
#directory = "/home/matthias/Promotion/data/kamping-examples/bfs_small_permute_ompi_24_03_24/"
#df_ompi_permute =read_file(directory, "ompi")
#df = pd.concat([df_intel1, df_intel2, df_intel2_rmat, df_ompi, df_intel_permute, df_ompi_permute])
df = pd.concat([df_intel1, df_intel2, df_intel2_rmat, df_intel2_small, df_intel2_rgg3d])
df = df.query('iteration > 0 and algorithm != "kamping_flattened" and graph != "rmat_permute:false"')

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf

pdf_overall = matplotlib.backends.backend_pdf.PdfPages("overall.pdf")
df_intel = df.query("mpi_type != 'ompi'")
df_ompi = df.query("mpi_type != 'intel'")
print(df_intel.head())
fg = sns.relplot(data=df_intel.sort_values(['graph', 'algorithm']),
                 x='p',
                 y='total_time',
                 hue='algorithm',
                 col='graph',
                 kind='line',
                 errorbar='sd',
                 facet_kws={'sharey': False, 'sharex': True},
                 marker='s')
fg.set(yscale="log")
plt.xscale('log', base=2)
plt.yscale('log', base=10)
#fg.figure.savefig("bfs_times.pdf")
pdf_overall.savefig(fg.figure)
pdf_overall.close()

#fg = sns.relplot(data=df.sort_values(['graph', 'exchange_type']),
#                 x='p',
#                 y='create_topology',
#                 hue='exchange_type',
#                 col='graph',
#                 kind='line',
#                 marker='s')
###fg.figure.savefig("bfs_times.pdf")
#pdf_overall.savefig(fg.figure)
#fg = sns.relplot(data=df.sort_values(['graph', 'exchange_type']),
#                 x='p',
#                 y='create_grid',
#                 hue='exchange_type',
#                 col='graph',
#                 kind='line',
#                 marker='s')
#pdf_overall.savefig(fg.figure)
#pdf_overall.close()
#
#
#
#def plot(key):
#    p = 3072
#    import seaborn as sns
#    data = df[df.p == p]
#    key_columns = ['p', 'exchange_type', 'graph']
#    data = data.set_index(key_columns).sort_index()
#    data['level'] = data[key].apply(lambda vals: list(range(len(vals))))
#    # sns.relplot(data=df, x='p', y='total_time', hue='exchange_type', col='graph', kind='line', marker='s')
#    data = data.explode([key, 'level'])
#    data = data.groupby(
#        key_columns + ['level'])[key].mean().reset_index().sort_values('level')
#    fg = sns.relplot(data=data.sort_values(['graph', 'exchange_type']),
#                     x='level',
#                     y=key,
#                     hue='exchange_type',
#                     col='graph',
#                     kind='line',
#                     marker='none',
#                     facet_kws={
#                         'sharex': False,
#                         'sharey': False
#                     })
#    return fg.figure
#    #fg.figure.savefig(f"time_per_level_{key}_p{p}.pdf")
#
#
def plot_iter(key, iteration):
    p = 8192
    import seaborn as sns
    data = df[df.p == p]
    data = data.query('algorithm == "mpi" or algorithm == "kamping_grid"')
    key_columns = ['p', 'algorithm', 'graph']
    data = data.set_index(key_columns).sort_index()
    data['level'] = data[key].apply(lambda vals: list(range(len(vals))))
    # sns.relplot(data=df, x='p', y='total_time', hue='exchange_type', col='graph', kind='line', marker='s')
    data = data.explode([key, 'level'])
    data = data[data.iteration == iteration]
    data = data.groupby(
        key_columns + ['level'])[key].mean().reset_index().sort_values('level')
    fg = sns.relplot(data=data.sort_values(['graph', 'algorithm']),
                     x='level',
                     y=key,
                     hue='algorithm',
                     col='graph',
                     kind='line',
                     marker='none',
                     facet_kws={
                         'sharex': False,
                         'sharey': False
                     })
    return fg.figure
#
#
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
#pdf.savefig(plot("alltoall"))
pdf.savefig(plot_iter("alltoall", 1))
pdf.savefig(plot_iter("alltoall", 2))
pdf.savefig(plot_iter("alltoall", 3))
#pdf.savefig(plot("local_frontier_processing"))
pdf.close()
