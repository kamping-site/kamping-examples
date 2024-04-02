import json
import os
import sys
import argparse
from pathlib import Path
import pandas as pd

def read_file(file):
  begin = sys.maxsize
  end = 0
  blank = 0
  comments = 0
  application = None
  implementation = None
  for idx, line in enumerate(file):
      if "//> START" in line:
          begin = idx
          split_line = line.strip().split(" ")
          print(split_line)
          application = split_line[2]
          implementation = split_line[3]
      if "//> END" in line:
          end = idx
          break
      if "//" in line and idx > begin:
          comments += 1
      if not line.strip() and idx > begin:
          blank += 1
  assert(end > begin)
  entry = {}
  entry["begin"] = begin
  entry["end"] = end
  entry["comment_lines"] = comments
  entry["blank_lines"] = blank
  # do not count blank lines and comment lines
  entry["length"] = (end - ((begin + 1) + comments + blank))
  entry["application"] = application
  entry["implementation"] = implementation
  return entry

def read_files(directory):
  path = Path(directory)
  df_entries = []
  for file_path in path.glob("*"):
      with open(file_path) as file:
          print(file_path)
          entry = read_file(file)
          df_entries.append(entry)
  df = pd.DataFrame(df_entries)
  return df

parser = argparse.ArgumentParser()
parser.add_argument("--paths",help = "path to base directory where the sources are located.", nargs = "*")
args = parser.parse_args()
dfs = []
for path in args.paths:
    df = read_files(path)
    dfs.append(df)
df = pd.concat(dfs)

cols = ['application','implementation','length','begin','end','comment_lines','blank_lines']
df = df.sort_values(cols)
df = df[cols]
df.reset_index(inplace = True, drop = True)
print(df)




