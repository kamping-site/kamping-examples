# Example Benchmarks

In this repository you find source code examples accompanying our paper submission.
We provide complete and executable source codes for our `allgatherv`, `sample sort`, and `breadth-first search (BFS)` benchmarks using:
- Boost.MPI
- KaMPIng
- MPI
- MPL
- RWTH-MPI

## Building

### Requirements
To compile this project you need:
- A C++17-ready compiler such as `g++` version 9 or higher or `clang` version 11 or higher.
- [OpenMPI](https://www.open-mpi.org/) or [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.pr0oht)
- (Boost)
- CMake 3.26
For generating job files:
- python3
- PyYAML (`pip install pyyaml`)



### Compiling

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build
```
(Note: in the anonymized version, it is quite likely that the source code in this repository will not compile out of the box)

## Example Codes/Benchmarks

### Allgatherv
### Sample Sort
The main executable file is `exectuables/sorting.cpp`. The sample sort implementation for each binding can be found in `include/sorting/bindings/`.
#### Running
For reproducing our experiments run
```shell
python ./experiments/run-experiments.py sorting                                             \
              --machine             generic-job-file                                        \
              --sbatch-template     ./experiments/sbatch-templates/generic_job_files.txt    \
              --command-template    ./experiments/command-templates/command_template_generic.txt
```
this will create a directory containing generic MPI jobfiles for all experiment configurations.

### BFS

The main executable file is `exectuables/bfs.cpp`. The core BFS algorithm shared by all implementations can be found in `include/bfs/bfs_algorithm`.
The frontier exchange functionality which is different for each binding/variant can be found in the accordingly named files in `include/bfs/bindings/`.

#### Running
For reproducing our experiments run
```shell
python ./experiments/run-experiments.py bfs                                                 \
              --machine             generic-job-file                                        \
              --sbatch-template     ./experiments/sbatch-templates/generic_job_files.txt    \
              --command-template    ./experiments/command-templates/command_template_generic.txt
```
this will create a directory containing generic MPI jobfiles for all experiment configurations.

