# Example Benchmarks

In this repository, you find source code examples accompanying our paper submission.
We provide complete and executable source codes for our `allgatherv`, `sample sort`, and `breadth-first search (BFS)` examples using:
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
- (Boost) disabled by default, use `-DKAMPING_EXAMPLES_USE_BOOST=ON` to enable, or fetch from cmake by providing `-DKAMPING_EXAMPLES_USE_BOOST_AS_SUBMODULE=ON`
- CMake 3.26

For generating job files from experiments suites from `experiment_suites/`:
- python3
- PyYAML (`pip install pyyaml` or use the `Pipfile` provided in `kaval/`)



### Compiling

```shell
cmake --preset experiments
cmake --build --preset experiments --parallel
```
(Note: in the anonymized version, it is quite likely that the source code in this repository will not compile out of the box)

## Example Codes/Benchmarks
Note that the parts of the source code which counted towards the reported LOC are marked by `//> START ...` and `//> END` (excluding blank and comment lines).
Run
```shell
cd evaluation
./run_LOC_counting
```
to obtain the reported lines of code.

### 1 Allgatherv
The main executable file is `exectuables/vector_allgather.cpp`. The vector allgather implementation for each binding can be found in `include/vector_allgather/`.

### 2 Sample Sort
The main executable file is `exectuables/sorting.cpp`. The sample sort implementation for each binding can be found in `include/sorting/bindings/`.

#### Running
For reproducing our experiments run
```shell
python ./kaval/run-experiments.py sorting                 \
              --machine             generic-job-file
```
this will create a directory containing generic MPI jobfiles for all experiment configurations.

### 3 BFS
The main executable file is `exectuables/bfs.cpp`. The core BFS algorithm shared by all implementations can be found in `include/bfs/bfs_algorithm`.
The frontier exchange functionality which is different for each binding/variant can be found in the accordingly named files in `include/bfs/bindings/`.

#### Running
For reproducing our experiments run
```shell
python ./kaval/run-experiments.py bfs                     \ 
              --machine             generic-job-file
```
this will create a directory containing generic MPI jobfiles for all experiment configurations.

### Making plots
Scripts for parsing log files and generating plots can be found in the `evaluation/` subdirectory
