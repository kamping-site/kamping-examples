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
- [Google Sparsehash](https://github.com/sparsehash/sparsehash)
- Boost


### Compiling

```shell
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Releas 
cmake --build build
```
(Note: in the anonymized version, it is quite likely that the source code in this repository will not compile out of the box)

## Benchmarks

### Allgatherv
### Sample Sort
### BFS

The main executable file is `exectuables/bfs.cpp`. The core BFS algorithm shared by all implementations can be found in `include/bfs/bfs_algorithm`.
The frontier exchange functionality which is different for each binding/variant can be found in the accordingly named files in `include/bfs/bindings`.


