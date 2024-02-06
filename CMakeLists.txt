cmake_minimum_required(VERSION 3.26)
project(kamping-examples)

find_package(MPI REQUIRED)
find_package(Boost COMPONENTS mpi REQUIRED)

include(FetchContent)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.3.2
  SYSTEM OVERRIDE_FIND_PACKAGE)

FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG d2df638
  SYSTEM OVERRIDE_FIND_PACKAGE)

FetchContent_Declare(
  mpl
  GIT_REPOSITORY https://github.com/rabauke/mpl.git
  GIT_TAG v0.3.0
  SYSTEM OVERRIDE_FIND_PACKAGE)

FetchContent_Declare(
  rwth-mpi
  GIT_REPOSITORY https://github.com/VRGroupRWTH/mpi
  GIT_TAG 4.1.1
  SYSTEM OVERRIDE_FIND_PACKAGE)

find_package(CLI11 REQUIRED)
find_package(kamping REQUIRED)
find_package(mpl REQUIRED)
find_package(rwth-mpi REQUIRED)
add_library(rwth::mpi ALIAS mpi)

add_executable(sorting sorting.cpp)
target_link_libraries(sorting PUBLIC MPI::MPI_CXX kamping::kamping mpl::mpl Boost::mpi rwth::mpi)
target_link_libraries(sorting PUBLIC CLI11::CLI11)
target_compile_features(sorting PUBLIC cxx_std_20)

add_executable(type_ping_pong type_ping_pong.cpp)
target_link_libraries(type_ping_pong PUBLIC MPI::MPI_CXX kamping::kamping)
target_link_libraries(type_ping_pong PUBLIC CLI11::CLI11)
target_compile_features(type_ping_pong PUBLIC cxx_std_20)
