cmake_minimum_required(VERSION 3.26)
project(kamping-examples)

option(KAMPING_EXAMPLES_USE_BOOST "Use Boost.MPI" OFF)

find_package(MPI REQUIRED)
if(KAMPING_EXAMPLES_USE_BOOST)
  find_package(
    Boost
    COMPONENTS mpi
    REQUIRED)
endif()

include(FetchContent)

FetchContent_Declare(
  Format.cmake
  GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
  GIT_TAG v1.8.1)
FetchContent_MakeAvailable(Format.cmake)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.1
  SYSTEM OVERRIDE_FIND_PACKAGE)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.13.0
  SYSTEM OVERRIDE_FIND_PACKAGE)

FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG 34f2a73
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

FetchContent_Declare(
  kagen
  GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen.git
  GIT_TAG v1.0.2
  SYSTEM OVERRIDE_FIND_PACKAGE)

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

find_package(kamping REQUIRED)
find_package(mpl REQUIRED)
find_package(rwth-mpi REQUIRED)
add_library(rwth::mpi ALIAS mpi)

add_library(all_mpi_wrappers INTERFACE)
target_link_libraries(all_mpi_wrappers INTERFACE MPI::MPI_CXX rwth::mpi
                                                 kamping::kamping mpl::mpl)
if(KAMPING_EXAMPLES_USE_BOOST)
  target_link_libraries(all_mpi_wrappers INTERFACE Boost::mpi)
endif()

find_package(KaGen REQUIRED)

add_executable(sorting sorting.cpp)
target_link_libraries(sorting PUBLIC all_mpi_wrappers)
target_link_libraries(sorting PUBLIC CLI11::CLI11)
target_compile_features(sorting PUBLIC cxx_std_20)
if(KAMPING_EXAMPLES_USE_BOOST)
  target_compile_definitions(sorting PUBLIC KAMPING_EXAMPLES_USE_BOOST)
endif()

add_executable(type_ping_pong type_ping_pong.cpp)
target_link_libraries(type_ping_pong PUBLIC MPI::MPI_CXX kamping::kamping)
target_link_libraries(type_ping_pong PUBLIC CLI11::CLI11)
target_compile_features(type_ping_pong PUBLIC cxx_std_20)

add_executable(bfs bfs.cpp)
target_link_libraries(bfs PUBLIC MPI::MPI_CXX kamping::kamping KaGen::KaGen
                                 CLI11::CLI11 spdlog::spdlog)
target_compile_features(bfs PUBLIC cxx_std_20)
