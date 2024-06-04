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
  GIT_TAG d295830
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

list(
  APPEND
  KAMPING_WARNING_FLAGS
  "-Wall"
  "-Wextra"
  "-Wconversion"
  "-Wnon-virtual-dtor"
  "-Woverloaded-virtual"
  "-Wshadow"
  "-Wsign-conversion"
  "-Wundef"
  "-Wunreachable-code"
  "-Wunused")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(
    APPEND
    KAMPING_WARNING_FLAGS
    "-Wcast-align"
    "-Wnull-dereference"
    "-Wpedantic"
    "-Wextra-semi"
    "-Wno-gnu-zero-variadic-macro-arguments")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(
    APPEND
    KAMPING_WARNING_FLAGS
    "-Wcast-align"
    "-Wnull-dereference"
    "-Wpedantic"
    "-Wnoexcept"
    "-Wsuggest-attribute=const"
    "-Wsuggest-attribute=noreturn"
    "-Wsuggest-override")
endif()

add_library(kamping_warnings INTERFACE)
target_compile_options(
  kamping_warnings INTERFACE ${KAMPING_WARNING_FLAGS} "-Wno-sign-conversion"
                             "-Wno-conversion" "-Wno-sign-compare")

if(KAMPING_EXAMPLES_USE_BOOST)
  target_link_libraries(all_mpi_wrappers INTERFACE Boost::mpi)
endif()

set(KAGEN_USE_SPARSE_OFF OFF CACHE BOOL "" FORCE)
#set(KAGEN_NODEPS ON CACHE BOOL "" FORCE)
#set(KAGEN_USE_XXHASH ON CACHE BOOL "" FORCE)
set(KAGEN_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(KAGEN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
find_package(KaGen REQUIRED)

add_executable(sorting executables/sorting.cpp)
target_include_directories(sorting PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sorting PUBLIC all_mpi_wrappers)
target_link_libraries(sorting PUBLIC CLI11::CLI11)
target_link_libraries(sorting PRIVATE kamping_warnings)
target_link_libraries(sorting PRIVATE spdlog::spdlog)
target_compile_features(sorting PUBLIC cxx_std_20)
if(KAMPING_EXAMPLES_USE_BOOST)
  target_compile_definitions(sorting PUBLIC KAMPING_EXAMPLES_USE_BOOST)
endif()

add_executable(vector_allgather executables/vector_allgather.cpp)
target_include_directories(vector_allgather PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(vector_allgather PUBLIC all_mpi_wrappers)
target_link_libraries(vector_allgather PUBLIC kamping_warnings)
target_link_libraries(vector_allgather PUBLIC spdlog::spdlog)
target_compile_features(vector_allgather PUBLIC cxx_std_20)
if(KAMPING_EXAMPLES_USE_BOOST)
  target_compile_definitions(vector_allgather PUBLIC KAMPING_EXAMPLES_USE_BOOST)
endif()

add_executable(type_ping_pong executables/type_ping_pong.cpp)
target_link_libraries(type_ping_pong PUBLIC MPI::MPI_CXX kamping::kamping)
target_link_libraries(type_ping_pong PUBLIC CLI11::CLI11)
target_link_libraries(type_ping_pong PRIVATE kamping_warnings)
target_compile_features(type_ping_pong PUBLIC cxx_std_20)

add_executable(bfs executables/bfs.cpp)
target_include_directories(bfs PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(bfs PUBLIC all_mpi_wrappers)
target_link_libraries(bfs PUBLIC CLI11::CLI11)
target_link_libraries(bfs PRIVATE kamping_warnings)
target_link_libraries(bfs PUBLIC KaGen::KaGen spdlog::spdlog)
target_compile_features(bfs PUBLIC cxx_std_20)
if(KAMPING_EXAMPLES_USE_BOOST)
  target_compile_definitions(bfs PUBLIC KAMPING_EXAMPLES_USE_BOOST)
endif()

