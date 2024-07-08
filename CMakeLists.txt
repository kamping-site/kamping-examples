cmake_minimum_required(VERSION 3.26)
project(kamping-examples)

option(KAMPING_EXAMPLES_USE_BOOST "Use Boost.MPI" OFF)
option(KAMPING_EXAMPLES_USE_BOOST_AS_SUBMODULE "Use Boost.MPI" OFF)

find_package(MPI REQUIRED)

include(FetchContent)

set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  kaval
  GIT_REPOSITORY https://github.com/niklas-uhl/kaval.git
  GIT_TAG f01703d
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/kaval)
FetchContent_MakeAvailable(kaval)

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
  GIT_TAG v0.1.1
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

set(KAGEN_USE_SPARSEHASH OFF)
set(KAGEN_BUILD_APPS OFF)
set(KAGEN_BUILD_EXAMPLES OFF)
FetchContent_Declare(
  kagen
  GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen.git
  GIT_TAG v1.0.2
  SYSTEM OVERRIDE_FIND_PACKAGE)

if (KAMPING_EXAMPLES_USE_BOOST_AS_SUBMODULE)
  FetchContent_Declare(
    Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
    URL_MD5 893b5203b862eb9bbd08553e24ff146a
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
    OVERRIDE_FIND_PACKAGE
  )
  set(BOOST_ENABLE_MPI TRUE)
  set(BOOST_INCLUDE_LIBRARIES mpi)
  set(KAMPING_EXAMPLES_USE_BOOST ON)
  message(WARNING "Downloading Boost, this may take some time!")
endif()
if(KAMPING_EXAMPLES_USE_BOOST)
  find_package(
    Boost
    COMPONENTS mpi
    REQUIRED)
endif()
if (KAMPING_EXAMPLES_USE_BOOST_AS_SUBMODULE)
  message(WARNING "Finished downloading Boost")
endif()



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

