cmake_minimum_required(VERSION 3.10)
project(PageRankHPC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Detect MPI and OpenMP automatically
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Point to header directory
include_directories(src)

# 1. Serial Executable Target
add_executable(pagerank_serial 
    executables/main_serial.cpp 
    src/graph.cpp 
    src/pagerank_serial.cpp
)
target_link_libraries(pagerank_serial PRIVATE MPI::MPI_CXX)

# 2. Pure MPI Executable Target
add_executable(pagerank_mpi 
    executables/main_mpi.cpp 
    src/graph.cpp 
    src/pagerank_mpi.cpp
)
target_link_libraries(pagerank_mpi PRIVATE MPI::MPI_CXX)

# 3. Hybrid MPI + OpenMP CPU Executable Target
add_executable(pagerank_hybrid 
    executables/main_hybrid.cpp 
    src/graph.cpp 
    src/pagerank_hybrid.cpp
)
target_link_libraries(pagerank_hybrid PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# 4. Hybrid MPI + OpenMP GPU Executable Target
add_executable(pagerank_gpu 
    executables/main_gpu.cpp 
    src/graph.cpp 
    src/pagerank_gpu.cpp
)
target_link_libraries(pagerank_gpu PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Compiler-specific GPU Offload Flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(pagerank_gpu PRIVATE -foffload=nvptx-none -fno-lto)
    target_link_options(pagerank_gpu PRIVATE -foffload=nvptx-none -fno-lto)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVCOMPILER" OR CMAKE_CXX_COMPILER_ID MATCHES "PGI")
    target_compile_options(pagerank_gpu PRIVATE -mp=gpu)
    target_link_options(pagerank_gpu PRIVATE -mp=gpu)
endif()