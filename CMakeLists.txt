cmake_minimum_required(VERSION 3.10)
project(Focalors)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 構建型別（與 poisson_base 類似）
set(CMAKE_CONFIGURATION_TYPES
    Debug Hybrid PureMPI
    CACHE STRING "Available build configurations" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
if(CMAKE_BUILD_TYPE STREQUAL "Hybrid")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -DHYBRID -DNDEBUG")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "PureMPI")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -DPUREMPI -DNDEBUG")
endif()

# 可執行文件輸出到 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 第三方庫搜尋
cmake_policy(SET CMP0074 NEW)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/poisson_base/cmake" ${CMAKE_MODULE_PATH})
find_package(FFTW REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI)

# ------------------------ 構建 Solver 庫（引用 poisson_base 源碼） ------------------------
file(GLOB_RECURSE SOLVER_HEADER_FILES
    "${CMAKE_SOURCE_DIR}/poisson_base/base/*.h" "${CMAKE_SOURCE_DIR}/poisson_base/base/*.hpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/io/*.h"   "${CMAKE_SOURCE_DIR}/poisson_base/io/*.hpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe/*.h"   "${CMAKE_SOURCE_DIR}/poisson_base/pe/*.hpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe_mpi/*.h" "${CMAKE_SOURCE_DIR}/poisson_base/pe_mpi/*.hpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe_mpi_distributed/*.h" "${CMAKE_SOURCE_DIR}/poisson_base/pe_mpi_distributed/*.hpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/instrumentor/*.h" "${CMAKE_SOURCE_DIR}/poisson_base/instrumentor/*.hpp"
    "${CMAKE_SOURCE_DIR}/ns/*.h"   "${CMAKE_SOURCE_DIR}/ns/*.hpp"
)
file(GLOB_RECURSE SOLVER_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/poisson_base/base/*.cpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/io/*.cpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe/*.cpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe_mpi/*.cpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe_mpi_distributed/*.cpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/instrumentor/*.cpp"
    "${CMAKE_SOURCE_DIR}/ns/*.cpp"
)

# 排除 pe/poisson_old 和 pe/concat_old 目录下所有 cpp 文件
file(GLOB_RECURSE POISSON_OLD_SOURCE_FILES "${CMAKE_SOURCE_DIR}/pe/poisson_old/*.cpp")
file(GLOB_RECURSE CONCAT_OLD_SOURCE_FILES "${CMAKE_SOURCE_DIR}/pe/concat_old/*.cpp")
list(REMOVE_ITEM SOLVER_SOURCE_FILES ${POISSON_OLD_SOURCE_FILES})
list(REMOVE_ITEM SOLVER_SOURCE_FILES ${CONCAT_OLD_SOURCE_FILES})

add_library(Solver ${SOLVER_SOURCE_FILES} ${SOLVER_HEADER_FILES})
target_include_directories(Solver PUBLIC
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/poisson_base"  # 使得 #include "base/..." 可解析
    ${FFTW_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(Solver PUBLIC
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19.14>>:stdc++fs>
)
target_link_libraries(Solver PUBLIC
    ${FFTW_LIBRARIES}
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX)

# -----------------------------------------------------------------------------------------

# 構建當前倉庫的案例
add_subdirectory(case)


