<p align="center">
  <img src="Focalors_logo.png" alt="Focalors Logo" width="220"/>
</p>

<div align="center">

# Focalors

基于 FFT 区域分解算法的高性能不可压缩流求解器  
High-Performance Incompressible Multi-Physics CFD Solver Based on FFT-DDM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/Build-CMake-green.svg)](https://cmake.org/)
[![Parallel](https://img.shields.io/badge/Parallel-MPI%20%2B%20OpenMP-orange.svg)](https://www.open-mpi.org/)

中文&nbsp;&nbsp;|&nbsp;&nbsp;[English](./README_en.md)&nbsp;&nbsp;|&nbsp;&nbsp;[日本語](./README_ja.md)

</div>

---

## 简介

**Focalors** 是一款面向拼接正交几何（如十字流道、T 型混合器、阶梯流道等）的高性能不可压缩流体力学（CFD）求解器。

在传统不可压缩流求解中，规则拼接多块区域上的压力泊松方程通常依赖多重网格或 Krylov 子空间迭代求解。Focalors 采用基于快速傅里叶变换的区域分解算法（FFT-based Domain Decomposition Method, FFT-DDM），将拼接域上的求解复杂度控制在与单域 FFT 相当的 O(N log N) 准线性级别。

求解器支持不可压缩 Navier-Stokes 方程求解，并集成了非牛顿流变模型、低磁雷诺数磁流体力学（MHD）、浸入边界法（IBM）与拉格朗日颗粒追踪等物理模块。

---

## 主要模块与功能

- **泊松方程求解 (`poisson_base`)**
  - 基于 FFT-DDM 算法求解二维/三维拼接正交几何上的泊松方程，计算复杂度为 O(N log N)。
  - 支持物理静压重构求解（`Physical PE Solver`）与全 Neumann 边界条件的可解性相容修正。
- **流场求解 (`ns`)**
  - 基于分步投影法（Fractional-Step Method）求解不可压缩 Navier-Stokes 方程。
  - 支持 MAC 交错网格与同位网格离散。
  - 包含被动标量对流扩散求解器（支持 QUICK 等高阶格式）。
- **非牛顿流变模型 (`ns`)**
  - 支持幂律模型（Power-law）、Carreau 模型与 Bird-Carreau 模型。
  - 支持黏度亚松弛调节与非线性剪切率迭代更新。
- **低磁雷诺数磁流体力学 (`ns`)**
  - 在低 Rm 近似下耦合电磁场与流场，求解电势泊松方程并计算 J × B 洛伦兹力项。
  - 支持绝缘壁面 (J · n = 0) 与导电壁面边界条件。
- **浸入边界法与颗粒追踪 (`ibm_Uhlmann`, `ibm_MirrorPoint`, `particle`)**
  - 提供基于离散 Delta 核函数的 Uhlmann 直接力法与锐利界面镜像点法。
  - 支持拉格朗日颗粒动力学追踪与空间映射。
- **并行计算架构**
  - 支持 MPI 进程间通信与 OpenMP 线程级共享内存的混合并行（Hybrid 模式）。
- **数据 I/O 与存储 (`io`)**
  - 三维算例支持将多子域拼接并导出为标准 VTK XML 非结构网格（`.vtu`）文件，可直接导入 ParaView 进行流场三维可视化。
  - 二维算例与监测点支持按子域输出各变量（速度、压力、电势等）的结构化 `.csv` 场数据与残差历程，便于后续分析与绘图。

---

## 目录结构

```text
Focalors/
├── case/               # 算例源码（基准算例 benchmark、验证算例 validation、测试用例 tests）
├── ibm_MirrorPoint/    # 镜像点法浸入边界求解
├── ibm_Uhlmann/        # Uhlmann 直接力浸入边界法
├── io/                 # 数据输入输出（VTK XML 写入、CSV 探针、重启检查点）
├── ns/                 # 不可压缩 N-S 方程求解核心（MAC 离散、MHD 模块、非牛顿流体、标量求解）
├── particle/           # 拉格朗日颗粒追踪模块
├── poisson_base/       # FFT 区域分解核心泊松求解器（基础数据结构与拼接泊松内核）
└── tools/              # 辅助预处理与对称变换工具
```

---

## 环境依赖

- **操作系统**: Linux（推荐 CentOS 7+、Ubuntu 20.04+ 或集群环境）
- **编译器**: 支持 C++17 标准的编译器（GCC >= 9.0、Clang >= 9.0 或 Intel oneAPI icpx）
- **CMake**: >= 3.10
- **FFTW3**: >= 3.3.8（快速傅里叶变换库，建议编译多线程支持）
- **MPI**: OpenMPI、MPICH 或 Intel MPI
- **OpenMP**: 支持多线程并行
- **VTK**: 8.x / 9.x（需包含 CommonCore、IOXML、IOParallelXML、ParallelMPI 等组件）

---

## 编译与配置

### 1. 编译类型

根目录 `CMakeLists.txt` 提供三种预设编译模式（通过 `-DCMAKE_BUILD_TYPE` 指定）：

- **`Hybrid`**（默认推荐）：启用 OpenMP 与 MPI 混合并行，编译参数为 `-g -O3 -DHYBRID -DNDEBUG`。
- **`PureMPI`**：关闭 OpenMP，仅保留 MPI，编译参数为 `-g -O3 -DPUREMPI -DNDEBUG`。
- **`Debug`**：关闭 OpenMP 与优化，生成调试符号，编译参数为 `-g -O0`。

### 2. 构建流程

```bash
# 1. 导出第三方库路径（如未在默认搜索路径）
export FFTW_ROOT=/path/to/fftw3
export VTK_DIR=/path/to/vtk/lib/cmake/vtk-9.x

# 2. 配置与生成构建目录（推荐 Hybrid 混合并行模式）
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=Hybrid

# 3. 并行编译
cmake --build build --parallel $(nproc)
```

编译生成的可执行文件统一存放于 `build/bin/` 目录下。

---

## 运行与示例

### 1. 算例目标命名规则

`case/` 目录下的所有 `.cpp` 源码由 `case/CMakeLists.txt` 自动递归构建，可执行目标名称为相对路径中的斜杠替换为下划线：

- `case/benchmark/2d/cross_shaped_channel.cpp`  
  -> `build/bin/benchmark_2d_cross_shaped_channel`
- `case/benchmark/3d/cross_shaped_channel.cpp`  
  -> `build/bin/benchmark_3d_cross_shaped_channel`

### 2. 运行示例

以二维十字流道基准算例为例：

```bash
# 编译指定算例目标
cmake --build build --target benchmark_2d_cross_shaped_channel --parallel $(nproc)

# 设置 OpenMP 线程数（可根据机器实际核心数配置）
export OMP_NUM_THREADS=$(nproc)

# 执行计算
./build/bin/benchmark_2d_cross_shaped_channel
```

对于多进程 MPI 算例（如二维多域泊松测试）：

```bash
mpirun -np 4 ./build/bin/precision_2d_five_domain_pe_mpi
```

计算输出的场数据（三维 `.vtu` 或二维 `.csv`）将保存至 `result/` 或对应算例指定的输出目录。三维流场可直接使用 ParaView 查看流线与涡量分布，二维数据与时间序列探针可结合常用数据处理工具进行分析与绘图。

---

## 自定义算例编写

在 `case/` 目录下创建新的 `.cpp` 源文件即可添加算例，基本结构示例：

```cpp
#include "base/domain/geometry2d.h"
#include "ns/ns_solver2d.h"
#include "io/csv_writer_2d.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // 1. 构建拼接几何（Geometry）与网格划分
    // 2. 设置速度、压力、电势或标量边界条件
    // 3. 配置物理参数（Re, Ha, 幂律指数 n, 表观黏度等）
    // 4. 初始化求解器并推进时间步循环
    // 5. 写入 VTK 场数据与 CSV 监测探针

    MPI_Finalize();
    return 0;
}
```

再次执行 `cmake --build build` 时，系统将自动识别并构建新增算例。

---

## 参考文献

相关核心算法与流场求解器文献：

1. **拼接几何 FFT 泊松快速求解算法 (FFT-DDM / GMRES)**:
   - Zichao Jiang, Jiacheng Lian, Zhuolin Wang. *A FFT-based GMRES for fast solving of Poisson equation in concatenated geometry*.
   - **arXiv**: [https://arxiv.org/abs/2509.23180](https://arxiv.org/abs/2509.23180)
   - **DOI**: [https://doi.org/10.48550/arXiv.2509.23180](https://doi.org/10.48550/arXiv.2509.23180)

2. **并行不可压缩流与重叠浸入边界算法 (IBM-PISO & FFT Solver)**:
   - Jiacheng Lian, Qinghe Yao, Zichao Jiang. *An Overlapping IBM-PISO Algorithm with an FFT-Based Poisson Solver for Parallel Incompressible Flow Simulations*.
   - ***Fluids***, 2025, 10(7), 176.
   - **DOI**: [https://doi.org/10.3390/fluids10070176](https://doi.org/10.3390/fluids10070176)

---

## 开发团队

本项目由 Zichao Jiang、Jiacheng Lian、Zhaowei Zhang 共同开发与维护。

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
