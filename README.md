<p align="center">
  <img src="Focalors_logo.png" alt="Focalors Logo" width="250"/>
</p>
<div align="center">

  # Focalors

  基于FFT区域分解算法的高性能不可压缩流求解器
</div>

<div align="center">

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Language](https://img.shields.io/badge/Language-C%2B%2B-blue.svg)](https://isocpp.org/)
  [![CMake](https://img.shields.io/badge/Build-CMake-green.svg)](https://cmake.org/)

  中文&nbsp;&nbsp;|&nbsp;&nbsp;[English](./README_en.md)&nbsp;&nbsp;|&nbsp;&nbsp;[日本語](./README_ja.md)

</div>

---

**Focalors** 

```shell
cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc) #推荐使用并行编译，将$(nproc)替换为机器的核心数
```