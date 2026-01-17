#include <iostream>
#include <mpi.h>


#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/geometry_tree.hpp"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"


#include "io/config.h"
#include "io/csv_writer_2d.h"


#include "pe_mpi/mpi_concat_poisson_solver2d.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 环境配置
    EnvironmentConfig env;
    env.showGmresRes    = true;
    env.showCurrentStep = true; // 由求解器内部仅在 root 打印，避免乱序

    // 构造与 concat_pe_only.cpp 相同的几何结构
    Geometry2D geo_tee;

    // Grid parameters
    Domain2DUniform T2(10, 10, 1.0, 1.0, "T2"); // 中心
    Domain2DUniform T1("T1");
    T1.set_nx(20);
    T1.set_lx(2.0);
    Domain2DUniform T3("T3");
    T3.set_nx(30);
    T3.set_lx(3.0);
    Domain2DUniform T4("T4");
    T4.set_ny(10);
    T4.set_ly(1.0);
    Domain2DUniform T5("T5");
    T5.set_ny(20);
    T5.set_ly(2.0);
    Domain2DUniform T6("T6");
    T6.set_nx(30);
    T6.set_lx(3.0);

    // Construct geometry
    geo_tee.add_domain(&T1);
    geo_tee.add_domain(&T2);
    geo_tee.add_domain(&T3);
    geo_tee.add_domain(&T4);
    geo_tee.add_domain(&T5);
    geo_tee.add_domain(&T6);

    geo_tee.connect(T2, LocationType::Left, T1);
    geo_tee.connect(T2, LocationType::Right, T3);
    geo_tee.connect(T2, LocationType::Down, T4);
    geo_tee.connect(T4, LocationType::Down, T5);
    geo_tee.connect(T5, LocationType::Right, T6);

    // Variable and fields
    Variable v("v");
    v.set_geometry(geo_tee);
    field2 v_T1("v_T1"), v_T2("v_T2"), v_T3("v_T3"), v_T4("v_T4"), v_T5("v_T5"), v_T6("v_T6");
    v.set_center_field(&T1, v_T1);
    v.set_center_field(&T2, v_T2);
    v.set_center_field(&T3, v_T3);
    v.set_center_field(&T4, v_T4);
    v.set_center_field(&T5, v_T5);
    v.set_center_field(&T6, v_T6);

    // 边界条件（与串行示例一致；几何相邻处由 set_geometry 自动设为 Adjacented）
    v.set_boundary_type(&T2, LocationType::Up, PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T1, LocationType::Left, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T1, LocationType::Up, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T1, LocationType::Down, PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T3, LocationType::Right, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T3, LocationType::Up, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T3, LocationType::Down, PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T4, LocationType::Left, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T4, LocationType::Right, PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T5, LocationType::Left, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T5, LocationType::Down, PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T6, LocationType::Right, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T6, LocationType::Up, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T6, LocationType::Down, PDEBoundaryType::Dirichlet);

    // 在全局 root 上设置右端项（仅 v_T4 内部区域为 1.0）
    if (world_rank == 0)
    {
        for (int i = 3; i < v_T4.get_nx() - 3; i++)
        {
            for (int j = 3; j < v_T4.get_ny() - 3; j++)
            {
                v_T4(i, j) = 1.0;
            }
        }
    }

    // MPI 版本 Concat 求解器：内部自动按层划分通信器，并在各层并行调用 MPIPoisson/MPIGMRES
    MPIConcatPoissonSolver2D solver(&v, MPI_COMM_WORLD, &env);
    solver.solve();

    // 输出：仅 root 输出 CSV
    if (world_rank == 0)
    {
        IO::field_to_csv(v_T1, "result/v_T1_mpi.txt");
        IO::field_to_csv(v_T2, "result/v_T2_mpi.txt");
        IO::field_to_csv(v_T3, "result/v_T3_mpi.txt");
        IO::field_to_csv(v_T4, "result/v_T4_mpi.txt");
        IO::field_to_csv(v_T5, "result/v_T5_mpi.txt");
        IO::field_to_csv(v_T6, "result/v_T6_mpi.txt");
    }

    MPI_Finalize();
    return 0;
}
