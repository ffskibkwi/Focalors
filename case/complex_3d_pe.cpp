#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "io/config.h"
#include "io/csv_writer_3d.h"
#include "pe/concat/concat_solver3d.h"
#include <cmath>
#include <iostream>
#include <map> // 补齐头文�?
#include <vector>
#include <string>

int main(int argc, char* argv[])
{
    // 修正 EnvironmentConfig 构�?
    EnvironmentConfig* env = new EnvironmentConfig();
    env->showGmresRes      = false;
    env->showCurrentStep   = true;
    // Enable debug output
    env->debugMode      = false;
    env->debugOutputDir = "./result/debug_output";

    std::vector<double> acc_ranks = {4, 8, 16};

    for (double rank : acc_ranks)
    {
        // 1. 几何参数
        int    m1 = rank, m3 = rank * 2, m5 = rank, m6 = rank * 3, m8 = rank;     // x
        int    n1 = rank, n2 = 2 * rank, n4 = 4 * rank, n7 = rank * 3, n9 = rank; // y
        int    l1 = rank;
        double H = 1.0 / rank;

        // 2. 构造多区域�?
        Domain3DUniform T1(m1, n1, l1, "T1");
        Domain3DUniform T2(m1, n2, l1, "T2");
        Domain3DUniform T3(m3, n2, l1, "T3");
        Domain3DUniform T4(m3, n4, l1, "T4");
        Domain3DUniform T5(m5, n2, l1, "T5");
        Domain3DUniform T6(m6, n2, l1, "T6");
        Domain3DUniform T7(m6, n7, l1, "T7");
        Domain3DUniform T8(m8, n2, l1, "T8");
        Domain3DUniform T9(m8, n9, l1, "T9");

        Geometry3D geo;
        geo.connect(&T1, LocationType::Up, &T2);
        geo.connect(&T2, LocationType::Right, &T3);
        geo.connect(&T3, LocationType::Up, &T4);
        geo.connect(&T3, LocationType::Right, &T5);
        geo.connect(&T5, LocationType::Right, &T6);
        geo.connect(&T6, LocationType::Down, &T7);
        geo.connect(&T6, LocationType::Right, &T8);
        geo.connect(&T8, LocationType::Up, &T9);

        geo.set_global_spatial_step(H, H, H);

        // 3. 变量与场初始�?(修正 Variable 构�?
        Variable3D p("p");
        p.set_geometry(geo);

        field3 p_T1("p_T1"), p_T2("p_T2"), p_T3("p_T3"), p_T4("p_T4"), p_T5("p_T5"), p_T6("p_T6"), p_T7("p_T7"),
            p_T8("p_T8"), p_T9("p_T9");
        p.set_center_field(&T1, p_T1);
        p.set_center_field(&T2, p_T2);
        p.set_center_field(&T3, p_T3);
        p.set_center_field(&T4, p_T4);
        p.set_center_field(&T5, p_T5);
        p.set_center_field(&T6, p_T6);
        p.set_center_field(&T7, p_T7);
        p.set_center_field(&T8, p_T8);
        p.set_center_field(&T9, p_T9);

        double k = 0.1;

        auto p_analy = [=](double x, double y, double z) { return std::exp(-k * (x * x + y * y + z * z)); };

        auto f_rhs = [&](double x, double y, double z) {
            return (-6.0 * k + 4.0 * k * k * (x * x + y * y + z * z)) * p_analy(x, y, z);
        };

        // 定义区域坐标偏移映射，提高可读�?
        geo.axis(&T1, LocationType::Left);
        // geo.global_move_x(0.5 * H);

        geo.axis(&T1, LocationType::Down);
        // geo.global_move_y(0.5 * H);

        geo.axis(&T1, LocationType::Front);

        // 4. 设置边界条件
#if 0
        p.set_boundary_type(&T1,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Down, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T1, LocationType::Left, p_analy);
        p.set_boundary_value_from_func_global(&T1, LocationType::Right, p_analy);
        p.set_boundary_value_from_func_global(&T1, LocationType::Down, p_analy);

        p.set_boundary_type(&T2,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Up, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T2, LocationType::Left, p_analy);
        p.set_boundary_value_from_func_global(&T2, LocationType::Up, p_analy);

        p.set_boundary_type(&T3,
                            {{LocationType::Down, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T3, LocationType::Down, p_analy);

        p.set_boundary_type(&T4,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Up, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T4, LocationType::Left, p_analy);
        p.set_boundary_value_from_func_global(&T4, LocationType::Right, p_analy);
        p.set_boundary_value_from_func_global(&T4, LocationType::Up, p_analy);

        p.set_boundary_type(&T5,
                            {{LocationType::Down, PDEBoundaryType::Dirichlet},
                             {LocationType::Up, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T5, LocationType::Down, p_analy);
        p.set_boundary_value_from_func_global(&T5, LocationType::Up, p_analy);

        p.set_boundary_type(&T6,
                            {{LocationType::Up, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T6, LocationType::Up, p_analy);

        p.set_boundary_type(&T7,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Down, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T7, LocationType::Left, p_analy);
        p.set_boundary_value_from_func_global(&T7, LocationType::Right, p_analy);
        p.set_boundary_value_from_func_global(&T7, LocationType::Down, p_analy);

        p.set_boundary_type(&T8,
                            {{LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Down, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T8, LocationType::Right, p_analy);
        p.set_boundary_value_from_func_global(&T8, LocationType::Down, p_analy);

        p.set_boundary_type(&T9,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Up, PDEBoundaryType::Dirichlet}});
        p.set_boundary_value_from_func_global(&T9, LocationType::Left, p_analy);
        p.set_boundary_value_from_func_global(&T9, LocationType::Right, p_analy);
        p.set_boundary_value_from_func_global(&T9, LocationType::Up, p_analy);
#endif
        p.fill_boundary_type(PDEBoundaryType::Dirichlet);
        p.fill_boundary_value_from_func_global(p_analy);

        // 5. 右端项填充与求解
        p.set_value_from_func_global(f_rhs);

        ConcatPoissonSolver3D solver(&p, env);
        solver.solve();

        int         rank_int = static_cast<int>(rank);
        std::string out_base = "result/p_rank_" + std::to_string(rank_int);
        // IO::var_to_csv(p, out_base);
        // IO::matlab_read_var(p, out_base + "_read.m");

        // 6. 误差统计 (修正 foreach 调用)
        double total_l2_sq = 0.0;
        auto   calc_err    = [&](field3* f, Domain3DUniform* s) {
            double local_sum = 0;
            double offx      = s->get_offset_x();
            double offy      = s->get_offset_y();
            double offz      = s->get_offset_z();

            OPENMP_PARALLEL_FOR(reduction(+ : local_sum))
            for (int i = 0; i < f->get_nx(); ++i)
            {
                for (int j = 0; j < f->get_ny(); ++j)
                {
                    for (int k = 0; k < f->get_nz(); ++k)
                    {
                        double diff =
                            (*f)(i, j, k) - p_analy(offx + (0.5 + i) * H, offy + (0.5 + j) * H, offz + (0.5 + k) * H);
                        local_sum += H * H * H * diff * diff;
                    }
                }
            }
            return local_sum;
        };

        total_l2_sq += calc_err(&p_T1, &T1);
        total_l2_sq += calc_err(&p_T2, &T2);
        total_l2_sq += calc_err(&p_T3, &T3);
        total_l2_sq += calc_err(&p_T4, &T4);
        total_l2_sq += calc_err(&p_T5, &T5);
        total_l2_sq += calc_err(&p_T6, &T6);
        total_l2_sq += calc_err(&p_T7, &T7);
        total_l2_sq += calc_err(&p_T8, &T8);
        total_l2_sq += calc_err(&p_T9, &T9);

        std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq) << std::endl;
    }
    delete env;
    return 0;
}



