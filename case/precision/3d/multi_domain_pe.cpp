#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "io/csv_writer_3d.h"
#include "pe/concat/concat_solver3d.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showGmresRes       = false;
    env_cfg.showCurrentStep    = false;
    env_cfg.debugOutputDir     = "./result/debug_output";

    std::vector<double> acc_ranks = {4, 8, 16};

    for (double rank : acc_ranks)
    {
        int    m1 = rank, m3 = rank * 2, m5 = rank, m6 = rank * 3, m8 = rank;     // x
        int    n1 = rank, n2 = 2 * rank, n4 = 4 * rank, n7 = rank * 3, n9 = rank; // y
        int    l1 = rank;
        double H  = 1.0 / rank;

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

        Variable3D p("p");
        p.set_geometry(geo);

        field3 p_T1, p_T2, p_T3, p_T4, p_T5, p_T6, p_T7, p_T8, p_T9;
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

        geo.axis(&T1, LocationType::Left);
        geo.axis(&T1, LocationType::Down);
        geo.axis(&T1, LocationType::Front);

        p.fill_boundary_type(PDEBoundaryType::Dirichlet);
        p.fill_boundary_value_from_func_global(p_analy);

        p.set_value_from_func_global(f_rhs);

        ConcatPoissonSolver3D solver(&p);
        solver.solve();

        int         rank_int = static_cast<int>(rank);
        std::string out_base = "result/p_rank_" + std::to_string(rank_int);
        // IO::write_csv(p, out_base);
        // IO::matlab_read_var(p, out_base + "_read.m");

        double total_l2_sq = 0.0;
        for (auto kv : p.field_map)
        {
            Domain3DUniform* domain = kv.first;
            field3&          f      = *kv.second;

            double local_sum = 0;
            double offx      = domain->get_offset_x();
            double offy      = domain->get_offset_y();
            double offz      = domain->get_offset_z();

            OPENMP_PARALLEL_FOR(reduction(+ : local_sum))
            for (int i = 0; i < f.get_nx(); ++i)
            {
                for (int j = 0; j < f.get_ny(); ++j)
                {
                    for (int k = 0; k < f.get_nz(); ++k)
                    {
                        double diff =
                            f(i, j, k) - p_analy(offx + (0.5 + i) * H, offy + (0.5 + j) * H, offz + (0.5 + k) * H);
                        local_sum += H * H * H * diff * diff;
                    }
                }
            }

            total_l2_sq += local_sum;
        }

        std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq) << std::endl;
    }
    return 0;
}
