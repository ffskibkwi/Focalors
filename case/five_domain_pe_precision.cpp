#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "io/config.h"
#include "io/csv_writer_2d.h"
#include "pe/concat/concat_solver2d.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    EnvironmentConfig& env_cfg        = EnvironmentConfig::Get();
    env_cfg.track_pe_solve_total_time = true;

    std::vector<double> acc_ranks = {4, 8, 16, 32, 64};

    for (double rank : acc_ranks)
    {
        int    n1 = rank, n2 = 2 * rank, n3 = 4 * rank;
        int    m4 = 2 * rank, m2 = 4 * rank, m5 = rank;
        double H = 1.0 / rank;

        Domain2DUniform T1(n1, m2, "T1");
        Domain2DUniform T2(n2, m2, "T2");
        Domain2DUniform T3(n3, m2, "T3");
        Domain2DUniform T4(n2, m4, "T4");
        Domain2DUniform T5(n2, m5, "T5");

        Geometry2D geo;
        geo.connect(&T2, LocationType::Left, &T1);
        geo.connect(&T2, LocationType::Right, &T3);
        geo.connect(&T2, LocationType::Down, &T4);
        geo.connect(&T2, LocationType::Up, &T5);

        geo.set_global_spatial_step(H, H);

        Variable2D p("p");
        p.set_geometry(geo);

        field2 p_T1, p_T2, p_T3, p_T4, p_T5;
        p.set_center_field(&T1, p_T1);
        p.set_center_field(&T2, p_T2);
        p.set_center_field(&T3, p_T3);
        p.set_center_field(&T4, p_T4);
        p.set_center_field(&T5, p_T5);

        long long total_mesh_size = 0;
        for (auto kv : p.field_map)
            total_mesh_size += kv.second->get_size_n();
        std::cout << "Total mesh size = " << total_mesh_size << std::endl;

        double k       = 0.1;
        auto   p_analy = [=](double x, double y) { return std::exp(-k * (x * x + y * y)); };

        auto f_rhs = [&](double x, double y) { return 4 * k * (k * (x * x + y * y) - 1) * p_analy(x, y); };

        geo.axis(&T1, LocationType::Left);
        geo.axis(&T1, LocationType::Down);

        p.fill_boundary_type(PDEBoundaryType::Dirichlet);
        p.fill_boundary_value_from_func_global(p_analy);

        ConcatPoissonSolver2D solver(&p);

        for (int i = 0; i < 100; i++)
        {
            p.set_value_from_func_global(f_rhs);
            solver.solve();
        }

        int         rank_int = static_cast<int>(rank);
        std::string out_base = "result/five_domain/p_rank_" + std::to_string(rank_int);
        IO::write_csv(p, out_base);
        IO::matlab_read_var(p, out_base + "_read.m");

        double total_l2_sq = 0.0;
        auto   calc_err    = [&](field2* f, Domain2DUniform* s) {
            double local_sum = 0;
            double offx      = s->get_offset_x();
            double offy      = s->get_offset_y();

            OPENMP_PARALLEL_FOR(reduction(+ : local_sum))
            for (int i = 0; i < f->get_nx(); ++i)
            {
                for (int j = 0; j < f->get_ny(); ++j)
                {
                    double x    = offx + (0.5 + i) * H;
                    double y    = offy + (0.5 + j) * H;
                    double diff = (*f)(i, j) - p_analy(x, y);
                    local_sum += H * H * diff * diff;
                }
            }
            return local_sum;
        };

        total_l2_sq += calc_err(&p_T1, &T1);
        total_l2_sq += calc_err(&p_T2, &T2);
        total_l2_sq += calc_err(&p_T3, &T3);
        total_l2_sq += calc_err(&p_T4, &T4);
        total_l2_sq += calc_err(&p_T5, &T5);

        std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq) << std::endl;
    }

    return 0;
}