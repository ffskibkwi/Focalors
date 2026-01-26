#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable.h"
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
    EnvironmentConfig* env         = new EnvironmentConfig();
    env->track_pe_solve_total_time = true;

    std::vector<double> acc_ranks = {4, 8, 16, 32, 64};

    for (double rank : acc_ranks)
    {
        int    m1 = rank, m3 = rank * 2, m5 = rank, m6 = rank * 3, m8 = rank;     // x
        int    n1 = rank, n2 = 2 * rank, n4 = 4 * rank, n7 = rank * 3, n9 = rank; // y
        double H = 1.0 / rank;

        Domain2DUniform T1(m1, n1, "T1");
        Domain2DUniform T2(m1, n2, "T2");
        Domain2DUniform T3(m3, n2, "T3");
        Domain2DUniform T4(m3, n4, "T4");
        Domain2DUniform T5(m5, n2, "T5");
        Domain2DUniform T6(m6, n2, "T6");
        Domain2DUniform T7(m6, n7, "T7");
        Domain2DUniform T8(m8, n2, "T8");
        Domain2DUniform T9(m8, n9, "T9");

        Geometry2D geo;
        geo.connect(&T1, LocationType::Up, &T2);
        geo.connect(&T2, LocationType::Right, &T3);
        geo.connect(&T3, LocationType::Up, &T4);
        geo.connect(&T3, LocationType::Right, &T5);
        geo.connect(&T5, LocationType::Right, &T6);
        geo.connect(&T6, LocationType::Down, &T7);
        geo.connect(&T6, LocationType::Right, &T8);
        geo.connect(&T8, LocationType::Up, &T9);

        geo.set_global_spatial_step(H, H);

        Variable p("p");
        p.set_geometry(geo);

        field2 p_T1("p_T1"), p_T2("p_T2"), p_T3("p_T3"), p_T4("p_T4"), p_T5("p_T5"), p_T6("p_T6"), p_T7("p_T7"),
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

        long long total_mesh_size = 0;
        for (auto kv : p.field_map)
            total_mesh_size += kv.second->get_size_n();
        std::cout << "Total mesh size = " << total_mesh_size << std::endl;

        double k = 0.1;

        auto p_analy = [=](double x, double y) { return std::exp(-k * (x * x + y * y)); };

        auto f_rhs = [&](double x, double y) { return 4 * k * (k * (x * x + y * y) - 1) * p_analy(x, y); };

        geo.axis(&T1, LocationType::Left);
        geo.axis(&T1, LocationType::Down);

        p.fill_boundary_type(PDEBoundaryType::Dirichlet);
        p.fill_boundary_value_from_func_global(p_analy);

        ConcatPoissonSolver2D solver(&p, env);

        for (int i = 0; i < 100; i++)
        {
            p.set_value_from_func_global(f_rhs);
            solver.solve();
        }

        int         rank_int = static_cast<int>(rank);
        std::string out_base = "result/nine_domain/p_rank_" + std::to_string(rank_int);
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
                    double diff = (*f)(i, j) - p_analy(offx + (0.5 + i) * H, offy + (0.5 + j) * H);
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
        total_l2_sq += calc_err(&p_T6, &T6);
        total_l2_sq += calc_err(&p_T7, &T7);
        total_l2_sq += calc_err(&p_T8, &T8);
        total_l2_sq += calc_err(&p_T9, &T9);

        std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq) << std::endl;
    }
    delete env;
    return 0;
}
