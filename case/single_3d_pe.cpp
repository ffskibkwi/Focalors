#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"

#include "ns/ns_solver3d.h"

#include "io/config.h"
#include "io/csv_writer_3d.h"

#include "pe/concat/concat_solver3d.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    // Geometry: Cross shape
    Geometry3D geo;

    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes      = true;

    double k = 0.1;

    auto p_analy = [=](double x, double y, double z) { return std::exp(-k * (x * x + y * y + z * z)); };

    auto f_rhs = [&](double x, double y, double z) {
        return (-6.0 * k + 4.0 * k * k * (x * x + y * y + z * z)) * p_analy(x, y, z);
    };

    std::vector<double> acc_ranks = {4, 8, 16, 32, 64};
    for (double rank : acc_ranks)
    {
        int nx1 = rank;
        int ny1 = 2 * rank;
        int nz1 = 3 * rank;

        double H = 1.0 / rank;

        double lx1 = nx1 * H;
        double ly1 = ny1 * H;
        double lz1 = nz1 * H;

        Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");

        geo.add_domain(&A1);

        geo.set_global_spatial_step(H, H, H);

        Variable3D p("p");
        p.set_geometry(geo);

        field3 p_A1;
        p.set_center_field(&A1, p_A1);

        geo.axis(&A1, LocationType::Left);
        geo.axis(&A1, LocationType::Front);
        geo.axis(&A1, LocationType::Down);

        p.fill_boundary_type(PDEBoundaryType::Dirichlet);
        p.fill_boundary_value_from_func_global(p_analy);

        p.set_value_from_func_global(f_rhs);

        ConcatPoissonSolver3D solver(&p, env_config);
        solver.solve();

        int         rank_int = static_cast<int>(rank);
        std::string out_base = "result/p_rank_" + std::to_string(rank_int);
        IO::write_csv(p, out_base);
        // IO::matlab_read_var(p, out_base + "_read.m");

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

        total_l2_sq += calc_err(&p_A1, &A1);

        std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq) << std::endl;
    }
}