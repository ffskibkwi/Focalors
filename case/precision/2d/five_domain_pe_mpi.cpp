#include "base/config.h"
#include "base/domain/domain2d_mpi.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/parallel/mpi/mpi_misc.h"
#include "io/csv_writer_2d.h"
#include "pe/concat/concat_solver2d_slab_x.h"

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h> // for sleep
#include <vector>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_rank = 0, mpi_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    EnvironmentConfig& env_cfg        = EnvironmentConfig::Get();
    env_cfg.track_pe_solve_total_time = true;

    std::vector<double> acc_ranks = {4};

    // // debug
    // volatile int ii = 0; // use volatile to avoid compiler optimatize out variable
    // while (ii == 0)
    //     sleep(1);

    for (double rank : acc_ranks)
    {
        int    n1 = rank, n2 = 2 * rank, n3 = 4 * rank;
        int    m4 = 2 * rank, m2 = 4 * rank, m5 = rank;
        double H = 1.0 / rank;

        Domain2DUniformMPI T1(n1, m2, "T1");
        Domain2DUniformMPI T2(n2, m2, "T2");
        Domain2DUniformMPI T3(n3, m2, "T3");
        Domain2DUniformMPI T4(n2, m4, "T4");
        Domain2DUniformMPI T5(n2, m5, "T5");

        Geometry2D geo;
        geo.connect(&T2, LocationType::Left, &T1);
        geo.connect(&T2, LocationType::Right, &T3);
        geo.connect(&T2, LocationType::Down, &T4);
        geo.connect(&T2, LocationType::Up, &T5);

        geo.set_global_spatial_step(H, H);

        geo.check();
        geo.solve_prepare();

        Variable2DSlabX p("p");
        p.set_geometry(geo);

        field2 p_T1, p_T2, p_T3, p_T4, p_T5;
        p.set_center_field(&T1, p_T1);
        p.set_center_field(&T2, p_T2);
        p.set_center_field(&T3, p_T3);
        p.set_center_field(&T4, p_T4);
        p.set_center_field(&T5, p_T5);

        long long total_mesh_size = 0;
        for (auto domain : geo.domains)
            total_mesh_size += domain->get_nx() * domain->get_ny();
        std::cout << "Total mesh size = " << total_mesh_size << std::endl;

        double k       = 0.1;
        auto   p_analy = [=](double x, double y) { return std::exp(-k * (x * x + y * y)); };

        auto f_rhs = [&](double x, double y) { return 4 * k * (k * (x * x + y * y) - 1) * p_analy(x, y); };

        geo.axis(&T1, LocationType::Left);
        geo.axis(&T1, LocationType::Down);

        p.fill_boundary_type(PDEBoundaryType::Dirichlet);
        p.fill_boundary_value_from_func_global(p_analy);

        ConcatPoissonSolver2DSlabX solver(&p);

        for (int i = 0; i < 2; i++)
        {
            p.set_value_from_func_global(f_rhs);
            solver.solve();

            if (mpi_rank == 0)
            {
                double total_time = TimerSingleton::Get().GetAcc(env_cfg.pe_solve_total_name);
                std::cout << "[Concat] Solve total (exclude boundary/scale) = " << total_time << "s" << std::endl;
            }
            TimerSingleton::Get().clearAcc();
        }

        int         rank_int = static_cast<int>(rank);
        std::string out_base = "result/five_domain/p_rank_" + std::to_string(rank_int);
        // IO::write_csv(p, out_base);
        // IO::matlab_read_var(p, out_base + "_read.m");

        double total_l2_sq = 0.0;
        for (auto kv : p.field_map)
        {
            Domain2DUniformMPI* domain = static_cast<Domain2DUniformMPI*>(kv.first);

            field2& f = *kv.second;

            double local_sum = 0;
            double offx      = domain->get_offset_x();
            double offy      = domain->get_offset_y();

            int disp = p.hierarchical_slab_disps[p.slab_parent_to_level[domain->get_uuid()]];

            OPENMP_PARALLEL_FOR(reduction(+ : local_sum))
            for (int i = 0; i < f.get_nx(); ++i)
            {
                for (int j = 0; j < f.get_ny(); ++j)
                {
                    double x    = offx + (0.5 + i + disp) * H;
                    double y    = offy + (0.5 + j) * H;
                    double diff = f(i, j) - p_analy(x, y);
                    local_sum += H * H * diff * diff;
                }
            }

            total_l2_sq += local_sum;
        }

        double total_l2_sq_reduced = 0.0;
        MPI_Reduce(&total_l2_sq, &total_l2_sq_reduced, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0)
            std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq_reduced) << std::endl;
    }

    MPI_Finalize();

    return 0;
}