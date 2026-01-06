#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/geometry_tree.hpp"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/config.h"
#include "io/csv_writer_2d.h"
#include "pe/concat/concat_solver2d.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

// 按照 petsc 没有 +1 来写的版本

void SolvePsiCoeffs(double K1, double K2, double K3, double A[4])
{
    double Kt2 = K1 + K2;
    double Kt3 = K1 + K2 + K3;
    A[1]       = M_PI / K1;
    A[2]       = (2.0 * M_PI / Kt2 - A[1]) / K2;
    A[3]       = (3.0 * M_PI / Kt3 - A[1] - A[2] * (Kt3 - K1)) / ((Kt3 - K1) * (Kt3 - Kt2));
}

double ComputePsi(double t, double K1, double K2, const double A[4], int deriv)
{
    double K12 = K1 + K2;
    if (deriv == 0)
        return t * (A[1] + (A[2] + A[3] * (t - K12)) * (t - K1));
    if (deriv == 1)
        return A[1] + A[2] * (2 * t - K1) + A[3] * (3 * t * t - 2 * t * (K1 + K12) + K1 * K12);
    if (deriv == 2)
        return 2 * A[2] + A[3] * (6 * t - 2 * (K1 + K12));
    return 0;
}

int main(int argc, char* argv[])
{
    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes      = false;
    env_config->showCurrentStep   = false;

    std::vector<double> acc_ranks = {1, 2, 4, 8, 16};
    int                 base_r    = 4;

    for (double acc_rank : acc_ranks)
    {
        int r = static_cast<int>(base_r * acc_rank);

        int nx_T1 = r;
        int nx_T2 = 2 * r;
        int nx_T3 = 4 * r;
        int ny_T4 = 2 * r;
        int ny_T2 = 4 * r;
        int ny_T5 = r;

        int    Nx_total = nx_T1 + nx_T2 + nx_T3;
        double H        = 1.0 / static_cast<double>(Nx_total);

        double L1 = nx_T1 * H;
        double L2 = nx_T2 * H;
        double L3 = nx_T3 * H;
        double H4 = ny_T4 * H;
        double H2 = ny_T2 * H;
        double H5 = ny_T5 * H;

        std::string output_dir = "result/cross_shaped_pe_vali/rank" + std::to_string((int)acc_rank);
        mkdir("result", 0777);
        mkdir("result/cross_shaped_pe_vali", 0777);
        mkdir(output_dir.c_str(), 0777);

        Domain2DUniform T2(nx_T2, ny_T2, L2, H2, "T2");
        Domain2DUniform T1(nx_T1, ny_T2, L1, H2, "T1");
        Domain2DUniform T3(nx_T3, ny_T2, L3, H2, "T3");
        Domain2DUniform T4(nx_T2, ny_T4, L2, H4, "T4");
        Domain2DUniform T5(nx_T2, ny_T5, L2, H5, "T5");

        Geometry2D geo_tee;
        geo_tee.add_domain(T1);
        geo_tee.add_domain(T2);
        geo_tee.add_domain(T3);
        geo_tee.add_domain(T4);
        geo_tee.add_domain(T5);

        geo_tee.connect(T2, LocationType::Left, T1);
        geo_tee.connect(T2, LocationType::Right, T3);
        geo_tee.connect(T2, LocationType::Down, T4);
        geo_tee.connect(T2, LocationType::Up, T5);

        Variable p("p");
        p.set_geometry(geo_tee);
        field2 p_T1("p_T1"), p_T2("p_T2"), p_T3("p_T3"), p_T4("p_T4"), p_T5("p_T5");
        p.set_center_field(&T1, p_T1);
        p.set_center_field(&T2, p_T2);
        p.set_center_field(&T3, p_T3);
        p.set_center_field(&T4, p_T4);
        p.set_center_field(&T5, p_T5);

        p.set_boundary_type(&T1, LocationType::Left, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T1, LocationType::Up, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T1, LocationType::Down, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T3, LocationType::Right, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T3, LocationType::Up, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T3, LocationType::Down, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T4, LocationType::Left, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T4, LocationType::Right, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T4, LocationType::Down, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T5, LocationType::Left, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T5, LocationType::Right, PDEBoundaryType::Dirichlet);
        p.set_boundary_type(&T5, LocationType::Up, PDEBoundaryType::Dirichlet);

        double Ax[4], Ay[4];
        SolvePsiCoeffs(L1, L2, L3, Ax);
        SolvePsiCoeffs(H4, H2, H5, Ay);

        auto p_analy = [&](double x, double y) {
            return std::sin(ComputePsi(x, L1, L2, Ax, 0)) * std::sin(ComputePsi(y, H4, H2, Ay, 0));
        };

        auto f_analy = [&](double x, double y) {
            double psx = ComputePsi(x, L1, L2, Ax, 0), psy = ComputePsi(y, H4, H2, Ay, 0);
            double dx = ComputePsi(x, L1, L2, Ax, 1), dy = ComputePsi(y, H4, H2, Ay, 1);
            double ddx = ComputePsi(x, L1, L2, Ax, 2), ddy = ComputePsi(y, H4, H2, Ay, 2);
            return ((ddx * std::cos(psx) * std::sin(psy) + ddy * std::cos(psy) * std::sin(psx)) -
                    std::sin(psx) * std::sin(psy) * (dx * dx + dy * dy)) *
                   H * H;
        };

        // 填充 RHS
        for (int i = 0; i < p_T1.get_nx(); i++)
            for (int j = 0; j < p_T1.get_ny(); j++)
                p_T1(i, j) = f_analy((i + 0 + 1.0) * H, (j + ny_T4 + 1.0) * H);
        for (int i = 0; i < p_T2.get_nx(); i++)
            for (int j = 0; j < p_T2.get_ny(); j++)
                p_T2(i, j) = f_analy((i + nx_T1 + 1.0) * H, (j + ny_T4 + 1.0) * H);
        for (int i = 0; i < p_T3.get_nx(); i++)
            for (int j = 0; j < p_T3.get_ny(); j++)
                p_T3(i, j) = f_analy((i + nx_T1 + nx_T2 + 1.0) * H, (j + ny_T4 + 1.0) * H);
        for (int i = 0; i < p_T4.get_nx(); i++)
            for (int j = 0; j < p_T4.get_ny(); j++)
                p_T4(i, j) = f_analy((i + nx_T1 + 1.0) * H, (j + 0 + 1.0) * H);
        for (int i = 0; i < p_T5.get_nx(); i++)
            for (int j = 0; j < p_T5.get_ny(); j++)
                p_T5(i, j) = f_analy((i + nx_T1 + 1.0) * H, (j + ny_T4 + ny_T2 + 1.0) * H);

        ConcatPoissonSolver2D solver(&p, env_config);
        solver.solve();

        // 1. 输出 Sol (直接保存求解后的 p 场)
        IO::field_to_csv(p_T1, output_dir + "/p_T1");
        IO::field_to_csv(p_T2, output_dir + "/p_T2");
        IO::field_to_csv(p_T3, output_dir + "/p_T3");
        IO::field_to_csv(p_T4, output_dir + "/p_T4");
        IO::field_to_csv(p_T5, output_dir + "/p_T5");

        // 2. 计算并保存 Diff，同时累计误差
        double total_err_sq = 0;
        auto   compute_diff = [&](field2& f, int oi, int oj, std::string label) {
            for (int i = 0; i < f.get_nx(); i++)
            {
                for (int j = 0; j < f.get_ny(); j++)
                {
                    double exact = p_analy((i + oi + 1.0) * H, (j + oj + 1.0) * H);
                    double diff  = f(i, j) - exact;
                    total_err_sq += diff * diff;
                    f(i, j) = diff; // 覆盖为 diff 场
                }
            }
            IO::field_to_csv(f, output_dir + "/diff_" + label);
        };

        // 临时保存 p_TX 的值用于后续恢复（如果需要），此处直接覆盖
        compute_diff(p_T1, 0, ny_T4, "T1");
        compute_diff(p_T2, nx_T1, ny_T4, "T2");
        compute_diff(p_T3, nx_T1 + nx_T2, ny_T4, "T3");
        compute_diff(p_T4, nx_T1, 0, "T4");
        compute_diff(p_T5, nx_T1, ny_T4 + ny_T2, "T5");

        // 3. 填充并保存 Analy
        for (int i = 0; i < p_T1.get_nx(); i++)
            for (int j = 0; j < p_T1.get_ny(); j++)
                p_T1(i, j) = p_analy((i + 0 + 1.0) * H, (j + ny_T4 + 1.0) * H);
        for (int i = 0; i < p_T2.get_nx(); i++)
            for (int j = 0; j < p_T2.get_ny(); j++)
                p_T2(i, j) = p_analy((i + nx_T1 + 1.0) * H, (j + ny_T4 + 1.0) * H);
        for (int i = 0; i < p_T3.get_nx(); i++)
            for (int j = 0; j < p_T3.get_ny(); j++)
                p_T3(i, j) = p_analy((i + nx_T1 + nx_T2 + 1.0) * H, (j + ny_T4 + 1.0) * H);
        for (int i = 0; i < p_T4.get_nx(); i++)
            for (int j = 0; j < p_T4.get_ny(); j++)
                p_T4(i, j) = p_analy((i + nx_T1 + 1.0) * H, (j + 0 + 1.0) * H);
        for (int i = 0; i < p_T5.get_nx(); i++)
            for (int j = 0; j < p_T5.get_ny(); j++)
                p_T5(i, j) = p_analy((i + nx_T1 + 1.0) * H, (j + ny_T4 + ny_T2 + 1.0) * H);

        IO::field_to_csv(p_T1, output_dir + "/analy_T1");
        IO::field_to_csv(p_T2, output_dir + "/analy_T2");
        IO::field_to_csv(p_T3, output_dir + "/analy_T3");
        IO::field_to_csv(p_T4, output_dir + "/analy_T4");
        IO::field_to_csv(p_T5, output_dir + "/analy_T5");

        int total_nodes =
            p_T1.get_size_n() + p_T2.get_size_n() + p_T3.get_size_n() + p_T4.get_size_n() + p_T5.get_size_n();
        std::cout << "acc_rank = " << acc_rank << " L2 Error = " << std::sqrt(total_err_sq / total_nodes) << std::endl;
    }

    delete env_config;
    return 0;
}