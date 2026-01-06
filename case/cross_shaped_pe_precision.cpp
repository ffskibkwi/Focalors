#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/geometry_tree.hpp"
#include "base/domain/variable.h"
#include "base/field/field2.h"

#include "base/location_boundary.h"

#include "pe/concat/concat_solver2d.h"

#include "io/config.h"
#include "io/csv_writer_2d.h"
#include <cmath>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

/**
 *
 * y
 * ▲
 * │
 * │      ┌──────┐
 * │      │      │
 * │      │  A5  │
 * │      │      │
 * ├──────┼──────┼──────┐
 * │      │      │      │
 * │  A1  │  A2  │  A3  │
 * │      │      │      │
 * ├──────┼──────┼──────┘
 * │      │      │
 * │      │  A4  │
 * │      │      │
 * └──────┴──────┴──────────►
 * O                         x
 *
 */
int main(int argc, char* argv[])
{
    Geometry2D         geo_tee;
    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes      = false;
    env_config->showCurrentStep   = false;

    std::vector<double> acc_ranks = {1, 2, 4, 8, 16};

    double acc_rank   = 1.0;
    int    base_nx_T1 = 4;
    int    base_nx_T2 = 8;
    int    base_nx_T3 = 16;
    int    base_ny_T4 = 8;
    int    base_ny_T2 = 16;
    int    base_ny_T5 = 4;

    // 循环测试不同dx值
    for (double acc_rank : acc_ranks)
    {
        // 创建dx命名的目录
        std::ostringstream dir_name;
        dir_name << "result/cross_shaped_pe_vali/rank" << acc_rank;
        std::string output_dir = dir_name.str();

        // 根据dx创建域
        int nx_T1 = static_cast<int>(std::round(base_nx_T1 * acc_rank));
        int nx_T2 = static_cast<int>(std::round(base_nx_T2 * acc_rank));
        int nx_T3 = static_cast<int>(std::round(base_nx_T3 * acc_rank));
        int ny_T4 = static_cast<int>(std::round(base_ny_T4 * acc_rank));
        int ny_T2 = static_cast<int>(std::round(base_ny_T2 * acc_rank));
        int ny_T5 = static_cast<int>(std::round(base_ny_T5 * acc_rank));
        int ny_T1 = ny_T2;
        int ny_T3 = ny_T2;
        int nx_T4 = nx_T2;
        int nx_T5 = nx_T2;

        double H  = 1. / (nx_T1 + nx_T2 + nx_T3 + 1.);
        double L1 = nx_T1 * H;
        double L2 = (nx_T2 + 1) * H;
        double L3 = nx_T3 * H;
        double H4 = ny_T4 * H;
        double H2 = (ny_T2 + 1) * H;
        double H5 = ny_T5 * H;

        Domain2DUniform T2(nx_T2, ny_T2, L2, H2, "T2"); // 中心
        Domain2DUniform T1("T1");
        T1.set_nx(nx_T1);
        T1.set_lx(L1);
        Domain2DUniform T3("T3");
        T3.set_nx(nx_T3);
        T3.set_lx(L3);
        Domain2DUniform T4("T4");
        T4.set_ny(ny_T4);
        T4.set_ly(H4);
        Domain2DUniform T5("T5");
        T5.set_ny(ny_T5);
        T5.set_ly(H5);

        // 构造几何
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

        // 设置变量和场
        Variable p("p");
        p.set_geometry(geo_tee);
        field2 p_T1("p_T1"), p_T2("p_T2"), p_T3("p_T3"), p_T4("p_T4"), p_T5("p_T5");
        p.set_center_field(&T1, p_T1);
        p.set_center_field(&T2, p_T2);
        p.set_center_field(&T3, p_T3);
        p.set_center_field(&T4, p_T4);
        p.set_center_field(&T5, p_T5);

        // 设置边界条件
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

        int total_p_nodes =
            p_T1.get_size_n() + p_T2.get_size_n() + p_T3.get_size_n() + p_T4.get_size_n() + p_T5.get_size_n();

        auto set_value_global = [&](const std::function<double(double, double)>& op) {
            for (int i = 0; i < p_T1.get_nx(); i++)
            {
                for (int j = 0; j < p_T1.get_ny(); j++)
                {
                    p_T1(i, j) = op((i + 1.) * H, (j + ny_T4 + 1.) * H);
                }
            }

            for (int i = 0; i < p_T2.get_nx(); i++)
            {
                for (int j = 0; j < p_T2.get_ny(); j++)
                {
                    p_T2(i, j) = op((i + nx_T1 + 1.) * H, (j + ny_T4 + 1.) * H);
                }
            }

            for (int i = 0; i < p_T3.get_nx(); i++)
            {
                for (int j = 0; j < p_T3.get_ny(); j++)
                {
                    p_T3(i, j) = op((i + nx_T1 + nx_T2 + 1.) * H, (j + ny_T4 + 1.) * H);
                }
            }

            for (int i = 0; i < p_T4.get_nx(); i++)
            {
                for (int j = 0; j < p_T4.get_ny(); j++)
                {
                    p_T4(i, j) = op((i + nx_T1 + 1.) * H, (j + 1.) * H);
                }
            }

            for (int i = 0; i < p_T5.get_nx(); i++)
            {
                for (int j = 0; j < p_T5.get_ny(); j++)
                {
                    p_T5(i, j) = op((i + nx_T1 + 1.) * H, (j + ny_T4 + ny_T2 + 1.) * H);
                }
            }
        };

        std::function<double(double, double)> f_analy = [&](double x, double y) {
            double A1_x           = M_PI / (2.0 * L1);
            double A2_x           = (M_PI / L2 - A1_x) / (L1 + L2);
            double numerator_A3_x = -((M_PI / L2 + M_PI / (2.0 * L3))) / (L2 + L3) - A2_x;
            double A3_x           = numerator_A3_x / (L1 + L2 + L3);

            double xi2_x    = A2_x + A3_x * (x - L1 - L2);
            double xi1_x    = A1_x + xi2_x * (x - L1);
            double psix_val = x * xi1_x;

            double xi1d_x     = xi2_x + A3_x * (x - L1);
            double dpsix_val  = xi1_x + x * xi1d_x;
            double ddpsix_val = 2.0 * xi1d_x + 2.0 * A3_x * x;

            double A1_y           = M_PI / (2.0 * H4);
            double A2_y           = (M_PI / H2 - A1_y) / (H4 + H2);
            double numerator_A3_y = -((M_PI / H2 + M_PI / (2.0 * H5))) / (H2 + H5) - A2_y;
            double A3_y           = numerator_A3_y / (H4 + H2 + H5);

            double xi2_y    = A2_y + A3_y * (y - H4 - H2);
            double xi1_y    = A1_y + xi2_y * (y - H4);
            double psiy_val = y * xi1_y;

            double xi1d_y     = xi2_y + A3_y * (y - H4);
            double dpsiy_val  = xi1_y + y * xi1d_y;
            double ddpsiy_val = 2.0 * xi1d_y + 2.0 * A3_y * y;

            double p_ana_val   = std::sin(psix_val) * std::sin(psiy_val);
            double pdd_ana_val = ddpsix_val * std::cos(psix_val) * std::sin(psiy_val) +
                                 ddpsiy_val * std::cos(psiy_val) * std::sin(psix_val) -
                                 p_ana_val * (dpsix_val * dpsix_val + dpsiy_val * dpsiy_val);
            return pdd_ana_val * H * H;
        };

        std::function<double(double, double)> p_analy = [&](double x, double y) {
            double A1_x           = M_PI / (2.0 * L1);
            double A2_x           = (M_PI / L2 - A1_x) / (L1 + L2);
            double numerator_A3_x = -((M_PI / L2 + M_PI / (2.0 * L3))) / (L2 + L3) - A2_x;
            double A3_x           = numerator_A3_x / (L1 + L2 + L3);

            double xi2_x    = A2_x + A3_x * (x - L1 - L2);
            double xi1_x    = A1_x + xi2_x * (x - L1);
            double psix_val = x * xi1_x;

            double A1_y           = M_PI / (2.0 * H4);
            double A2_y           = (M_PI / H2 - A1_y) / (H4 + H2);
            double numerator_A3_y = -((M_PI / H2 + M_PI / (2.0 * H5))) / (H2 + H5) - A2_y;
            double A3_y           = numerator_A3_y / (H4 + H2 + H5);

            double xi2_y    = A2_y + A3_y * (y - H4 - H2);
            double xi1_y    = A1_y + xi2_y * (y - H4);
            double psiy_val = y * xi1_y;

            double p_ana_val = std::sin(psix_val) * std::sin(psiy_val);
            return p_ana_val;
        };

        // 填充右端项
        set_value_global(f_analy);

        // 创建求解器并求解
        ConcatPoissonSolver2D solver(&p, env_config);
        solver.solve();

        // 输出结果
        IO::field_to_csv(p_T1, output_dir + "/p_T1");
        IO::field_to_csv(p_T2, output_dir + "/p_T2");
        IO::field_to_csv(p_T3, output_dir + "/p_T3");
        IO::field_to_csv(p_T4, output_dir + "/p_T4");
        IO::field_to_csv(p_T5, output_dir + "/p_T5");

        double sum = 0.0;

        OPENMP_PARALLEL_FOR(reduction(+ : sum))
        for (int i = 0; i < p_T1.get_nx(); i++)
        {
            for (int j = 0; j < p_T1.get_ny(); j++)
            {
                double diff = p_T1(i, j) - p_analy((i + 1.) * H, (j + ny_T4 + 1.) * H);
                sum += diff * diff;
            }
        }

        OPENMP_PARALLEL_FOR(reduction(+ : sum))
        for (int i = 0; i < p_T2.get_nx(); i++)
        {
            for (int j = 0; j < p_T2.get_ny(); j++)
            {
                double diff = p_T2(i, j) - p_analy((i + nx_T1 + 1.) * H, (j + ny_T4 + 1.) * H);
                sum += diff * diff;
            }
        }

        OPENMP_PARALLEL_FOR(reduction(+ : sum))
        for (int i = 0; i < p_T3.get_nx(); i++)
        {
            for (int j = 0; j < p_T3.get_ny(); j++)
            {
                double diff = p_T3(i, j) - p_analy((i + nx_T1 + nx_T2 + 1.) * H, (j + ny_T4 + 1.) * H);
                sum += diff * diff;
            }
        }

        OPENMP_PARALLEL_FOR(reduction(+ : sum))
        for (int i = 0; i < p_T4.get_nx(); i++)
        {
            for (int j = 0; j < p_T4.get_ny(); j++)
            {
                double diff = p_T4(i, j) - p_analy((i + nx_T1 + 1.) * H, (j + 1.) * H);
                sum += diff * diff;
            }
        }

        OPENMP_PARALLEL_FOR(reduction(+ : sum))
        for (int i = 0; i < p_T5.get_nx(); i++)
        {
            for (int j = 0; j < p_T5.get_ny(); j++)
            {
                double diff = p_T5(i, j) - p_analy((i + nx_T1 + 1.) * H, (j + ny_T4 + ny_T2 + 1.) * H);
                sum += diff * diff;
            }
        }

        sum /= total_p_nodes;
        sum = std::sqrt(sum);

        std::cout << "acc_rank = " << acc_rank << " l2 norm = " << sum << '\n';
    }

    delete env_config;
    return 0;
}