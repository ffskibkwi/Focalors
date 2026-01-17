#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "io/config.h"
#include "io/csv_writer_2d.h"
#include "pe/concat/concat_solver2d.h"
#include <cmath>
#include <iostream>
#include <map> // 补齐头文件
#include <vector>

// ---------------------------------------------------------------------
// 物理函数封装 (保持与解析解一致)
// ---------------------------------------------------------------------
void SolvePsiCoeffs(double K1, double K2, double K3, double A[4])
{
    double Kt2 = K1 + K2, Kt3 = K1 + K2 + K3;
    A[1] = M_PI / K1;
    A[2] = (2.0 * M_PI / Kt2 - A[1]) / K2;
    A[3] = (3.0 * M_PI / Kt3 - A[1] - A[2] * (Kt3 - K1)) / ((Kt3 - K1) * (Kt3 - Kt2));
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

// ---------------------------------------------------------------------
// 改进版主程序
// ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // 修正 EnvironmentConfig 构造
    EnvironmentConfig* env = new EnvironmentConfig();
    env->showGmresRes      = true;
    env->showCurrentStep   = false;
    // Enable debug output
    env->debugMode      = true;
    env->debugOutputDir = "./result/debug_output";

    // Create debug directory if it doesn't exist (Platform dependent, but using system command for simplicity or assume
    // user created it) std::filesystem::create_directory("debug_output"); // C++17 feature, using system command is
    // safer if unsure about C++ version support in user env, but here we just setting config.

    std::vector<double> acc_ranks = {4, 8, 16, 32, 64};

    for (double rank : acc_ranks)
    {
        // 1. 几何参数计算
        int    n1 = rank, n2 = 2 * rank, n3 = 4 * rank;
        int    m4 = 2 * rank, m2 = 4 * rank, m5 = rank;
        double H  = 1.0 / (n1 + n2 + n3 + 1.0);
        double L1 = n1 * H, L2 = (n2 + 1) * H, L3 = n3 * H;
        double H4 = m4 * H, H2 = (m2 + 1) * H, H5 = m5 * H;

        // 2. 构造多区域域
        Domain2DUniform T1(n1, m2, "T1");
        Domain2DUniform T2(n2, m2, "T2");
        Domain2DUniform T3(n3, m2, "T3");
        Domain2DUniform T4(n2, m4, "T4");
        Domain2DUniform T5(n2, m5, "T5");

        Geometry2D geo;
        // 修正 add_domains 为 add_domain
        // geo.add_domain({&T1, &T2, &T3, &T4, &T5});

        geo.connect(T2, LocationType::Left, T1);
        geo.connect(T2, LocationType::Right, T3);
        geo.connect(T2, LocationType::Down, T4);
        geo.connect(T2, LocationType::Up, T5);

        geo.set_global_spatial_step(H, H);

        // 3. 变量与场初始化 (修正 Variable 构造)
        Variable p("p");
        p.set_geometry(geo);

        field2 p_T1("p_T1"), p_T2("p_T2"), p_T3("p_T3"), p_T4("p_T4"), p_T5("p_T5");
        p.set_center_field(&T1, p_T1);
        p.set_center_field(&T2, p_T2);
        p.set_center_field(&T3, p_T3);
        p.set_center_field(&T4, p_T4);
        p.set_center_field(&T5, p_T5);

        double Ax[4], Ay[4];
        SolvePsiCoeffs(L1, L2, L3, Ax);
        SolvePsiCoeffs(H4, H2, H5, Ay);

        auto p_analy = [&](double x, double y) {
            return std::sin(ComputePsi(x, L1, L2, Ax, 0)) * std::sin(ComputePsi(y, H4, H2, Ay, 0));
        };

        auto f_rhs = [&](double x, double y) {
            double psx = ComputePsi(x, L1, L2, Ax, 0), psy = ComputePsi(y, H4, H2, Ay, 0);
            double dx = ComputePsi(x, L1, L2, Ax, 1), dy = ComputePsi(y, H4, H2, Ay, 1);
            double ddx = ComputePsi(x, L1, L2, Ax, 2), ddy = ComputePsi(y, H4, H2, Ay, 2);
            double f = (ddx * std::cos(psx) * std::sin(psy) + ddy * std::cos(psy) * std::sin(psx)) -
                       std::sin(psx) * std::sin(psy) * (dx * dx + dy * dy);
            return f * H * H;
        };

        // 定义区域坐标偏移映射，提高可读性
        struct Offset
        {
            double x, y;
        };
        std::map<field2*, Offset> offsets = {{&p_T1, {H, H4 + H}},
                                             {&p_T2, {L1 + H, H4 + H}},
                                             {&p_T3, {L1 + L2, H4 + H}}, // L2 = (N2 + 1) * H
                                             {&p_T4, {L1 + H, H}},
                                             {&p_T5, {L1 + H, H4 + H2}}};

        // 4. 设置边界条件 (修正 API 调用)
        // 4. 设置边界条件 (修正 API 调用)
        p.set_boundary_type(&T1,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Up, PDEBoundaryType::Dirichlet},
                             {LocationType::Down, PDEBoundaryType::Dirichlet}});
        p.set_boundary_type(&T3,
                            {{LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Up, PDEBoundaryType::Dirichlet},
                             {LocationType::Down, PDEBoundaryType::Dirichlet}});
        p.set_boundary_type(&T4,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Down, PDEBoundaryType::Dirichlet}});
        p.set_boundary_type(&T5,
                            {{LocationType::Left, PDEBoundaryType::Dirichlet},
                             {LocationType::Right, PDEBoundaryType::Dirichlet},
                             {LocationType::Up, PDEBoundaryType::Dirichlet}});

        // 5. 右端项填充与求解 (修正 set_values 和 get_center_field)
        auto fill_f = [&](field2& f, double offx, double offy) {
            OPENMP_PARALLEL_FOR()
            for (int i = 0; i < f.get_nx(); ++i)
                for (int j = 0; j < f.get_ny(); ++j)
                    f(i, j) = f_rhs(offx + i * H, offy + j * H);
        };

        for (auto kv : offsets)
            fill_f(*kv.first, kv.second.x, kv.second.y);

        ConcatPoissonSolver2D solver(&p, env);
        solver.solve();

        // 6. 误差统计 (修正 foreach 调用)
        double total_l2_sq = 0.0;
        auto   calc_err    = [&](field2& f, double offx, double offy) {
            double local_sum = 0;
            OPENMP_PARALLEL_FOR(reduction(+ : local_sum))
            for (int i = 0; i < f.get_nx(); ++i)
            {
                for (int j = 0; j < f.get_ny(); ++j)
                {
                    double diff = f(i, j) - p_analy(offx + i * H, offy + j * H);
                    local_sum += H * H * diff * diff;
                }
            }
            return local_sum;
        };

        for (auto kv : offsets)
            total_l2_sq += calc_err(*kv.first, kv.second.x, kv.second.y);

        std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq) << std::endl;

        // std::cout << "=== y = H + L4 (cal)===" << std::endl;
        // for (int i = 0; i < p_T1.get_nx(); i++)
        //     std::cout << p_T1(i, 0) << " ";
        // for (int i = 0; i < p_T2.get_nx(); i++)
        //     std::cout << p_T2(i, 0) << " ";
        // for (int i = 0; i < p_T3.get_nx(); i++)
        //     std::cout << p_T3(i, 0) << " ";
        // std::cout << std::endl << "==================" << std::endl;
        // std::cout << "=== y = H + L4 (analy) ===" << std::endl;
        // for (int i = 0; i < p_T1.get_nx(); i++)
        //     std::cout << p_analy(i * H + offsets[&p_T1].x, offsets[&p_T1].y) << " ";
        // for (int i = 0; i < p_T2.get_nx(); i++)
        //     std::cout << p_analy(i * H + offsets[&p_T2].x, offsets[&p_T2].y) << " ";
        // for (int i = 0; i < p_T3.get_nx(); i++)
        //     std::cout << p_analy(i * H + offsets[&p_T3].x, offsets[&p_T3].y) << " ";
        // std::cout << std::endl << "==================" << std::endl;
    }
    delete env;
    return 0;
}