#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "pe/concat/concat_solver2d.h"
#include "io/csv_writer_2d.h"
#include <cmath>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------
// 物理函数封装 (保持与解析解一致)
// ---------------------------------------------------------------------
void SolvePsiCoeffs(double K1, double K2, double K3, double A[4]) {
    double Kt2 = K1 + K2, Kt3 = K1 + K2 + K3;
    A[1] = M_PI / K1;
    A[2] = (2.0 * M_PI / Kt2 - A[1]) / K2;
    A[3] = (3.0 * M_PI / Kt3 - A[1] - A[2] * (Kt3 - K1)) / ((Kt3 - K1) * (Kt3 - Kt2));
}

double ComputePsi(double t, double K1, double K2, const double A[4], int deriv) {
    double K12 = K1 + K2;
    if (deriv == 0) return t * (A[1] + (A[2] + A[3] * (t - K12)) * (t - K1));
    if (deriv == 1) return A[1] + A[2] * (2 * t - K1) + A[3] * (3 * t * t - 2 * t * (K1 + K12) + K1 * K12);
    if (deriv == 2) return 2 * A[2] + A[3] * (6 * t - 2 * (K1 + K12));
    return 0;
}

// ---------------------------------------------------------------------
// 改进版主程序
// ---------------------------------------------------------------------
int main(int argc, char* argv[]) {
    EnvironmentConfig env(false, false);
    std::vector<double> acc_ranks = {1, 2, 4, 8, 16};

    for (double rank : acc_ranks) {
        // 1. 几何参数计算
        int n1 = 4 * rank, n2 = 8 * rank, n3 = 16 * rank;
        int m4 = 8 * rank, m2 = 16 * rank, m5 = 4 * rank;
        double H = 1.0 / (n1 + n2 + n3 + 1.0);
        double L1 = n1 * H, L2 = (n2 + 1) * H, L3 = n3 * H;
        double H4 = m4 * H, H2 = (m2 + 1) * H, H5 = m5 * H;

        // 2. 构造多区域域
        Domain2DUniform T1(n1, m2, L1, H2, "T1");
        Domain2DUniform T2(n2, m2, L2, H2, "T2");
        Domain2DUniform T3(n3, m2, L3, H2, "T3");
        Domain2DUniform T4(n2, m4, L2, H4, "T4");
        Domain2DUniform T5(n2, m5, L2, H5, "T5");

        Geometry2D geo;
        geo.add_domains({&T1, &T2, &T3, &T4, &T5});
        geo.connect(T2, LocationType::Left,  T1);
        geo.connect(T2, LocationType::Right, T3);
        geo.connect(T2, LocationType::Down,  T4);
        geo.connect(T2, LocationType::Up,    T5);

        // 3. 变量与场初始化
        Variable p("p", geo);
        double Ax[4], Ay[4];
        SolvePsiCoeffs(L1, L2, L3, Ax);
        SolvePsiCoeffs(H4, H2, H5, Ay);

        auto p_analy = [&](double x, double y) {
            return std::sin(ComputePsi(x, L1, L2, Ax, 0)) * std::sin(ComputePsi(y, H4, H2, Ay, 0));
        };

        auto f_rhs = [&](double x, double y) {
            double psx = ComputePsi(x, L1, L2, Ax, 0), psy = ComputePsi(y, H4, H2, Ay, 0);
            double dx  = ComputePsi(x, L1, L2, Ax, 1),  dy = ComputePsi(y, H4, H2, Ay, 1);
            double ddx = ComputePsi(x, L1, L2, Ax, 2), ddy = ComputePsi(y, H4, H2, Ay, 2);
            double f = (ddx * std::cos(psx) * std::sin(psy) + ddy * std::cos(psy) * std::sin(psx)) -
                       std::sin(psx) * std::sin(psy) * (dx * dx + dy * dy);
            return f * H * H;
        };

        // 4. 设置边界与初值 (使用高层访客简化代码)
        p.set_all_boundaries(PDEBoundaryType::Dirichlet); // 假设 API 支持批量设置

        // 定义区域坐标偏移映射，提高可读性
        struct Offset { double x, y; };
        std::map<std::string, Offset> offsets = {
            {"T1", {0, H4 + H}}, {"T2", {L1 + H, H4 + H}}, {"T3", {L1 + L2 + H, H4 + H}},
            {"T4", {L1 + H, 0}}, {"T5", {L1 + H, H4 + H2 + H}}
        };

        // 5. 右端项填充与求解
        for (auto* dom : geo.get_domains()) {
            field2& f = p.get_center_field(dom);
            Offset off = offsets[dom->get_name()];
            f.set_values([&](int i, int j) { return f_rhs(off.x + i * H, off.y + j * H); });
        }

        ConcatPoissonSolver2D solver(&p, &env);
        solver.solve();

        // 6. 误差统计 (修正 L2 范数逻辑)
        double total_l2_sq = 0.0;
        for (auto* dom : geo.get_domains()) {
            field2& pf = p.get_center_field(dom);
            Offset off = offsets[dom->get_name()];
            pf.foreach([&](int i, int j, double& val) {
                double diff = val - p_analy(off.x + i * H, off.y + j * H);
                total_l2_sq += H * H * diff * diff;
            });
        }
        std::cout << "rank: " << rank << " L2 Error: " << std::sqrt(total_l2_sq) << std::endl;
    }
    return 0;
}