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

int main(int argc, char* argv[])
{
    Geometry2D         geo_tee;
    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes      = true;
    env_config->showCurrentStep   = true;

    // Geometry layout:
    // +-----------+-----------+-----------+
    // |           |           |           |
    // |     Ω1    |     Ω2    |     Ω3    |
    // |           |           |           |
    // +-----------+-----------+-----------+
    // |           |           |           |
    // |           |     Ω4    |           |
    // |           |           |           |
    // |           |           |           |
    // +-----------+-----------+-----------+
    // |           |           |           |
    // |     Ω5    |     Ω6    |     Ω7    |
    // |           |           |           |
    // +-----------+-----------+-----------+
    //
    // 测试不同的dx值
    // 选择能被所有尺寸整除的值: L1=2.0, L2=1.0, L3=3.0, H2=1.5, H4=2.0, H6=1.0
    std::vector<double> dx_values = {0.1, 0.1 / 2, 0.1 / 4, 0.1 / 8, 0.1 / 16}; // 不同的网格间距

    // 参数同 test_pe/pe_analytical_full.m
    const double L1 = 2.0, L2 = 1.0, L3 = 3.0;
    const double H2 = 1.5, H4 = 2.0, H6 = 1.0;
    const double k = 1.0, m = 2.0, n = 3.0;
    const double PI = M_PI;

    auto X_term = [&](double x) {
        return PI * x *
               ((x - L1) * (x - L1 - L2) * (k + m + n) / (L3 * (L2 + L3) * (L1 + L2 + L3)) -
                (k + m) * (x - L1) * (x - L1 - L2 - L3) / (L2 * (L1 + L2) * L3) +
                k * (L1 + L2 - x) * (L1 + L2 + L3 - x) / (L1 * L2 * (L2 + L3)));
    };

    auto Y_term = [&](double y) {
        return PI * y *
               (-((y - H6) * (H4 + H6 - y) * (k + m + n)) / (H2 * (H2 + H4) * (H2 + H4 + H6)) +
                ((y - H6) * (H2 + H4 + H6 - y) * (k + m)) / (H2 * H4 * (H4 + H6)) +
                (k * (H2 + H4 + H6 - y) * (H4 + H6 - y)) / (H4 * (H2 + H4) * H6));
    };

    auto common_X_sin = [&](double x) { return std::sin(X_term(x)); };
    auto common_Y_sin = [&](double y) { return std::sin(Y_term(y)); };

    auto term_X_squared = [&](double x) {
        double t1 = PI * ((k + m + n) * (x - L1) * (x - L1 - L2) / (L3 * (L2 + L3) * (L1 + L2 + L3)) +
                          k * (-x + L1 + L2) * (-x + L1 + L2 + L3) / (L1 * L2 * (L2 + L3)) -
                          (k + m) * (x - L1) * (x - L1 - L2 - L3) / (L2 * (L1 + L2) * L3));
        double t2 =
            PI * x *
            (-(k + m) * (x - L1) / (L2 * (L1 + L2) * L3) + (k + m + n) * (x - L1) / (L3 * (L2 + L3) * (L1 + L2 + L3)) -
             (k + m) * (x - L1 - L2 - L3) / (L2 * (L1 + L2) * L3) - k * (-x + L1 + L2) / (L1 * L2 * (L2 + L3)) -
             k * (-x + L1 + L2 + L3) / (L1 * L2 * (L2 + L3)) +
             (k + m + n) * (x - L1 - L2) / (L3 * (L2 + L3) * (L1 + L2 + L3)));
        double s = t1 + t2;
        return s * s;
    };

    auto term_Y_squared = [&](double y) {
        double t1 = PI * (k * (-y + H2 + H4 + H6) * (-y + H4 + H6) / (H4 * (H2 + H4) * H6) -
                          (k + m + n) * (y - H6) * (-y + H4 + H6) / (H2 * (H2 + H4) * (H2 + H4 + H6)) +
                          (k + m) * (y - H6) * (-y + H2 + H4 + H6) / (H2 * H4 * (H4 + H6)));
        double t2 =
            PI * y *
            (-(k + m) * (y - H6) / (H2 * H4 * (H4 + H6)) + (k + m + n) * (y - H6) / (H2 * (H2 + H4) * (H2 + H4 + H6)) +
             (k + m) * (-y + H2 + H4 + H6) / (H2 * H4 * (H4 + H6)) - k * (-y + H4 + H6) / (H4 * (H2 + H4) * H6) -
             k * (-y + H2 + H4 + H6) / (H4 * (H2 + H4) * H6) -
             (k + m + n) * (-y + H4 + H6) / (H2 * (H2 + H4) * (H2 + H4 + H6)));
        double s = t1 + t2;
        return s * s;
    };

    auto term_X_cos = [&](double x) {
        return 2 * PI *
                   (-(k + m) * (x - L1) / (L2 * (L1 + L2) * L3) +
                    (k + m + n) * (x - L1) / (L3 * (L2 + L3) * (L1 + L2 + L3)) -
                    (k + m) * (x - L1 - L2 - L3) / (L2 * (L1 + L2) * L3) - k * (-x + L1 + L2) / (L1 * L2 * (L2 + L3)) -
                    k * (-x + L1 + L2 + L3) / (L1 * L2 * (L2 + L3)) +
                    (k + m + n) * (x - L1 - L2) / (L3 * (L2 + L3) * (L1 + L2 + L3))) +
               PI * x *
                   (2 * k / (L1 * L2 * (L2 + L3)) - 2 * (k + m) / (L2 * (L1 + L2) * L3) +
                    2 * (k + m + n) / (L3 * (L2 + L3) * (L1 + L2 + L3)));
    };

    auto term_Y_cos = [&](double y) {
        return 2 * PI *
                   (-(k + m) * (y - H6) / (H2 * H4 * (H4 + H6)) +
                    (k + m + n) * (y - H6) / (H2 * (H2 + H4) * (H2 + H4 + H6)) +
                    (k + m) * (-y + H2 + H4 + H6) / (H2 * H4 * (H4 + H6)) - k * (-y + H4 + H6) / (H4 * (H2 + H4) * H6) -
                    k * (-y + H2 + H4 + H6) / (H4 * (H2 + H4) * H6) -
                    (k + m + n) * (-y + H4 + H6) / (H2 * (H2 + H4) * (H2 + H4 + H6))) +
               PI * y *
                   (2 * k / (H4 * (H2 + H4) * H6) - 2 * (k + m) / (H2 * H4 * (H4 + H6)) +
                    2 * (k + m + n) / (H2 * (H2 + H4) * (H2 + H4 + H6)));
    };

    auto Lap_U = [&](double x, double y) {
        return -common_Y_sin(y) * common_X_sin(x) * term_Y_squared(y) -
               common_Y_sin(y) * common_X_sin(x) * term_X_squared(x) +
               std::cos(Y_term(y)) * common_X_sin(x) * term_Y_cos(y) +
               std::cos(X_term(x)) * common_Y_sin(y) * term_X_cos(x);
    };

    auto fill_field = [&](Domain2DUniform& D, field2& F, double x0, double y0) {
        const double dx = D.get_hx();
        const double dy = D.get_hy();
        const int    nx = F.get_nx();
        const int    ny = F.get_ny();
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                const double x = x0 + (i + 0.5) * dx;
                const double y = y0 + (j + 0.5) * dy;
                F(i, j)        = Lap_U(x, y) * dx * dy;
            }
        }
    };

    // 循环测试不同dx值
    for (double dx_val : dx_values)
    {
        // 创建dx命名的目录
        std::ostringstream dir_name;
        dir_name << "result/H-shaped_pe_vali/dx" << dx_val;
        std::string output_dir = dir_name.str();

        // 目录由 IO::field_to_csv 自动创建

        std::cout << "\n========================================" << std::endl;
        std::cout << "测试网格间距: dx=" << dx_val << std::endl;
        std::cout << "输出目录: " << output_dir << std::endl;
        std::cout << "========================================\n" << std::endl;

        // 根据dx创建域
        int nx_T1 = static_cast<int>(L1 / dx_val);
        int nx_T2 = static_cast<int>(L2 / dx_val);
        int nx_T3 = static_cast<int>(L3 / dx_val);
        int ny_T2 = static_cast<int>(H2 / dx_val);
        int ny_T4 = static_cast<int>(H4 / dx_val);
        int ny_T6 = static_cast<int>(H6 / dx_val);

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
        Domain2DUniform T6("T6");
        T6.set_ny(ny_T6);
        T6.set_ly(H6);
        Domain2DUniform T5("T5");
        T5.set_nx(nx_T1);
        T5.set_lx(L1);
        Domain2DUniform T7("T7");
        T7.set_nx(nx_T3);
        T7.set_lx(L3);

        // 构造几何
        Geometry2D geo_tee;
        geo_tee.add_domain(T1);
        geo_tee.add_domain(T2);
        geo_tee.add_domain(T3);
        geo_tee.add_domain(T4);
        geo_tee.add_domain(T5);
        geo_tee.add_domain(T6);
        geo_tee.add_domain(T7);

        geo_tee.connect(T2, LocationType::Left, T1);
        geo_tee.connect(T2, LocationType::Right, T3);
        geo_tee.connect(T2, LocationType::Down, T4);
        geo_tee.connect(T4, LocationType::Down, T6);
        geo_tee.connect(T6, LocationType::Left, T5);
        geo_tee.connect(T6, LocationType::Right, T7);

        // 设置变量和场
        Variable v("v");
        v.set_geometry(geo_tee);
        field2 v_T1("v_T1"), v_T2("v_T2"), v_T3("v_T3"), v_T4("v_T4"), v_T5("v_T5"), v_T6("v_T6"), v_T7("v_T7");
        v.set_center_field(&T1, v_T1);
        v.set_center_field(&T2, v_T2);
        v.set_center_field(&T3, v_T3);
        v.set_center_field(&T4, v_T4);
        v.set_center_field(&T5, v_T5);
        v.set_center_field(&T6, v_T6);
        v.set_center_field(&T7, v_T7);

        // 设置边界条件
        v.set_boundary_type(&T2, LocationType::Up, PDEBoundaryType::Dirichlet);

        v.set_boundary_type(&T1, LocationType::Left, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T1, LocationType::Up, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T1, LocationType::Down, PDEBoundaryType::Dirichlet);

        v.set_boundary_type(&T3, LocationType::Right, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T3, LocationType::Up, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T3, LocationType::Down, PDEBoundaryType::Dirichlet);

        v.set_boundary_type(&T4, LocationType::Left, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T4, LocationType::Right, PDEBoundaryType::Dirichlet);

        v.set_boundary_type(&T5, LocationType::Left, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T5, LocationType::Up, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T5, LocationType::Down, PDEBoundaryType::Dirichlet);

        v.set_boundary_type(&T6, LocationType::Down, PDEBoundaryType::Dirichlet);

        v.set_boundary_type(&T7, LocationType::Right, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T7, LocationType::Up, PDEBoundaryType::Dirichlet);
        v.set_boundary_type(&T7, LocationType::Down, PDEBoundaryType::Dirichlet);

        // 填充右端项
        fill_field(T1, v_T1, 0.0, H6 + H4);
        fill_field(T2, v_T2, L1, H6 + H4);
        fill_field(T3, v_T3, L1 + L2, H6 + H4);
        fill_field(T4, v_T4, L1, H6);
        fill_field(T5, v_T5, 0.0, 0.0);
        fill_field(T6, v_T6, L1, 0.0);
        fill_field(T7, v_T7, L1 + L2, 0.0);

        // 创建求解器并求解
        ConcatPoissonSolver2D solver(&v, env_config);
        solver.solve();

        // 输出结果
        IO::field_to_csv(v_T1, output_dir + "/v_T1");
        IO::field_to_csv(v_T2, output_dir + "/v_T2");
        IO::field_to_csv(v_T3, output_dir + "/v_T3");
        IO::field_to_csv(v_T4, output_dir + "/v_T4");
        IO::field_to_csv(v_T5, output_dir + "/v_T5");
        IO::field_to_csv(v_T6, output_dir + "/v_T6");
        IO::field_to_csv(v_T7, output_dir + "/v_T7");

        std::cout << "结果已保存到: " << output_dir << "\n" << std::endl;
    }

    delete env_config;
    return 0;
}
