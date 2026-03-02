#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "base/math/compare.h"
#include "io/csv_handler.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver3d.h"
#include "ns/physical_pe_solver3d.h"
#include "pe/concat/concat_solver3d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <cmath>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Error argument! Usage: program rank[int > 0]" << std::endl;
        return 0;
    }

    int rank = std::stoi(argv[1]);

    Geometry3D geo;

    double Lx  = 1.0;
    double Ly  = 2.0;
    double Lz  = 3.0;
    double rho = 2.0;

    Domain3DUniform A1(rank, rank, rank, Lx, Ly, Lz, "A1");

    double hx = A1.get_hx();
    double hy = A1.get_hy();
    double hz = A1.get_hz();

    geo.add_domain(&A1);

    geo.axis(&A1, LocationType::Left);
    geo.axis(&A1, LocationType::Front);
    geo.axis(&A1, LocationType::Down);

    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    field3 u_A1;
    field3 v_A1;
    field3 w_A1;
    field3 p_A1;

    u.set_x_face_center_field(&A1, u_A1);
    v.set_y_face_center_field(&A1, v_A1);
    w.set_z_face_center_field(&A1, w_A1);
    p.set_center_field(&A1, p_A1);

    auto calc_u = [&](double x, double y, double z) {
        return (2 * pi * std::cos((2 * pi * x) / Lx) * std::sin((2 * pi * y) / Ly) * std::sin((2 * pi * z) / Lz)) / Lx;
    };

    auto calc_v = [&](double x, double y, double z) {
        return (2 * pi * std::cos((2 * pi * y) / Ly) * std::sin((2 * pi * x) / Lx) * std::sin((2 * pi * z) / Lz)) / Ly;
    };

    auto calc_w = [&](double x, double y, double z) {
        double term1 = (4 * pi * pi) / (Lx * Lx);
        double term2 = (4 * pi * pi) / (Ly * Ly);
        return -(Lz * std::cos((2 * pi * z) / Lz) * std::sin((2 * pi * x) / Lx) * std::sin((2 * pi * y) / Ly) *
                 (term1 + term2)) /
               (2 * pi);
    };

    const double pi2 = pi * pi;
    const double pi3 = pi * pi * pi;

    auto get_common_vars = [&](double x, double y, double z) {
        return std::make_tuple(std::sin((2 * pi * x) / Lx),
                               std::cos((2 * pi * x) / Lx),
                               std::sin((2 * pi * y) / Ly),
                               std::cos((2 * pi * y) / Ly),
                               std::sin((2 * pi * z) / Lz),
                               std::cos((2 * pi * z) / Lz),
                               (4 * pi2) / (Lx * Lx) + (4 * pi2) / (Ly * Ly) // k2_xy
        );
    };

    // --- conv_u 系列 ---
    auto calc_conv_u_x = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return -(8 * pi3 * cosX * sinX * sinY * sinY * sinZ * sinZ) / (Lx * Lx * Lx);
    };

    auto calc_conv_u_y = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return (8 * pi3 * cosX * cosY * cosY * sinX * sinZ * sinZ) / (Lx * Ly * Ly);
    };

    auto calc_conv_u_z = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return -(2 * pi * cosX * cosZ * cosZ * sinX * sinY * sinY * k2_xy) / Lx;
    };

    // --- conv_v 系列 ---
    auto calc_conv_v_x = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return (8 * pi3 * cosX * cosX * cosY * sinY * sinZ * sinZ) / (Lx * Lx * Ly);
    };

    auto calc_conv_v_y = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return -(8 * pi3 * cosY * sinX * sinX * sinY * sinZ * sinZ) / (Ly * Ly * Ly);
    };

    auto calc_conv_v_z = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return -(2 * pi * cosY * cosZ * cosZ * sinX * sinX * sinY * k2_xy) / Ly;
    };

    // --- conv_w 系列 ---
    auto calc_conv_w_x = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return -(2 * Lz * pi * cosX * cosX * cosZ * sinY * sinY * sinZ * k2_xy) / (Lx * Lx);
    };

    auto calc_conv_w_y = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return -(2 * Lz * pi * cosY * cosY * cosZ * sinX * sinX * sinZ * k2_xy) / (Ly * Ly);
    };

    auto calc_conv_w_z = [&](double x, double y, double z) {
        auto [sinX, cosX, sinY, cosY, sinZ, cosZ, k2_xy] = get_common_vars(x, y, z);
        return -(Lz * cosZ * sinX * sinX * sinY * sinY * sinZ * k2_xy * k2_xy) / (2 * pi);
    };

    auto calc_conv_x = [&](double x, double y, double z) {
        double sinX = std::sin((2 * pi * x) / Lx);
        double cosX = std::cos((2 * pi * x) / Lx);
        double sinY = std::sin((2 * pi * y) / Ly);
        double cosY = std::cos((2 * pi * y) / Ly);
        double sinZ = std::sin((2 * pi * z) / Lz);
        double cosZ = std::cos((2 * pi * z) / Lz);

        double pi3   = pi * pi * pi;
        double Lx2   = Lx * Lx;
        double Ly2   = Ly * Ly;
        double k2_xy = (4 * pi * pi) / Lx2 + (4 * pi * pi) / Ly2;

        double term1 = (8 * pi3 * cosX * cosY * cosY * sinX * sinZ * sinZ) / (Lx * Ly2);
        double term2 = (8 * pi3 * cosX * sinX * sinY * sinY * sinZ * sinZ) / (Lx2 * Lx);
        double term3 = (2 * pi * cosX * cosZ * cosZ * sinX * sinY * sinY * k2_xy) / Lx;

        return term1 - term2 - term3;
    };

    auto calc_conv_y = [&](double x, double y, double z) {
        double sinX = std::sin((2 * pi * x) / Lx);
        double cosX = std::cos((2 * pi * x) / Lx);
        double sinY = std::sin((2 * pi * y) / Ly);
        double cosY = std::cos((2 * pi * y) / Ly);
        double sinZ = std::sin((2 * pi * z) / Lz);
        double cosZ = std::cos((2 * pi * z) / Lz);

        double pi3   = pi * pi * pi;
        double Lx2   = Lx * Lx;
        double Ly2   = Ly * Ly;
        double k2_xy = (4 * pi * pi) / Lx2 + (4 * pi * pi) / Ly2;

        double term1 = (8 * pi3 * cosX * cosX * cosY * sinY * sinZ * sinZ) / (Lx2 * Ly);
        double term2 = (8 * pi3 * cosY * sinX * sinX * sinY * sinZ * sinZ) / (Ly2 * Ly);
        double term3 = (2 * pi * cosY * cosZ * cosZ * sinX * sinX * sinY * k2_xy) / Ly;

        return term1 - term2 - term3;
    };

    auto calc_conv_z = [&](double x, double y, double z) {
        double sinX = std::sin((2 * pi * x) / Lx);
        double cosX = std::cos((2 * pi * x) / Lx);
        double sinY = std::sin((2 * pi * y) / Ly);
        double cosY = std::cos((2 * pi * y) / Ly);
        double sinZ = std::sin((2 * pi * z) / Lz);
        double cosZ = std::cos((2 * pi * z) / Lz);

        double pi2   = pi * pi;
        double k2_xy = (4 * pi2) / (Lx * Lx) + (4 * pi2) / (Ly * Ly);

        double term1 = (Lz * cosZ * sinX * sinX * sinY * sinY * sinZ * k2_xy * k2_xy) / (2 * pi);
        double term2 = (2 * Lz * pi * cosX * cosX * cosZ * sinY * sinY * sinZ * k2_xy) / (Lx * Lx);
        double term3 = (2 * Lz * pi * cosY * cosY * cosZ * sinX * sinX * sinZ * k2_xy) / (Ly * Ly);

        return -term1 - term2 - term3;
    };

    auto calc_p = [&](double x, double y, double z) {
        return std::cos((2 * pi * x) / Lx) * std::cos((2 * pi * y) / Ly) * std::cos((2 * pi * z) / Lz);
    };

    auto calc_laplacian_p = [&](double x, double y, double z) {
        double cosX = std::cos((2 * pi * x) / Lx);
        double cosY = std::cos((2 * pi * y) / Ly);
        double cosZ = std::cos((2 * pi * z) / Lz);
        double pi2  = pi * pi;
        double Lx2  = Lx * Lx;
        double Ly2  = Ly * Ly;
        double Lz2  = Lz * Lz;

        return -(4 * pi2 * cosX * cosY * cosZ * (Lx2 * Ly2 + Lx2 * Lz2 + Ly2 * Lz2)) / (Lx2 * Ly2 * Lz2);
    };

    auto calc_rho_div_f = [&](double x, double y, double z) {
        double cosX = std::cos((2 * pi * x) / Lx);
        double cosY = std::cos((2 * pi * y) / Ly);
        double cosZ = std::cos((2 * pi * z) / Lz);

        double cosX2 = cosX * cosX;
        double cosY2 = cosY * cosY;
        double cosZ2 = cosZ * cosZ;

        double Lx2 = Lx * Lx;
        double Lx4 = Lx2 * Lx2;
        double Ly2 = Ly * Ly;
        double Ly4 = Ly2 * Ly2;
        double Lz2 = Lz * Lz;
        double pi2 = pi * pi;
        double pi4 = pi2 * pi2;

        double term1  = 32 * Lx4 * Lz2 * rho * pi4 * cosX2;
        double term2  = 32 * Ly4 * Lz2 * rho * pi4;
        double term3  = 32 * Lx4 * Lz2 * rho * pi4;
        double term4  = 32 * Ly4 * Lz2 * rho * pi4 * cosX2;
        double term5  = 32 * Lx4 * Lz2 * rho * pi4 * cosY2;
        double term6  = 32 * Ly4 * Lz2 * rho * pi4 * cosY2;
        double term7  = 32 * Lx4 * Lz2 * rho * pi4 * cosZ2;
        double term8  = 32 * Ly4 * Lz2 * rho * pi4 * cosZ2;
        double term9  = 32 * Lx2 * Ly2 * Lz2 * rho * pi4;
        double term10 = 32 * Lx4 * Lz2 * rho * pi4 * cosX2 * cosY2;
        double term11 = 32 * Ly4 * Lz2 * rho * pi4 * cosX2 * cosY2;
        double term12 = 32 * Lx4 * Lz2 * rho * pi4 * cosX2 * cosZ2;
        double term13 = 32 * Ly4 * Lz2 * rho * pi4 * cosY2 * cosZ2;
        double term14 = 4 * Lx4 * Ly4 * pi2 * cosX * cosY * cosZ;
        double term15 = 32 * Lx2 * Ly2 * Lz2 * rho * pi4 * cosX2;
        double term16 = 32 * Lx2 * Ly2 * Lz2 * rho * pi4 * cosY2;
        double term17 = 32 * Lx2 * Ly2 * Lz2 * rho * pi4 * cosZ2;
        double term18 = 64 * Lx2 * Ly2 * Lz2 * rho * pi4 * cosX2 * cosY2;
        double term19 = 4 * Lx2 * Ly4 * Lz2 * pi2 * cosX * cosY * cosZ;
        double term20 = 4 * Lx4 * Ly2 * Lz2 * pi2 * cosX * cosY * cosZ;

        double numerator = term1 - term2 - term3 + term4 + term5 + term6 + term7 + term8 - term9 - term10 - term11 -
                           term12 - term13 + term14 + term15 + term16 + term17 - term18 + term19 + term20;

        return numerator / (Lx4 * Ly4 * Lz2);
    };

    auto calc_rho_div_u_grad_u = [&](double x, double y, double z) {
        double sinX = std::sin((2 * pi * x) / Lx);
        double sinY = std::sin((2 * pi * y) / Ly);
        double sinZ = std::sin((2 * pi * z) / Lz);

        double sinX2 = sinX * sinX;
        double sinY2 = sinY * sinY;
        double sinZ2 = sinZ * sinZ;

        double Lx2 = Lx * Lx;
        double Lx4 = Lx2 * Lx2;
        double Ly2 = Ly * Ly;
        double Ly4 = Ly2 * Ly2;
        double pi4 = (pi * pi) * (pi * pi);

        double term1  = Lx4 * sinX2 * sinY2;
        double term2  = Ly4 * sinY2;
        double term3  = Lx4 * sinX2;
        double term4  = Ly4 * sinX2 * sinY2;
        double term5  = Lx4 * sinX2 * sinZ2;
        double term6  = Ly4 * sinY2 * sinZ2;
        double term7  = Lx2 * Ly2 * sinX2;
        double term8  = Lx2 * Ly2 * sinY2;
        double term9  = Lx2 * Ly2 * sinZ2;
        double term10 = 2 * Lx2 * Ly2 * sinX2 * sinY2;

        double numerator =
            32 * rho * pi4 * (term1 - term2 - term3 + term4 + term5 + term6 - term7 - term8 + term9 + term10);

        return numerator / (Lx4 * Ly4);
    };

    u.fill_boundary_type(PDEBoundaryType::Dirichlet);
    u.fill_boundary_value_from_func_global(calc_u);
    u.set_value_from_func_global(calc_u);
    u.fill_corner_value_from_func_global(calc_u);
    u.fill_buffer_value_from_func_global(calc_u);

    v.fill_boundary_type(PDEBoundaryType::Dirichlet);
    v.fill_boundary_value_from_func_global(calc_v);
    v.set_value_from_func_global(calc_v);
    v.fill_corner_value_from_func_global(calc_v);
    v.fill_buffer_value_from_func_global(calc_v);

    w.fill_boundary_type(PDEBoundaryType::Dirichlet);
    w.fill_boundary_value_from_func_global(calc_w);
    w.set_value_from_func_global(calc_w);
    w.fill_corner_value_from_func_global(calc_w);
    w.fill_buffer_value_from_func_global(calc_w);

    p.fill_boundary_type(PDEBoundaryType::Neumann);

    ConcatPoissonSolver3D p_solver(&p);
    PhysicalPESolver3D    ppe_solver(&u, &v, &w, &p, &p_solver, rho);

    // The following pe solve validates that pe solver is correct.

    p.set_value_from_func_global(calc_laplacian_p);

    p_solver.solve();

    {
        double error = 0.0;
        for (int i = 0; i < p_A1.get_nx(); i++)
        {
            for (int j = 0; j < p_A1.get_ny(); j++)
            {
                for (int k = 0; k < p_A1.get_nz(); k++)
                {
                    double x    = (i + 0.5) * hx;
                    double y    = (j + 0.5) * hy;
                    double z    = (k + 0.5) * hz;
                    double diff = p_A1(i, j, k) - calc_p(x, y, z);
                    error += diff * diff;
                }
            }
        }
        error = std::sqrt(error / p_A1.get_size_n());
        std::cout << "error_p_from_exact_rhs = " << error << std::endl;
    }

    ConcatNSSolver3D ns_solver(&u, &v, &w, &p, &p_solver);

    // The following validates that velocity is correct.

    {
        double error = 0.0;
        for (int i = 0; i < u_A1.get_nx(); i++)
        {
            for (int j = 0; j < u_A1.get_ny(); j++)
            {
                for (int k = 0; k < u_A1.get_nz(); k++)
                {
                    double x    = i * hx;
                    double y    = (j + 0.5) * hy;
                    double z    = (k + 0.5) * hz;
                    double diff = u_A1(i, j, k) - calc_u(x, y, z);
                    error += diff * diff;
                }
            }
        }
        field2& u_buffer_right = *u.buffer_map[&A1][LocationType::Right];
        for (int j = 0; j < u_A1.get_ny(); j++)
        {
            for (int k = 0; k < u_A1.get_nz(); k++)
            {
                double x    = u_A1.get_nx() * hx;
                double y    = (j + 0.5) * hy;
                double z    = (k + 0.5) * hz;
                double diff = u_buffer_right(j, k) - calc_u(x, y, z);
                error += diff * diff;
            }
        }
        error = std::sqrt(error / (u_A1.get_nx() + 1) / u_A1.get_ny() / u_A1.get_nz());
        std::cout << "error_u = " << error << std::endl;
    }

    {
        double error = 0.0;
        for (int i = 0; i < v_A1.get_nx(); i++)
        {
            for (int j = 0; j < v_A1.get_ny(); j++)
            {
                for (int k = 0; k < v_A1.get_nz(); k++)
                {
                    double x    = (i + 0.5) * hx;
                    double y    = j * hy;
                    double z    = (k + 0.5) * hz;
                    double diff = v_A1(i, j, k) - calc_v(x, y, z);
                    error += diff * diff;
                }
            }
        }
        field2& v_buffer_back = *v.buffer_map[&A1][LocationType::Back];
        for (int i = 0; i < v_A1.get_nx(); i++)
        {
            for (int k = 0; k < v_A1.get_nz(); k++)
            {
                double x    = (i + 0.5) * hx;
                double y    = v_A1.get_ny() * hy;
                double z    = (k + 0.5) * hz;
                double diff = v_buffer_back(i, k) - calc_v(x, y, z);
                error += diff * diff;
            }
        }
        error = std::sqrt(error / v_A1.get_nx() / (v_A1.get_ny() + 1) / v_A1.get_nz());
        std::cout << "error_v = " << error << std::endl;
    }

    {
        double error = 0.0;
        for (int i = 0; i < w_A1.get_nx(); i++)
        {
            for (int j = 0; j < w_A1.get_ny(); j++)
            {
                for (int k = 0; k < w_A1.get_nz(); k++)
                {
                    double x    = (i + 0.5) * hx;
                    double y    = (j + 0.5) * hy;
                    double z    = k * hz;
                    double diff = w_A1(i, j, k) - calc_w(x, y, z);
                    error += diff * diff;
                }
            }
        }
        field2& w_buffer_up = *w.buffer_map[&A1][LocationType::Up];
        for (int i = 0; i < w_A1.get_nx(); i++)
        {
            for (int j = 0; j < w_A1.get_ny(); j++)
            {
                double x    = (i + 0.5) * hx;
                double y    = (j + 0.5) * hy;
                double z    = w_A1.get_nz() * hz;
                double diff = w_buffer_up(i, j) - calc_w(x, y, z);
                error += diff * diff;
            }
        }
        error = std::sqrt(error / w_A1.get_nx() / w_A1.get_ny() / (w_A1.get_nz() + 1));
        std::cout << "error_w = " << error << std::endl;
    }

    ppe_solver.calc_conv_inner();
    ppe_solver.calc_conv_outer();
    ppe_solver.nondiag_shared_boundary_update();
    ppe_solver.calc_rhs();

    auto calc_error_u = [&](const std::string& name, field3& target, std::function<double(double, double, double)> f) {
        double error = 0.0;
        for (int i = 0; i < target.get_nx(); i++)
        {
            for (int j = 0; j < target.get_ny(); j++)
            {
                for (int k = 0; k < target.get_nz(); k++)
                {
                    double x    = i * hx;
                    double y    = (j + 0.5) * hy;
                    double z    = (k + 0.5) * hz;
                    double diff = target(i, j, k) - f(x, y, z);
                    error += diff * diff;

                    if (!approximatelyEqualAbsRel(target(i, j, k), f(x, y, z), 1e-12, 1e-8))
                    {
                        std::cout << "debugging " << name << " i, j, k = " << i << ", " << j << ", " << k
                                  << " target = " << target(i, j, k) << " f = " << f(x, y, z) << std::endl;
                    }
                }
            }
        }
        error = std::sqrt(error / target.get_size_n());
        std::cout << "error " << name << " = " << error << std::endl;
    };
    auto calc_error_v = [&](const std::string& name, field3& target, std::function<double(double, double, double)> f) {
        double error = 0.0;
        for (int i = 0; i < target.get_nx(); i++)
        {
            for (int j = 0; j < target.get_ny(); j++)
            {
                for (int k = 0; k < target.get_nz(); k++)
                {
                    double x    = (i + 0.5) * hx;
                    double y    = j * hy;
                    double z    = (k + 0.5) * hz;
                    double diff = target(i, j, k) - f(x, y, z);
                    error += diff * diff;
                }
            }
        }
        error = std::sqrt(error / target.get_size_n());
        std::cout << "error " << name << " = " << error << std::endl;
    };
    auto calc_error_w = [&](const std::string& name, field3& target, std::function<double(double, double, double)> f) {
        double error = 0.0;
        for (int i = 0; i < target.get_nx(); i++)
        {
            for (int j = 0; j < target.get_ny(); j++)
            {
                for (int k = 0; k < target.get_nz(); k++)
                {
                    double x    = (i + 0.5) * hx;
                    double y    = (j + 0.5) * hy;
                    double z    = k * hz;
                    double diff = target(i, j, k) - f(x, y, z);
                    error += diff * diff;
                }
            }
        }
        error = std::sqrt(error / target.get_size_n());
        std::cout << "error " << name << " = " << error << std::endl;
    };

    calc_error_u("conv u x", *ppe_solver.conv_u_x_map[&A1], calc_conv_u_x);
    calc_error_u("conv u y", *ppe_solver.conv_u_y_map[&A1], calc_conv_u_y);
    calc_error_u("conv u z", *ppe_solver.conv_u_z_map[&A1], calc_conv_u_z);
    calc_error_u("conv u", *ppe_solver.c_u_map[&A1], calc_conv_x);

    calc_error_v("conv v x", *ppe_solver.conv_v_x_map[&A1], calc_conv_v_x);
    calc_error_v("conv v y", *ppe_solver.conv_v_y_map[&A1], calc_conv_v_y);
    calc_error_v("conv v z", *ppe_solver.conv_v_z_map[&A1], calc_conv_v_z);
    calc_error_v("conv v", *ppe_solver.c_v_map[&A1], calc_conv_y);

    calc_error_w("conv w x", *ppe_solver.conv_w_x_map[&A1], calc_conv_w_x);
    calc_error_w("conv w y", *ppe_solver.conv_w_y_map[&A1], calc_conv_w_y);
    calc_error_w("conv w z", *ppe_solver.conv_w_z_map[&A1], calc_conv_w_z);
    calc_error_w("conv w", *ppe_solver.c_w_map[&A1], calc_conv_z);

    double error_rho_div_u_grad_u = 0.0;
    for (int i = 0; i < p_A1.get_nx(); i++)
    {
        for (int j = 0; j < p_A1.get_ny(); j++)
        {
            for (int k = 0; k < p_A1.get_nz(); k++)
            {
                double x    = (i + 0.5) * hx;
                double y    = (j + 0.5) * hy;
                double z    = (k + 0.5) * hz;
                double diff = p_A1(i, j, k) + calc_rho_div_u_grad_u(x, y, z);
                error_rho_div_u_grad_u += diff * diff;
            }
        }
    }
    error_rho_div_u_grad_u = std::sqrt(error_rho_div_u_grad_u / p_A1.get_size_n());

    for (int i = 0; i < p_A1.get_nx(); i++)
    {
        for (int j = 0; j < p_A1.get_ny(); j++)
        {
            for (int k = 0; k < p_A1.get_nz(); k++)
            {
                double x = (i + 0.5) * hx;
                double y = (j + 0.5) * hy;
                double z = (k + 0.5) * hz;
                p_A1(i, j, k) += calc_rho_div_f(x, y, z);
            }
        }
    }

    p_solver.solve();

    double error_p = 0.0;
    for (int i = 0; i < p_A1.get_nx(); i++)
    {
        for (int j = 0; j < p_A1.get_ny(); j++)
        {
            for (int k = 0; k < p_A1.get_nz(); k++)
            {
                double x    = (i + 0.5) * hx;
                double y    = (j + 0.5) * hy;
                double z    = (k + 0.5) * hz;
                double diff = p_A1(i, j, k) - calc_p(x, y, z);
                error_p += diff * diff;
            }
        }
    }
    error_p = std::sqrt(error_p / p_A1.get_size_n());

    double error_laplacian_p = 0.0;
    for (int i = 1; i < p_A1.get_nx() - 1; i++)
    {
        for (int j = 1; j < p_A1.get_ny() - 1; j++)
        {
            for (int k = 1; k < p_A1.get_nz() - 1; k++)
            {
                double x           = (i + 0.5) * hx;
                double y           = (j + 0.5) * hy;
                double z           = (k + 0.5) * hz;
                double laplacian_p = (p_A1(i + 1, j, k) + p_A1(i - 1, j, k) - 2.0 * p_A1(i, j, k)) / hx / hx +
                                     (p_A1(i, j + 1, k) + p_A1(i, j - 1, k) - 2.0 * p_A1(i, j, k)) / hy / hy +
                                     (p_A1(i, j, k + 1) + p_A1(i, j, k - 1) - 2.0 * p_A1(i, j, k)) / hz / hz;
                double diff = laplacian_p - calc_laplacian_p(x, y, z);
                error_laplacian_p += diff * diff;
            }
        }
    }
    error_laplacian_p = std::sqrt(error_laplacian_p / (p_A1.get_nx() - 1) / (p_A1.get_ny() - 1) / (p_A1.get_nz() - 1));

    std::cout << "error_rho_div_u_grad_u = " << error_rho_div_u_grad_u << std::endl;
    std::cout << "error_p = " << error_p << std::endl;
    std::cout << "error_laplacian_p = " << error_laplacian_p << std::endl;
}