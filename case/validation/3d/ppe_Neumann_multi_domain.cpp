#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "ns/physical_pe_solver3d.h"
#include "pe/concat/concat_solver3d.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/**
 *
 * y
 * ▲
 * │
 * ├──────┬──────┬──────┐
 * │      │      │      │
 * │  A1  │  A2  │  A3  │
 * │      │      │      │
 * ├──────┼──────┼──────┘
 * │      │      │
 * │      │  A4  │
 * │      │      │
 * └──────┴──────┴──────────► x
 *
 */
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Error argument! Usage: program rank[int > 0]" << std::endl;
        return 0;
    }

    int rank = std::stoi(argv[1]);

    Geometry3D geo;

    double lx  = 1.0;
    double ly  = 1.0;
    double lz  = 1.0;
    double rho = 2.0;

    Domain3DUniform A2(rank, rank, rank, lx, ly, lz, "A2");

    Domain3DUniform A1(rank, rank, rank, lx, ly, lz, "A1");
    Domain3DUniform A3(rank, rank, rank, lx, ly, lz, "A3");
    Domain3DUniform A4(rank, rank, rank, lx, ly, lz, "A4");

    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);

    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);
    geo.axis(&A1, LocationType::ZNegative);

    std::vector<Domain3DUniform*> domains = {&A1, &A2, &A3, &A4};

    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    field3 u_A1, u_A2, u_A3, u_A4;
    field3 v_A1, v_A2, v_A3, v_A4;
    field3 w_A1, w_A2, w_A3, w_A4;
    field3 p_A1, p_A2, p_A3, p_A4;

    u.set_x_face_center_field(&A1, u_A1);
    u.set_x_face_center_field(&A2, u_A2);
    u.set_x_face_center_field(&A3, u_A3);
    u.set_x_face_center_field(&A4, u_A4);

    v.set_y_face_center_field(&A1, v_A1);
    v.set_y_face_center_field(&A2, v_A2);
    v.set_y_face_center_field(&A3, v_A3);
    v.set_y_face_center_field(&A4, v_A4);

    w.set_z_face_center_field(&A1, w_A1);
    w.set_z_face_center_field(&A2, w_A2);
    w.set_z_face_center_field(&A3, w_A3);
    w.set_z_face_center_field(&A4, w_A4);

    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);

    const double kx    = pi;
    const double ky    = pi;
    const double kz    = pi;
    const double kx2   = kx * kx;
    const double ky2   = ky * ky;
    const double k2_xy = kx2 + ky2;
    const double kp    = pi;

    auto calc_u = [&](double x, double y, double z) {
        return kx * std::cos(kx * x) * std::sin(ky * y) * std::sin(kz * z);
    };

    auto calc_v = [&](double x, double y, double z) {
        return ky * std::sin(kx * x) * std::cos(ky * y) * std::sin(kz * z);
    };

    auto calc_w = [&](double x, double y, double z) {
        return -(k2_xy / kz) * std::sin(kx * x) * std::sin(ky * y) * std::cos(kz * z);
    };

    auto calc_dudx = [&](double x, double y, double z) {
        return -kx2 * std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);
    };

    auto calc_dudy = [&](double x, double y, double z) {
        return kx * ky * std::cos(kx * x) * std::cos(ky * y) * std::sin(kz * z);
    };

    auto calc_dudz = [&](double x, double y, double z) {
        return kx * kz * std::cos(kx * x) * std::sin(ky * y) * std::cos(kz * z);
    };

    auto calc_dvdx = [&](double x, double y, double z) {
        return kx * ky * std::cos(kx * x) * std::cos(ky * y) * std::sin(kz * z);
    };

    auto calc_dvdy = [&](double x, double y, double z) {
        return -ky2 * std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);
    };

    auto calc_dvdz = [&](double x, double y, double z) {
        return ky * kz * std::sin(kx * x) * std::cos(ky * y) * std::cos(kz * z);
    };

    auto calc_dwdx = [&](double x, double y, double z) {
        return -(kx * k2_xy / kz) * std::cos(kx * x) * std::sin(ky * y) * std::cos(kz * z);
    };

    auto calc_dwdy = [&](double x, double y, double z) {
        return -(ky * k2_xy / kz) * std::sin(kx * x) * std::cos(ky * y) * std::cos(kz * z);
    };

    auto calc_dwdz = [&](double x, double y, double z) {
        return k2_xy * std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);
    };

    auto calc_neg_rho_div_u_grad_u = [&](double x, double y, double z) {
        double s_x = std::sin(kx * x);
        double c_x = std::cos(kx * x);
        double s_y = std::sin(ky * y);
        double c_y = std::cos(ky * y);
        double s_z = std::sin(kz * z);
        double c_z = std::cos(kz * z);

        double L11 = -kx2 * s_x * s_y * s_z;
        double L12 = kx * ky * c_x * c_y * s_z;
        double L13 = kx * kz * c_x * s_y * c_z;

        double L21 = kx * ky * c_x * c_y * s_z;
        double L22 = -ky2 * s_x * s_y * s_z;
        double L23 = ky * kz * s_x * c_y * c_z;

        double L31 = -(kx * k2_xy / kz) * c_x * s_y * c_z;
        double L32 = -(ky * k2_xy / kz) * s_x * c_y * c_z;
        double L33 = k2_xy * s_x * s_y * s_z;

        return -rho * (L11 * L11 + L22 * L22 + L33 * L33 + 2.0 * L12 * L21 + 2.0 * L13 * L31 + 2.0 * L23 * L32);
    };

    auto calc_p = [&](double x, double y, double z) { return std::cos(kp * x) * std::cos(kp * y) * std::cos(kp * z); };

    auto calc_laplacian_p = [&](double x, double y, double z) { return -3.0 * kp * kp * calc_p(x, y, z); };

    auto calc_rho_div_f = [&](double x, double y, double z) {
        return calc_laplacian_p(x, y, z) - calc_neg_rho_div_u_grad_u(x, y, z);
    };

    auto calc_error_centered = [&](const std::string&                                   name,
                                   const std::unordered_map<Domain3DUniform*, field3*>& field_map,
                                   std::function<double(double, double, double)>        f) {
        double error = 0.0;
        size_t size  = 0;

        for (auto* domain : domains)
        {
            field3& target = *field_map.at(domain);

            for (int i = 0; i < target.get_nx(); i++)
            {
                for (int j = 0; j < target.get_ny(); j++)
                {
                    for (int k = 0; k < target.get_nz(); k++)
                    {
                        double x    = domain->get_offset_x() + (i + 0.5) * domain->get_hx();
                        double y    = domain->get_offset_y() + (j + 0.5) * domain->get_hy();
                        double z    = domain->get_offset_z() + (k + 0.5) * domain->get_hz();
                        double diff = target(i, j, k) - f(x, y, z);
                        error += diff * diff;
                        ++size;
                    }
                }
            }
        }

        error = std::sqrt(error / static_cast<double>(size));
        std::cout << "error " << name << " = " << error << std::endl;
    };

    auto add_centered_value = [&](const std::unordered_map<Domain3DUniform*, field3*>& field_map,
                                  std::function<double(double, double, double)>        f) {
        for (auto* domain : domains)
        {
            field3& target = *field_map.at(domain);

            for (int i = 0; i < target.get_nx(); i++)
            {
                for (int j = 0; j < target.get_ny(); j++)
                {
                    for (int k = 0; k < target.get_nz(); k++)
                    {
                        double x = domain->get_offset_x() + (i + 0.5) * domain->get_hx();
                        double y = domain->get_offset_y() + (j + 0.5) * domain->get_hy();
                        double z = domain->get_offset_z() + (k + 0.5) * domain->get_hz();
                        target(i, j, k) += f(x, y, z);
                    }
                }
            }
        }
    };

    auto normalize_pressure = [&]() {
        double sum_num   = 0.0;
        double sum_exact = 0.0;
        size_t sum_size  = 0;

        for (auto* domain : domains)
        {
            field3& p_field = *p.field_map.at(domain);
            sum_num += p_field.sum();
            sum_size += static_cast<size_t>(p_field.get_size_n());

            for (int i = 0; i < p_field.get_nx(); i++)
            {
                for (int j = 0; j < p_field.get_ny(); j++)
                {
                    for (int k = 0; k < p_field.get_nz(); k++)
                    {
                        double x = domain->get_offset_x() + (i + 0.5) * domain->get_hx();
                        double y = domain->get_offset_y() + (j + 0.5) * domain->get_hy();
                        double z = domain->get_offset_z() + (k + 0.5) * domain->get_hz();
                        sum_exact += calc_p(x, y, z);
                    }
                }
            }
        }

        double avg_diff = (sum_num - sum_exact) / static_cast<double>(sum_size);
        for (auto* domain : domains)
            *p.field_map.at(domain) -= avg_diff;
    };

    auto normalize_rhs = [&]() {
        double sum_rhs  = 0.0;
        size_t sum_size = 0;

        for (auto* domain : domains)
        {
            field3& p_field = *p.field_map.at(domain);
            sum_rhs += p_field.sum();
            sum_size += static_cast<size_t>(p_field.get_size_n());
        }

        double avg_rhs = sum_rhs / static_cast<double>(sum_size);
        for (auto* domain : domains)
            *p.field_map.at(domain) -= avg_rhs;
    };

    u.set_boundary_type(PDEBoundaryType::Dirichlet);
    u.set_boundary(calc_u);
    u.set_value(calc_u);
    u.set_corner(calc_u);
    u.set_buffer(calc_u);

    v.set_boundary_type(PDEBoundaryType::Dirichlet);
    v.set_boundary(calc_v);
    v.set_value(calc_v);
    v.set_corner(calc_v);
    v.set_buffer(calc_v);

    w.set_boundary_type(PDEBoundaryType::Dirichlet);
    w.set_boundary(calc_w);
    w.set_value(calc_w);
    w.set_corner(calc_w);
    w.set_buffer(calc_w);

    p.set_boundary_type(PDEBoundaryType::Neumann);
    p.set_buffer([](double, double, double) { return 0.0; });

    ConcatPoissonSolver3D p_solver(&p);
    PhysicalPESolver3D    ppe_solver(&u, &v, &w, &p, &p_solver, rho);

    p.set_value(calc_laplacian_p);
    normalize_rhs();
    p_solver.solve();
    normalize_pressure();
    calc_error_centered("p_from_exact_rhs", p.field_map, calc_p);

    ppe_solver.phys_boundary_update();
    ppe_solver.nondiag_shared_boundary_update();
    ppe_solver.diag_shared_boundary_update();
    ppe_solver.calc_rhs();

    calc_error_centered("rho_div_u_grad_u", p.field_map, calc_neg_rho_div_u_grad_u);

    add_centered_value(p.field_map, calc_rho_div_f);
    calc_error_centered("rhs", p.field_map, calc_laplacian_p);

    normalize_rhs();
    p_solver.solve();
    normalize_pressure();

    calc_error_centered("dudx", ppe_solver.dudx_map, calc_dudx);
    calc_error_centered("dudy", ppe_solver.dudy_map, calc_dudy);
    calc_error_centered("dudz", ppe_solver.dudz_map, calc_dudz);
    calc_error_centered("dvdx", ppe_solver.dvdx_map, calc_dvdx);
    calc_error_centered("dvdy", ppe_solver.dvdy_map, calc_dvdy);
    calc_error_centered("dvdz", ppe_solver.dvdz_map, calc_dvdz);
    calc_error_centered("dwdx", ppe_solver.dwdx_map, calc_dwdx);
    calc_error_centered("dwdy", ppe_solver.dwdy_map, calc_dwdy);
    calc_error_centered("dwdz", ppe_solver.dwdz_map, calc_dwdz);

    double error_laplacian_p = 0.0;
    size_t lap_size          = 0;
    for (auto* domain : domains)
    {
        field3& p_field = *p.field_map.at(domain);
        double  hx      = domain->get_hx();
        double  hy      = domain->get_hy();
        double  hz      = domain->get_hz();

        for (int i = 1; i < p_field.get_nx() - 1; i++)
        {
            for (int j = 1; j < p_field.get_ny() - 1; j++)
            {
                for (int k = 1; k < p_field.get_nz() - 1; k++)
                {
                    double x = domain->get_offset_x() + (i + 0.5) * hx;
                    double y = domain->get_offset_y() + (j + 0.5) * hy;
                    double z = domain->get_offset_z() + (k + 0.5) * hz;

                    double laplacian_p =
                        (p_field(i + 1, j, k) + p_field(i - 1, j, k) - 2.0 * p_field(i, j, k)) / hx / hx +
                        (p_field(i, j + 1, k) + p_field(i, j - 1, k) - 2.0 * p_field(i, j, k)) / hy / hy +
                        (p_field(i, j, k + 1) + p_field(i, j, k - 1) - 2.0 * p_field(i, j, k)) / hz / hz;
                    double diff = laplacian_p - calc_laplacian_p(x, y, z);
                    error_laplacian_p += diff * diff;
                    ++lap_size;
                }
            }
        }
    }

    if (lap_size > 0)
        error_laplacian_p = std::sqrt(error_laplacian_p / static_cast<double>(lap_size));

    std::cout << "error_laplacian_p = " << error_laplacian_p << std::endl;

    calc_error_centered("p_from_sol_rhs", p.field_map, calc_p);

    return 0;
}
