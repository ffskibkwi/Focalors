#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "io/common.h"
#include "io/csv_writer_3d.h"
#include "ns/ns_solver3d.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static double max_abs(const field3& f)
{
    double m = 0.0;
    for (int i = 0; i < f.get_nx(); ++i)
    {
        for (int j = 0; j < f.get_ny(); ++j)
        {
            for (int k = 0; k < f.get_nz(); ++k)
            {
                m = std::max(m, std::abs(f(i, j, k)));
            }
        }
    }
    return m;
}

static void calc_diff_with_two_field_along_x(const field3& src1, const field3& src2, field3& diff)
{
    int nx = std::min({src1.get_nx(), src2.get_nx(), diff.get_nx()});
    int ny = std::min({src1.get_ny(), src2.get_ny(), diff.get_ny()});
    int nz = std::min({src1.get_nz(), src2.get_nz(), diff.get_nz()});
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                diff(i, j, k) = src1(i, j, k) - src2(src2.get_nx() - 1 - i, j, k);
            }
        }
    }
}

static void calc_diff_with_two_field_reversed_along_x(const field3& src1, const field3& src2, field3& diff)
{
    int nx = std::min(src1.get_nx(), src2.get_nx());
    if (nx <= 1)
        return;

    int nx_diff = std::min(diff.get_nx(), nx - 1);
    int ny      = std::min({src1.get_ny(), src2.get_ny(), diff.get_ny()});
    int nz      = std::min({src1.get_nz(), src2.get_nz(), diff.get_nz()});

    for (int i = 0; i < nx_diff; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                diff(i, j, k) = src1(i + 1, j, k) + src2(nx - 1 - i, j, k);
            }
        }
    }
}

static void calc_diff_with_two_field_along_y(const field3& src1, const field3& src2, field3& diff)
{
    int nx = std::min({src1.get_nx(), src2.get_nx(), diff.get_nx()});
    int ny = std::min({src1.get_ny(), src2.get_ny(), diff.get_ny()});
    int nz = std::min({src1.get_nz(), src2.get_nz(), diff.get_nz()});
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                diff(i, j, k) = src1(i, j, k) - src2(i, src2.get_ny() - 1 - j, k);
            }
        }
    }
}

static void calc_diff_with_two_field_reversed_along_y(const field3& src1, const field3& src2, field3& diff)
{
    int ny = std::min(src1.get_ny(), src2.get_ny());
    if (ny <= 1)
        return;

    int nx      = std::min({src1.get_nx(), src2.get_nx(), diff.get_nx()});
    int ny_diff = std::min(diff.get_ny(), ny - 1);
    int nz      = std::min({src1.get_nz(), src2.get_nz(), diff.get_nz()});

    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny_diff; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                diff(i, j, k) = src1(i, j + 1, k) + src2(i, ny - 1 - j, k);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = 0.001;
    time_cfg.num_iterations       = 1;
    time_cfg.corr_iter            = 1;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.nu             = 0.01;

    Domain3DUniform a2(3, 3, 3, 1.0, 1.0, 1.0, "A2");

    Domain3DUniform a1("A1");
    a1.set_nx(3);
    a1.set_lx(1.0);

    Domain3DUniform a3("A3");
    a3.set_nx(3);
    a3.set_lx(1.0);

    Domain3DUniform a4("A4");
    a4.set_ny(3);
    a4.set_ly(1.0);

    Domain3DUniform a5("A5");
    a5.set_ny(3);
    a5.set_ly(1.0);

    geo.connect(&a2, LocationType::Left, &a1);
    geo.connect(&a2, LocationType::Right, &a3);
    geo.connect(&a2, LocationType::Front, &a4);
    geo.connect(&a2, LocationType::Back, &a5);

    geo.axis(&a2, LocationType::Left);
    geo.axis(&a2, LocationType::Front);
    geo.axis(&a2, LocationType::Down);

    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    field3 u_a1, u_a2, u_a3, u_a4, u_a5;
    field3 v_a1, v_a2, v_a3, v_a4, v_a5;
    field3 w_a1, w_a2, w_a3, w_a4, w_a5;
    field3 p_a1, p_a2, p_a3, p_a4, p_a5;

    u.set_x_face_center_field(&a1, u_a1);
    u.set_x_face_center_field(&a2, u_a2);
    u.set_x_face_center_field(&a3, u_a3);
    u.set_x_face_center_field(&a4, u_a4);
    u.set_x_face_center_field(&a5, u_a5);

    v.set_y_face_center_field(&a1, v_a1);
    v.set_y_face_center_field(&a2, v_a2);
    v.set_y_face_center_field(&a3, v_a3);
    v.set_y_face_center_field(&a4, v_a4);
    v.set_y_face_center_field(&a5, v_a5);

    w.set_z_face_center_field(&a1, w_a1);
    w.set_z_face_center_field(&a2, w_a2);
    w.set_z_face_center_field(&a3, w_a3);
    w.set_z_face_center_field(&a4, w_a4);
    w.set_z_face_center_field(&a5, w_a5);

    p.set_center_field(&a1, p_a1);
    p.set_center_field(&a2, p_a2);
    p.set_center_field(&a3, p_a3);
    p.set_center_field(&a4, p_a4);
    p.set_center_field(&a5, p_a5);

    auto set_dirichlet_zero = [](Variable3D& var, Domain3DUniform* domain, LocationType loc) {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(domain, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable3D& var, Domain3DUniform* domain, LocationType loc) {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Neumann);
    };
    auto is_adjacented = [&](Domain3DUniform* domain, LocationType loc) {
        return geo.adjacency.count(domain) && geo.adjacency[domain].count(loc);
    };

    std::vector<Domain3DUniform*> domains = {&a1, &a2, &a3, &a4, &a5};
    std::vector<LocationType>     dirs    = {LocationType::Left,
                                             LocationType::Right,
                                             LocationType::Front,
                                             LocationType::Back,
                                             LocationType::Down,
                                             LocationType::Up};

    for (auto* domain : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(domain, loc))
                continue;

            set_dirichlet_zero(u, domain, loc);
            set_dirichlet_zero(v, domain, loc);
            set_dirichlet_zero(w, domain, loc);
            set_neumann_zero(p, domain, loc);
        }
    }

    const double u0 = 1.0;

    field2* a1_left_bound = u.boundary_value_map[&a1][LocationType::Left];
    if (a1_left_bound == nullptr)
    {
        u.set_boundary_value(&a1, LocationType::Left, 0.0);
        a1_left_bound = u.boundary_value_map[&a1][LocationType::Left];
    }
    for (int j = 0; j < a1_left_bound->get_nx(); ++j)
    {
        for (int k = 0; k < a1_left_bound->get_ny(); ++k)
        {
            double y_norm          = (j + 0.5) / static_cast<double>(a1_left_bound->get_nx());
            double z_norm          = (k + 0.5) / static_cast<double>(a1_left_bound->get_ny());
            double u_val           = 36.0 * u0 * y_norm * (1.0 - y_norm) * z_norm * (1.0 - z_norm);
            (*a1_left_bound)(j, k) = u_val;
        }
    }

    field2* a3_right_bound = u.boundary_value_map[&a3][LocationType::Right];
    if (a3_right_bound == nullptr)
    {
        u.set_boundary_value(&a3, LocationType::Right, 0.0);
        a3_right_bound = u.boundary_value_map[&a3][LocationType::Right];
    }
    for (int j = 0; j < a3_right_bound->get_nx(); ++j)
    {
        for (int k = 0; k < a3_right_bound->get_ny(); ++k)
        {
            double y_norm           = (j + 0.5) / static_cast<double>(a3_right_bound->get_nx());
            double z_norm           = (k + 0.5) / static_cast<double>(a3_right_bound->get_ny());
            double u_val            = -36.0 * u0 * y_norm * (1.0 - y_norm) * z_norm * (1.0 - z_norm);
            (*a3_right_bound)(j, k) = u_val;
        }
    }

    set_neumann_zero(u, &a4, LocationType::Front);
    set_neumann_zero(v, &a4, LocationType::Front);
    set_neumann_zero(w, &a4, LocationType::Front);
    set_neumann_zero(u, &a5, LocationType::Back);
    set_neumann_zero(v, &a5, LocationType::Back);
    set_neumann_zero(w, &a5, LocationType::Back);

    const double center_x = a2.get_offset_x() + 0.5 * a2.get_lx();
    const double center_y = a2.get_offset_y() + 0.5 * a2.get_ly();
    const double center_z = a2.get_offset_z() + 0.5 * a2.get_lz();

    u.set_value_from_func_global([=](double x, double y, double z) {
        double xr = x - center_x;
        double yr = y - center_y;
        double zr = z - center_z;
        return xr * (1.0 + 0.20 * yr * yr + 0.10 * zr * zr);
    });
    v.set_value_from_func_global([=](double x, double y, double z) {
        double xr = x - center_x;
        double yr = y - center_y;
        double zr = z - center_z;
        return -yr * (1.0 + 0.15 * xr * xr + 0.05 * zr * zr);
    });
    w.set_value_from_func_global([=](double x, double y, double z) {
        double xr = x - center_x;
        double yr = y - center_y;
        double zr = z - center_z;
        return 0.25 + 0.10 * xr * xr + 0.08 * yr * yr + 0.03 * zr * zr;
    });
    p.set_value_from_func_global([=](double x, double y, double z) {
        double xr = x - center_x;
        double yr = y - center_y;
        double zr = z - center_z;
        return 0.50 + 0.06 * xr * xr - 0.04 * yr * yr + 0.02 * zr * zr;
    });

    ConcatPoissonSolver3D p_solver(&p);
    ConcatNSSolver3D      ns_solver(&u, &v, &w, &p, &p_solver);
    ns_solver.variable_check();
    ns_solver.phys_boundary_update();
    ns_solver.nondiag_shared_boundary_update();
    ns_solver.diag_shared_boundary_update();

    field3 u_diff_r_1_3(u_a1.get_nx() - 1, u_a1.get_ny(), u_a1.get_nz(), "u_diff_r_1_3");
    field3 v_diff_1_3(v_a1.get_nx(), v_a1.get_ny(), v_a1.get_nz(), "v_diff_1_3");
    field3 w_diff_1_3(w_a1.get_nx(), w_a1.get_ny(), w_a1.get_nz(), "w_diff_1_3");

    field3 u_diff_4_5(u_a4.get_nx(), u_a4.get_ny(), u_a4.get_nz(), "u_diff_4_5");
    field3 v_diff_r_4_5(v_a4.get_nx(), v_a4.get_ny() - 1, v_a4.get_nz(), "v_diff_r_4_5");
    field3 w_diff_4_5(w_a4.get_nx(), w_a4.get_ny(), w_a4.get_nz(), "w_diff_4_5");

    auto print_symmetry = [&](const std::string& stage) {
        calc_diff_with_two_field_reversed_along_x(u_a1, u_a3, u_diff_r_1_3);
        calc_diff_with_two_field_along_x(v_a1, v_a3, v_diff_1_3);
        calc_diff_with_two_field_along_x(w_a1, w_a3, w_diff_1_3);

        calc_diff_with_two_field_along_y(u_a4, u_a5, u_diff_4_5);
        calc_diff_with_two_field_reversed_along_y(v_a4, v_a5, v_diff_r_4_5);
        calc_diff_with_two_field_along_y(w_a4, w_a5, w_diff_4_5);

        std::cout << "\n[NS 3D Symmetry] " << stage << "\n";
        std::cout << "L_inf(u_1 + u_3^Rx) = " << max_abs(u_diff_r_1_3) << "\n";
        std::cout << "L_inf(v_1 - v_3^Rx) = " << max_abs(v_diff_1_3) << "\n";
        std::cout << "L_inf(w_1 - w_3^Rx) = " << max_abs(w_diff_1_3) << "\n";
        std::cout << "L_inf(u_4 - u_5^Ry) = " << max_abs(u_diff_4_5) << "\n";
        std::cout << "L_inf(v_4 + v_5^Ry) = " << max_abs(v_diff_r_4_5) << "\n";
        std::cout << "L_inf(w_4 - w_5^Ry) = " << max_abs(w_diff_4_5) << "\n";
    };

    print_symmetry("before_step");

    std::string nowtime_dir = "result/cross_ns_sym_3d/" + IO::create_timestamp();
    IO::create_directory(nowtime_dir);
    IO::write_csv(u, nowtime_dir + "/u_init");
    IO::write_csv(v, nowtime_dir + "/v_init");
    IO::write_csv(w, nowtime_dir + "/w_init");
    IO::write_csv(p, nowtime_dir + "/p_init");

    ns_solver.euler_conv_diff_inner();
    ns_solver.euler_conv_diff_outer();

    ns_solver.phys_boundary_update();
    ns_solver.nondiag_shared_boundary_update();

    ns_solver.velocity_div_inner();
    ns_solver.velocity_div_outer();
    ns_solver.pressure_buffer_update();
    ns_solver.add_pressure_gradient();

    ns_solver.phys_boundary_update();
    ns_solver.nondiag_shared_boundary_update();
    ns_solver.diag_shared_boundary_update();

    print_symmetry("after_one_step");

    IO::write_csv(u, nowtime_dir + "/u");
    IO::write_csv(v, nowtime_dir + "/v");
    IO::write_csv(w, nowtime_dir + "/w");
    IO::write_csv(p, nowtime_dir + "/p");

    return 0;
}
