#include "cross_shaped_channel.h"

#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "io/common.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver3d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

/**
 *
 * y
 * ▲
 * │
 * │      ┌──────┐
 * │      │  A5  │
 * │      │      │
 * ├──────┼──────┼──────┐
 * │  A1  │  A2  │  A3  │
 * ├──────┼──────┼──────┘
 * │      │  A4  │
 * │      │      │
 * └──────┴──────┴──────────► x
 * O
 *
 * z direction: depth is shared by all sub-domains (3D extruded cross)
 */
int main(int argc, char* argv[])
{
    TIMER_BEGIN(Init, "Init", TimeRecordType::None, true);

    CrossShapedChannel3DCase case_param(argc, argv);
    case_param.read_paras();
    case_param.record_paras();

    Geometry3D geo;

    double h = case_param.h;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.debugOutputDir     = case_param.root_dir;

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.dt_factor * h;
    time_cfg.set_t_max(case_param.T_total);

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Re);

    // Reserved multi-physics slots (kept synchronized with case parameters)
    case_param.init_nonnewton_config(physics_cfg);
    case_param.init_mhd_config(physics_cfg);

    double lx2 = case_param.lx_2;
    double ly2 = case_param.ly_2;
    double lz2 = case_param.lz_2;

    double lx1 = case_param.lx_1;
    double ly1 = case_param.ly_1;
    double lz1 = case_param.lz_1;

    double lx3 = case_param.lx_3;
    double ly3 = case_param.ly_3;
    double lz3 = case_param.lz_3;

    double lx4 = case_param.lx_4;
    double ly4 = case_param.ly_4;
    double lz4 = case_param.lz_4;

    double lx5 = case_param.lx_5;
    double ly5 = case_param.ly_5;
    double lz5 = case_param.lz_5;

    int nx2 = static_cast<int>(lx2 / h);
    int ny2 = static_cast<int>(ly2 / h);
    int nz2 = static_cast<int>(lz2 / h);

    int nx1 = static_cast<int>(lx1 / h);
    int ny1 = static_cast<int>(ly1 / h);
    int nz1 = static_cast<int>(lz1 / h);

    int nx3 = static_cast<int>(lx3 / h);
    int ny3 = static_cast<int>(ly3 / h);
    int nz3 = static_cast<int>(lz3 / h);

    int nx4 = static_cast<int>(lx4 / h);
    int ny4 = static_cast<int>(ly4 / h);
    int nz4 = static_cast<int>(lz4 / h);

    int nx5 = static_cast<int>(lx5 / h);
    int ny5 = static_cast<int>(ly5 / h);
    int nz5 = static_cast<int>(lz5 / h);

    std::cout << "Construct 3D cross-shaped channel geometry..." << std::endl;
    std::cout << "Domain A2: " << nx2 << " x " << ny2 << " x " << nz2 << " (" << lx2 << " x " << ly2 << " x " << lz2
              << ")" << std::endl;
    std::cout << "Domain A1: " << nx1 << " x " << ny1 << " x " << nz1 << " (" << lx1 << " x " << ly1 << " x " << lz1
              << ")" << std::endl;
    std::cout << "Domain A3: " << nx3 << " x " << ny3 << " x " << nz3 << " (" << lx3 << " x " << ly3 << " x " << lz3
              << ")" << std::endl;
    std::cout << "Domain A4: " << nx4 << " x " << ny4 << " x " << nz4 << " (" << lx4 << " x " << ly4 << " x " << lz4
              << ")" << std::endl;
    std::cout << "Domain A5: " << nx5 << " x " << ny5 << " x " << nz5 << " (" << lx5 << " x " << ly5 << " x " << lz5
              << ")" << std::endl;

    int total_grid = nx1 * ny1 * nz1 + nx2 * ny2 * nz2 + nx3 * ny3 * nz3 + nx4 * ny4 * nz4 + nx5 * ny5 * nz5;
    std::cout << "Total grid points: " << total_grid << std::endl;

    Domain3DUniform A2(nx2, ny2, nz2, lx2, ly2, lz2, "A2");
    Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");
    Domain3DUniform A3(nx3, ny3, nz3, lx3, ly3, lz3, "A3");
    Domain3DUniform A4(nx4, ny4, nz4, lx4, ly4, lz4, "A4");
    Domain3DUniform A5(nx5, ny5, nz5, lx5, ly5, lz5, "A5");

    // Construct cross connectivity
    geo.connect(&A2, case_param.topology.link_a2_a1, &A1);
    geo.connect(&A2, case_param.topology.link_a2_a3, &A3);
    geo.connect(&A2, case_param.topology.link_a2_a4, &A4);
    geo.connect(&A2, case_param.topology.link_a2_a5, &A5);

    geo.axis(&A1, LocationType::Left);
    geo.axis(&A4, LocationType::Front);
    geo.axis(&A2, LocationType::Down);

    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    // Standardized extension data slots for future modules
    [[maybe_unused]] Variable3D mu(case_param.multiphysics_slots.mu_name),
        tau_xx(case_param.multiphysics_slots.stress_names[0]), tau_yy(case_param.multiphysics_slots.stress_names[1]),
        tau_zz(case_param.multiphysics_slots.stress_names[2]), tau_xy(case_param.multiphysics_slots.stress_names[3]),
        tau_xz(case_param.multiphysics_slots.stress_names[4]), tau_yz(case_param.multiphysics_slots.stress_names[5]),
        phi(case_param.multiphysics_slots.phi_name), jx(case_param.multiphysics_slots.current_density_names[0]),
        jy(case_param.multiphysics_slots.current_density_names[1]),
        jz(case_param.multiphysics_slots.current_density_names[2]);
    mu.set_geometry(geo);
    tau_xx.set_geometry(geo);
    tau_yy.set_geometry(geo);
    tau_zz.set_geometry(geo);
    tau_xy.set_geometry(geo);
    tau_xz.set_geometry(geo);
    tau_yz.set_geometry(geo);
    phi.set_geometry(geo);
    jx.set_geometry(geo);
    jy.set_geometry(geo);
    jz.set_geometry(geo);

    // Fields on each domain
    field3 u_A1, u_A2, u_A3, u_A4, u_A5;
    field3 v_A1, v_A2, v_A3, v_A4, v_A5;
    field3 w_A1, w_A2, w_A3, w_A4, w_A5;
    field3 p_A1, p_A2, p_A3, p_A4, p_A5;

    u.set_x_face_center_field(&A1, u_A1);
    u.set_x_face_center_field(&A2, u_A2);
    u.set_x_face_center_field(&A3, u_A3);
    u.set_x_face_center_field(&A4, u_A4);
    u.set_x_face_center_field(&A5, u_A5);

    v.set_y_face_center_field(&A1, v_A1);
    v.set_y_face_center_field(&A2, v_A2);
    v.set_y_face_center_field(&A3, v_A3);
    v.set_y_face_center_field(&A4, v_A4);
    v.set_y_face_center_field(&A5, v_A5);

    w.set_z_face_center_field(&A1, w_A1);
    w.set_z_face_center_field(&A2, w_A2);
    w.set_z_face_center_field(&A3, w_A3);
    w.set_z_face_center_field(&A4, w_A4);
    w.set_z_face_center_field(&A5, w_A5);

    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);
    p.set_center_field(&A5, p_A5);

    auto set_dirichlet_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto is_adjacented = [&](Domain3DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };

    std::vector<Domain3DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};
    std::vector<LocationType>     dirs    = {LocationType::Left,
                                             LocationType::Right,
                                             LocationType::Front,
                                             LocationType::Back,
                                             LocationType::Down,
                                             LocationType::Up};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(d, loc))
                continue;
            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            set_dirichlet_zero(w, d, loc);
            set_neumann_zero(p, d, loc);
        }
    }

    // Inlets: A1 Left and A3 Right (opposite Poiseuille profiles)
    u.set_boundary_type(&A1, LocationType::Left, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A1, LocationType::Left, 0.0);
    u.has_boundary_value_map[&A1][LocationType::Left] = true;
    set_dirichlet_zero(v, &A1, LocationType::Left);
    set_dirichlet_zero(w, &A1, LocationType::Left);

    for (int j = 0; j < u_A1.get_ny(); ++j)
    {
        for (int k = 0; k < u_A1.get_nz(); ++k)
        {
            double y_norm = (j + 0.5) / static_cast<double>(u_A1.get_ny());
            double z_norm = (k + 0.5) / static_cast<double>(u_A1.get_nz());
            double u_val  = 36.0 * case_param.U0 * y_norm * (1.0 - y_norm) * z_norm * (1.0 - z_norm);
            (*u.boundary_value_map[&A1][LocationType::Left])(j, k) = u_val;
        }
    }

    u.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A3, LocationType::Right, 0.0);
    u.has_boundary_value_map[&A3][LocationType::Right] = true;
    set_dirichlet_zero(v, &A3, LocationType::Right);
    set_dirichlet_zero(w, &A3, LocationType::Right);

    for (int j = 0; j < u_A3.get_ny(); ++j)
    {
        for (int k = 0; k < u_A3.get_nz(); ++k)
        {
            double y_norm = (j + 0.5) / static_cast<double>(u_A3.get_ny());
            double z_norm = (k + 0.5) / static_cast<double>(u_A3.get_nz());
            double u_val  = -36.0 * case_param.U0 * y_norm * (1.0 - y_norm) * z_norm * (1.0 - z_norm);
            (*u.boundary_value_map[&A3][LocationType::Right])(j, k) = u_val;
        }
    }

    // Outlets: A4 and A5 (open/symmetry for velocity)
    set_neumann_zero(u, &A4, case_param.topology.outlet_a4_loc);
    set_neumann_zero(v, &A4, case_param.topology.outlet_a4_loc);
    set_neumann_zero(w, &A4, case_param.topology.outlet_a4_loc);
    set_neumann_zero(u, &A5, case_param.topology.outlet_a5_loc);
    set_neumann_zero(v, &A5, case_param.topology.outlet_a5_loc);
    set_neumann_zero(w, &A5, case_param.topology.outlet_a5_loc);

    // Unified outlet pressure: Dirichlet p = 0
    set_dirichlet_zero(p, &A4, case_param.topology.outlet_a4_loc);
    set_dirichlet_zero(p, &A5, case_param.topology.outlet_a5_loc);

    ConcatPoissonSolver3D p_solver(&p);
    ConcatNSSolver3D      solver(&u, &v, &w, &p, &p_solver);
    solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

    VTKWriter vtk_writer;
    vtk_writer.add_vector_as_cell_data(&u, &v, &w, "velocity");
    vtk_writer.add_scalar_as_cell_data(&p);
    vtk_writer.validate();

    IO::create_directory(env_cfg.debugOutputDir + "/vtk");

    TIMER_END(Init);

    int vtk_step = case_param.vtk_output_step > 0 ? case_param.vtk_output_step : 200;

    for (int step = 1; step <= time_cfg.num_iterations; ++step)
    {
        if (step % 200 == 0)
        {
            env_cfg.showGmresRes = true;
            std::cout << "step: " << step << "/" << time_cfg.num_iterations << "\n";
        }
        else
        {
            env_cfg.showGmresRes = (step <= 5);
        }

        {
            Timer step_timer("step_time", TimeRecordType::None, step % 200 == 0);
            solver.solve();
        }

        if (step % vtk_step == 0)
        {
            vtk_writer.write(env_cfg.debugOutputDir + "/vtk/step_" + std::to_string(step));
        }

        if (std::isnan(u_A1(1, 1, 1)))
        {
            std::cout << "=== DIVERGENCE ===" << std::endl;
            return -1;
        }
    }

    vtk_writer.write(env_cfg.debugOutputDir + "/vtk/final");
    std::cout << "Simulation finished." << std::endl;
    return 0;
}
