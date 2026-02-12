#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"

#include "ns/ns_solver3d.h"

#include "base/config.h"
#include "io/vtk_writer.h"

#include "pe/concat/concat_solver3d.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    TIMER_BEGIN(Init, "Init", TimeRecordType::None, true);

    // Geometry: Cross shape
    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.debugOutputDir     = "./result/backstep_3d";

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = 0.001;
    time_cfg.num_iterations       = 1e5;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(200);

    double lx1 = 0.5;
    double lx2 = 0.5;
    double ly1 = 0.5;
    double lz1 = 0.5;
    double lz3 = 0.5;

    double ly2 = ly1;
    double ly3 = ly1;
    double lz2 = lz1;
    double lx3 = lx2;

    double h = 1e-2;

    int nx1 = lx1 / h;
    int ny1 = ly1 / h;
    int nz1 = lz1 / h;
    int nx2 = lx2 / h;
    int ny2 = ly2 / h;
    int nz2 = lz2 / h;
    int nx3 = lx3 / h;
    int ny3 = ly3 / h;
    int nz3 = lz3 / h;

    Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");
    Domain3DUniform A2(nx2, ny2, nz2, lx2, ly2, lz2, "A2");
    Domain3DUniform A3(nx3, ny3, nz3, lx3, ly3, lz3, "A3");

    geo.add_domain(&A1);
    geo.add_domain(&A2);
    geo.add_domain(&A3);

    // Construct cross connectivity
    geo.connect(&A2, LocationType::Left, &A1);
    geo.connect(&A2, LocationType::Down, &A3);

    geo.axis(&A1, LocationType::Left);
    geo.axis(&A1, LocationType::Front);
    geo.axis(&A1, LocationType::Down);

    // Variable2Ds
    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    // Fields on each domain
    field3 u_A1, u_A2, u_A3;
    field3 v_A1, v_A2, v_A3;
    field3 w_A1, w_A2, w_A3;
    field3 p_A1, p_A2, p_A3;

    u.set_x_face_center_field(&A1, u_A1);
    u.set_x_face_center_field(&A2, u_A2);
    u.set_x_face_center_field(&A3, u_A3);
    v.set_y_face_center_field(&A1, v_A1);
    v.set_y_face_center_field(&A2, v_A2);
    v.set_y_face_center_field(&A3, v_A3);
    w.set_z_face_center_field(&A1, w_A1);
    w.set_z_face_center_field(&A2, w_A2);
    w.set_z_face_center_field(&A3, w_A3);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);

    // Helper setters
    auto set_dirichlet_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto isdjacented = [&](Domain3DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain3DUniform*> domains = {&A1, &A2, &A3};
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
            if (isdjacented(d, loc))
                continue; // internal boundaries handled automatically
            // velocity: default wall (Dirichlet 0)
            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            // pressure: default Neumann (zero gradient)
            set_neumann_zero(p, d, loc);
        }
    }

    // Inlet
    {
        u.has_boundary_value_map[&A1][LocationType::Left] = true;
        field2& u_inlet_buffer                            = *u.boundary_value_map[&A1][LocationType::Left];
        for (int j = 0; j < u_A1.get_ny(); ++j)
        {
            for (int k = 0; k < u_A1.get_nz(); ++k)
            {
                double z             = k * h + 0.5 * h;
                u_inlet_buffer(j, k) = -24.0 * z * z + 12.0 * z;
            }
        }
    }
    // Outlet
    u.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    u.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);

    ConcatNSSolver3D solver(&u, &v, &w, &p);

    VTKWriter vtk_writer;
    vtk_writer.add_vector_as_cell_data(&u, &v, &w, "velocity");
    vtk_writer.validate();

    TIMER_END(Init);

    for (int iter = 0; iter <= time_cfg.num_iterations; iter++)
    {
        SCOPE_TIMER("Iteration", TimeRecordType::None, iter % 100 == 0);

        if (iter % 100 == 0)
        {
            std::cout << "iter: " << iter << "/" << time_cfg.num_iterations << "\n";

            env_cfg.track_pe_solve_detail_time = true;
            env_cfg.showGmresRes               = true;
        }

        solver.solve();

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (iter % 5000 == 0 && iter != 0)
        {
            vtk_writer.write(env_cfg.debugOutputDir + "/" + std::to_string(iter));
        }
    }

    std::cout << "Finished" << std::endl;
}