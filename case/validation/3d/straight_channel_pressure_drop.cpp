#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
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

int main(int argc, char* argv[])
{
    TIMER_BEGIN(Init, "Init", TimeRecordType::None, true);

    if (argc != 2)
    {
        std::cerr << "Error argument! Usage: program Re[double > 0]" << std::endl;
        return 0;
    }

    double channel_height = 1.0;
    double alpha          = 0.5;

    double lx1 = 100 * channel_height;
    double ly1 = 1.0 / alpha * channel_height;
    double lz1 = channel_height;

    int nx1 = 512;
    int ny1 = 64;
    int nz1 = 64;

    double hx = lx1 / nx1;
    double hy = ly1 / ny1;
    double hz = lz1 / nz1;

    double dt = hx / 10.0;

    double Re                = std::stod(argv[1]);
    double density           = 1e3;
    double dynamic_viscosity = 1.01e-3;
    // hydrodynamic_diameter = 2ab/(a+b) = 2aαa/(a+αa) = 2aα/(1+α)
    double hydrodynamic_diameter = 2.0 * channel_height * alpha / (1 + alpha);
    double inlet_velocity        = Re * dynamic_viscosity / (density * hydrodynamic_diameter);
    double kinematic_viscosity   = dynamic_viscosity / density;

    // Geometry: Cross shape
    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.debugOutputDir     = "./result/straight_channel_pressure_drop/Re" + std::to_string(static_cast<int>(Re));

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = dt;
    time_cfg.num_iterations       = 3e4;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_nu(kinematic_viscosity);

    Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");

    geo.add_domain(&A1);

    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);
    geo.axis(&A1, LocationType::ZNegative);

    // Variable2Ds
    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    // Fields on each domain
    field3 u_A1;
    field3 v_A1;
    field3 w_A1;
    field3 p_A1;

    u.set_x_face_center_field(&A1, u_A1);
    v.set_y_face_center_field(&A1, v_A1);
    w.set_z_face_center_field(&A1, w_A1);
    p.set_center_field(&A1, p_A1);

    std::cout << "mesh num = " << u_A1.get_size_n() << std::endl;

    // Helper setters
    auto set_dirichlet_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto isdjacented = [&](Domain3DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain3DUniform*> domains = {&A1};
    std::vector<LocationType>     dirs    = {LocationType::XNegative,
                                             LocationType::XPositive,
                                             LocationType::YNegative,
                                             LocationType::YPositive,
                                             LocationType::ZNegative,
                                             LocationType::ZPositive};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (isdjacented(d, loc))
                continue; // internal boundaries handled automatically
            // velocity: default wall (Dirichlet 0)
            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            set_dirichlet_zero(w, d, loc);
            // pressure: default Neumann (zero gradient)
            set_neumann_zero(p, d, loc);
        }
    }

    // Inlet
    {
        u.has_boundary_value_map[&A1][LocationType::XNegative] = true;

        field2& u_inlet_buffer_xneg = *u.boundary_value_map[&A1][LocationType::XNegative];

        for (int j = 0; j < u_A1.get_ny(); ++j)
        {
            for (int k = 0; k < u_A1.get_nz(); ++k)
            {
                double z = k * hz + 0.5 * hz;
                z /= lz1;
                double vel                = 6.0 * inlet_velocity * (1.0 - z) * z;
                u_inlet_buffer_xneg(j, k) = vel;
            }
        }
    }
    // Outlet
    u.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);

    ConcatPoissonSolver3D p_solver(&p);
    ConcatNSSolver3D      ns_solver(&u, &v, &w, &p, &p_solver);
    PhysicalPESolver3D    ppe_solver(&u, &v, &w, &p, &p_solver, density);

    VTKWriter vtk_writer;
    vtk_writer.add_vector_as_cell_data(&u, &v, &w, "velocity");
    vtk_writer.add_scalar_as_cell_data(&p);
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

        ns_solver.solve();

        if (iter % 200 == 0)
        {
            ppe_solver.solve();

            CSVHandler p_YZ_file(env_cfg.debugOutputDir + "/p_YZ");
            for (int i = 0; i < nx1; i++)
            {
                p_YZ_file.stream << p_A1.mean_at_yz_plane(i);
                if (i != nx1 - 1)
                    p_YZ_file.stream << ',';
                else
                    p_YZ_file.stream << std::endl;
            }
        }

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (false && iter % static_cast<int>(1e4) == 0)
        {
            static int count = 0;
            vtk_writer.write(env_cfg.debugOutputDir + "/vtk/" + std::to_string(count++));
        }

        if (std::isnan(u_A1(0, 0, 0)))
        {
            std::cout << "Error: Find nan! Break solving." << std::endl;
            break;
        }
    }

    std::cout << "Finished" << std::endl;
}
