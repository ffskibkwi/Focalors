#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "base/math/random.h"
#include "ibm_MirrorPoint/ib_solver_3d_mirror_point.h"
#include "ibm_MirrorPoint/velocity_fixer_3d.h"
#include "ibm_Uhlmann/ib_velocity_solver_3d_Uhlmann.h"
#include "io/case_base.hpp"
#include "io/csv_handler.h"
#include "io/stat.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver3d.h"
#include "ns/scalar_solver3d.h"
#include "particle/particles_coordinate_map_3d.h"
#include "pe/concat/concat_solver3d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

/**
 * y
 * ▲
 * │
 * │
 * │
 * │
 * │
 * ├──────┬──────┬──────┐
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

class TShapedMixerObstacleCase : public CaseBase
{
public:
    TShapedMixerObstacleCase(int argc, char* argv[])
        : CaseBase(argc, argv)
    {}

    void read_paras() override
    {
        CaseBase::read_paras();

        // Geometry parameters
        IO::read_number(para_map, "Height", Height);
        IO::read_number(para_map, "mesh_density", mesh_density);

        // Time stepping
        IO::read_number(para_map, "cfl", cfl);
        IO::read_number(para_map, "pv_output_step", pv_output_step);
        IO::read_number(para_map, "statistics_output_step", statistics_output_step);

        // Physics parameters
        IO::read_number(para_map, "Re", Reynolds_number);
        IO::read_number(para_map, "Sc", Schmidt_number);

        // Obstacle parameters
        IO::read_number(para_map, "sphere_radius_ratio", sphere_radius_ratio);

        // IBM parameters
        IO::read_number(para_map, "ibm_repeat_number", ibm_repeat_number);
        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        // Calculate derived parameters
        lx1 = Height;
        ly1 = Height;
        lz1 = Height;

        hx = Height / mesh_density;
        hy = Height / mesh_density;
        hz = Height / mesh_density;

        nx1 = static_cast<int>(lx1 / hx);
        ny1 = static_cast<int>(ly1 / hy);
        nz1 = static_cast<int>(lz1 / hz);

        dt = cfl * hx;

        Pe = Schmidt_number * Reynolds_number;
        nr = 1.0 / Pe;

        // Sphere parameters
        sphere_radius   = sphere_radius_ratio * Height;
        sphere_center_x = lx1 / 2.0;
        sphere_center_y = ly1 / 2.0;
        sphere_center_z = lz1 / 2.0;

        double mixing_channel_hydraulic_diameter = lx1;
        double density                           = 1e3;
        double dynamic_viscosity                 = 1.01e-3;
        double inlet_velocity  = Reynolds_number * dynamic_viscosity / (density * mixing_channel_hydraulic_diameter);
        double convective_time = mixing_channel_hydraulic_diameter / inlet_velocity;

        paras_record.record("mixing_channel_hydraulic_diameter", mixing_channel_hydraulic_diameter)
            .record("density", density)
            .record("dynamic_viscosity", dynamic_viscosity)
            .record("inlet_velocity", inlet_velocity)
            .record("convective_time", convective_time);

        // Non-dimensionalize
        lx1 /= mixing_channel_hydraulic_diameter;
        ly1 /= mixing_channel_hydraulic_diameter;
        lz1 /= mixing_channel_hydraulic_diameter;

        hx /= mixing_channel_hydraulic_diameter;
        hy /= mixing_channel_hydraulic_diameter;
        hz /= mixing_channel_hydraulic_diameter;

        dt /= convective_time;

        sphere_radius /= mixing_channel_hydraulic_diameter;
        sphere_center_x /= mixing_channel_hydraulic_diameter;
        sphere_center_y /= mixing_channel_hydraulic_diameter;
        sphere_center_z /= mixing_channel_hydraulic_diameter;
    }

    bool record_paras() override
    {
        if (!CaseBase::record_paras())
            return false;

        paras_record.record("Height", Height)
            .record("mesh_density", mesh_density)
            .record("lx1", lx1)
            .record("ly1", ly1)
            .record("lz1", lz1)
            .record("hx", hx)
            .record("hy", hy)
            .record("hz", hz)
            .record("nx1", nx1)
            .record("ny1", ny1)
            .record("nz1", nz1)
            .record("cfl", cfl)
            .record("dt", dt)
            .record("Reynolds_number", Reynolds_number)
            .record("Schmidt_number", Schmidt_number)
            .record("Pe", Pe)
            .record("nr", nr)
            .record("sphere_radius_ratio", sphere_radius_ratio)
            .record("sphere_radius", sphere_radius)
            .record("sphere_center_x", sphere_center_x)
            .record("sphere_center_y", sphere_center_y)
            .record("sphere_center_z", sphere_center_z)
            .record("ibm_repeat_number", ibm_repeat_number)
            .record("gmres_m", gmres_m)
            .record("gmres_tol", gmres_tol)
            .record("gmres_max_iter", gmres_max_iter)
            .record("pv_output_step", pv_output_step)
            .record("statistics_output_step", statistics_output_step);

        return true;
    }

    // Geometry parameters
    double Height       = 1e-3;
    int    mesh_density = 64;

    // Derived geometry
    double lx1, ly1, lz1;
    double hx, hy, hz;
    int    nx1, ny1, nz1;

    // Time stepping
    double cfl                    = 0.1;
    double dt                     = 0.0;
    int    pv_output_step         = 10000;
    int    statistics_output_step = 20;

    // Physics parameters
    double Reynolds_number = 100.0;
    double Schmidt_number  = 5000;
    double Pe              = 0.0;
    double nr              = 0.0;

    // Obstacle parameters
    double sphere_radius_ratio = 1.0;
    double sphere_radius       = 0.0;
    double sphere_center_x     = 0.0;
    double sphere_center_y     = 0.0;
    double sphere_center_z     = 0.0;

    // IBM parameters
    int    ibm_repeat_number = 1;
    int    gmres_m           = 20;
    double gmres_tol         = 1e-6;
    int    gmres_max_iter    = 100;
};

int main(int argc, char* argv[])
{
    TShapedMixerObstacleCase case_param(argc, argv);
    case_param.read_paras();

    // Configuration
    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showGmresRes       = false;
    env_cfg.showCurrentStep    = false;

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.dt;
    time_cfg.num_iterations       = case_param.max_step;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Reynolds_number);

    case_param.record_paras();

    // Geometry: Cross shape
    Geometry3D geo;

    std::cout << "=== T-shaped Mixer with Obstacle ===\n";
    std::cout << "Domain dimensions (non-dim):\n";
    std::cout << "  A1: " << case_param.lx1 << " x " << case_param.ly1 << " x " << case_param.lz1 << "\n";
    std::cout << "Grid: " << case_param.nx1 << "x" << case_param.ny1 << "x" << case_param.nz1 << " (A1)\n";
    std::cout << "Grid spacing: " << case_param.hx << " x " << case_param.hy << " x " << case_param.hz << "\n";
    std::cout << "Sphere: center = (" << case_param.sphere_center_x << ", " << case_param.sphere_center_y << ", "
              << case_param.sphere_center_z << "), radius = " << case_param.sphere_radius << "\n";
    std::cout << "Re = " << case_param.Reynolds_number << ", Sc = " << case_param.Schmidt_number << "\n";
    std::cout << "dt = " << case_param.dt << ", max_step = " << case_param.max_step << "\n\n";

    Domain3DUniform A1(
        case_param.nx1, case_param.ny1, case_param.nz1, case_param.lx1, case_param.ly1, case_param.lz1, "A1");

    geo.add_domain(&A1);

    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);
    geo.axis(&A1, LocationType::ZNegative);

    // Variable3Ds
    Variable3D u("u"), v("v"), w("w"), p("p"), c("concentration");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);
    c.set_geometry(geo);

    // Fields on each domain
    field3 u_A1;
    field3 v_A1;
    field3 w_A1;
    field3 p_A1;
    field3 c_A1;

    u.set_x_face_center_field(&A1, u_A1);
    v.set_y_face_center_field(&A1, v_A1);
    w.set_z_face_center_field(&A1, w_A1);
    p.set_center_field(&A1, p_A1);
    c.set_center_field(&A1, c_A1);

    std::cout << "mesh num = " << u_A1.get_size_n() << std::endl;

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
            set_dirichlet_zero(c, d, loc);
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
                double z = k * case_param.hz + 0.5 * case_param.hz;
                z /= case_param.lz1;
                double vel                = 6.0 * (1.0 - z) * z;
                u_inlet_buffer_xneg(j, k) = vel;
            }
        }

        c.has_boundary_value_map[&A1][LocationType::XNegative] = true;

        field2& c_inlet_buffer_xneg = *c.boundary_value_map[&A1][LocationType::XNegative];

        for (int j = 0; j < c_A1.get_ny(); ++j)
        {
            for (int k = 0; k < c_A1.get_nz(); ++k)
            {
                c_inlet_buffer_xneg(j, k) = 1.0;
            }
        }
    }
    // Outlet
    u.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    c.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);

    add_random_number(u_A1, -0.01, 0.01, 42);

    add_random_number(v_A1, -0.01, 0.01, 42);

    add_random_number(w_A1, -0.01, 0.01, 42);

    DifferenceSchemeType c_scheme = DifferenceSchemeType::Conv_QUICK_Diff_Center2nd;

    ConcatPoissonSolver3D p_solver(&p);
    ConcatNSSolver3D      ns_solver(&u, &v, &w, &p, &p_solver);
    ScalarSolver3D        scalar_solver_c(&u, &v, &w, &c, case_param.nr, c_scheme);

    // IBM setup - Uhlmann velocity solver
    PCoordMap3D coord_map;
    coord_map.add_sphere(case_param.hx,
                         case_param.sphere_radius,
                         case_param.sphere_center_x,
                         case_param.sphere_center_y,
                         case_param.sphere_center_z);
    coord_map.generate_map(&geo);
    auto coord_map_raw = coord_map.get_map();

    IBVelocitySolver3D_Uhlmann ib_solver_vel(&u, &v, &w, coord_map_raw);
    ib_solver_vel.set_parameters(coord_map.get_h(), case_param.hx);

    // Initialize IBM particle velocities to zero (solid sphere)
    for (auto& kv : coord_map_raw)
    {
        auto* p_coord = kv.second;
        auto* ib_data = ib_solver_vel.get_ib_data(kv.first);

        EXPOSE_PCOORD3D(p_coord)
        EXPOSE_PIB3D(ib_data)

        for (int i = 0; i < p_coord->cur_n; i++)
        {
            Up[i] = 0.0;
            Vp[i] = 0.0;
            Wp[i] = 0.0;
        }
    }

    // MirrorPoint concentration solver (Neumann: zero flux)
    Sphere sphere(
        case_param.sphere_center_x, case_param.sphere_center_y, case_param.sphere_center_z, case_param.sphere_radius);
    IBSolver3D_MirrorPoint ib_solver_c(&c, PDEBoundaryType::Neumann, 0.0);
    ib_solver_c.add_shape(&sphere);
    ib_solver_c.build();

    SolidVelocityFixer3D solid_velocity_fixer(&u, &v, &w);
    solid_velocity_fixer.add_shape(&sphere);
    solid_velocity_fixer.build();

    VTKWriter vtk_writer;
    vtk_writer.add_vector_as_cell_data(&u, &v, &w, "velocity");
    vtk_writer.add_scalar_as_cell_data(&c);
    vtk_writer.validate();

    for (int iter = 0; iter <= time_cfg.num_iterations; iter++)
    {
        SCOPE_TIMER("Iteration", TimeRecordType::None, iter % 100 == 0);

        if (iter % 100 == 0)
        {
            std::cout << "iter: " << iter << "/" << time_cfg.num_iterations << "\n";

            env_cfg.track_pe_solve_detail_time = true;
            env_cfg.showGmresRes               = true;
        }
        {
            SCOPE_TIMER("NS euler", TimeRecordType::None, iter % 100 == 0);
            ns_solver.euler_conv_diff_inner();
            ns_solver.euler_conv_diff_outer();
        }
        {
            SCOPE_TIMER("ib_solver_vel", TimeRecordType::None, iter % 100 == 0);
            // Apply Uhlmann velocity IBM solver
            for (int ib_iter = 0; ib_iter < case_param.ibm_repeat_number; ib_iter++)
            {
                ib_solver_vel.solve();
            }
        }
        {
            SCOPE_TIMER("NS bound", TimeRecordType::None, iter % 100 == 0);
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
        }
        {
            SCOPE_TIMER("divu", TimeRecordType::None, iter % 100 == 0);
            // divu
            ns_solver.velocity_div_inner();
            ns_solver.velocity_div_outer();
        }
        {
            SCOPE_TIMER("PE", TimeRecordType::None, iter % 100 == 0);
            // PE
            ns_solver.normalize_pressure();
            p_solver.solve();
        }
        {
            SCOPE_TIMER("p grad", TimeRecordType::None, iter % 100 == 0);
            // update buffer for p
            ns_solver.pressure_buffer_update();

            // p grad
            ns_solver.add_pressure_gradient();
        }
        {
            SCOPE_TIMER("NS bound", TimeRecordType::None, iter % 100 == 0);
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
        }
        // {
        //     SCOPE_TIMER("solid_velocity_fixer", TimeRecordType::None, iter % 100 == 0);
        //     solid_velocity_fixer.apply();
        // }
        {
            SCOPE_TIMER("ib_solver_c", TimeRecordType::None, iter % 100 == 0);
            ib_solver_c.apply();
        }
        {
            SCOPE_TIMER("scalar_solver_c", TimeRecordType::None, iter % 100 == 0);
            scalar_solver_c.solve();
        }

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (iter % static_cast<int>(1e4) == 0)
        {
            static int count = 0;
            vtk_writer.write(case_param.root_dir + "/vtk/" + std::to_string(count++));
        }

        if (iter % case_param.statistics_output_step == 0)
        {
            CSVHandler u_rms_file(case_param.root_dir + "/u_rms");
            u_rms_file.stream << calc_rms(u) << std::endl;

            CSVHandler c_rms_file(case_param.root_dir + "/c_rms");
            c_rms_file.stream << calc_rms(c) << std::endl;
        }

        if (std::isnan(u_A1(0, 0, 0)))
        {
            std::cout << "Error: Find nan at u_A1! Break solving." << std::endl;
            break;
        }

        if (std::isnan(c_A1(0, 0, 0)))
        {
            std::cout << "Error: Find nan at c_A1! Break solving." << std::endl;
            break;
        }
    }

    std::cout << "Finished" << std::endl;
}
