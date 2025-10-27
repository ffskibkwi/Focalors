#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/geometry_tree.hpp"
#include "base/domain/variable.h"
#include "base/field/field2.h"

#include "base/location_boundary.h"

#include "ns/ns_solver2d.h"
#include "pe/concat/concat_solver2d.h"

#include "io/config.h"
#include "io/csv_writer_2d.h"

int main(int argc, char* argv[])
{
    Geometry2D         geo_tee;
    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes      = true;
    env_config->showCurrentStep   = true;

    TimeAdvancingConfig* time_config = new TimeAdvancingConfig();
    time_config->dt                  = 0.01;
    time_config->num_iterations      = 100;

    PhysicsConfig* physics_config = new PhysicsConfig();
    physics_config->nu            = 0.01;

    // clang-format off
    // Grid parameters
    Domain2DUniform T2(10, 10, 1.0, 1.0, "T2"); // 中心
    Domain2DUniform T1("T1"); T1.set_nx(20); T1.set_lx(2.0);
    Domain2DUniform T3("T3"); T3.set_nx(30); T3.set_lx(3.0);
    Domain2DUniform T4("T4"); T4.set_ny(10); T4.set_ly(1.0);
    Domain2DUniform T5("T5"); T5.set_ny(20); T5.set_ly(2.0);
    Domain2DUniform T6("T6"); T6.set_nx(30); T6.set_lx(3.0);
    // clang-format on

    // Set boundary on variable (new logic)

    // Construct geometry
    geo_tee.add_domain(T1);
    geo_tee.add_domain(T2);
    geo_tee.add_domain(T3);
    geo_tee.add_domain(T4);
    geo_tee.add_domain(T5);
    geo_tee.add_domain(T6);

    // clang-format off
    geo_tee.connect(T2, LocationType::Left,  T1);
    geo_tee.connect(T2, LocationType::Right, T3);
    geo_tee.connect(T2, LocationType::Down,  T4);
    geo_tee.connect(T4, LocationType::Down,  T5);
    geo_tee.connect(T5, LocationType::Right, T6);
    // clang-format on

    Variable u("u");
    u.set_geometry(geo_tee);
    field2 u_T1("u_T1"), u_T2("u_T2"), u_T3("u_T3"), u_T4("u_T4"), u_T5("u_T5"), u_T6("u_T6");
    u.set_y_edge_field(&T1, u_T1);
    u.set_y_edge_field(&T2, u_T2);
    u.set_y_edge_field(&T3, u_T3);
    u.set_y_edge_field(&T4, u_T4);
    u.set_y_edge_field(&T5, u_T5);
    u.set_y_edge_field(&T6, u_T6);

    // clang-format off
    u.set_boundary_type(&T2, LocationType::Up,    PDEBoundaryType::Dirichlet);

    u.set_boundary_type(&T1, LocationType::Left,  PDEBoundaryType::Dirichlet);
    u.set_boundary_type(&T1, LocationType::Up,    PDEBoundaryType::Dirichlet);
    u.set_boundary_type(&T1, LocationType::Down,  PDEBoundaryType::Dirichlet);

    u.set_boundary_type(&T3, LocationType::Right, PDEBoundaryType::Dirichlet);
    u.set_boundary_type(&T3, LocationType::Up,    PDEBoundaryType::Dirichlet);
    u.set_boundary_type(&T3, LocationType::Down,  PDEBoundaryType::Dirichlet);

    u.set_boundary_type(&T4, LocationType::Left,  PDEBoundaryType::Dirichlet);
    u.set_boundary_type(&T4, LocationType::Right, PDEBoundaryType::Dirichlet);

    u.set_boundary_type(&T5, LocationType::Left,  PDEBoundaryType::Dirichlet);
    u.set_boundary_type(&T5, LocationType::Down,  PDEBoundaryType::Dirichlet);

    u.set_boundary_type(&T6, LocationType::Right, PDEBoundaryType::Neumann);
    u.set_boundary_type(&T6, LocationType::Up,    PDEBoundaryType::Dirichlet);
    u.set_boundary_type(&T6, LocationType::Down,  PDEBoundaryType::Dirichlet);
    // clang-format on

    Variable v("v");
    v.set_geometry(geo_tee);
    field2 v_T1("v_T1"), v_T2("v_T2"), v_T3("v_T3"), v_T4("v_T4"), v_T5("v_T5"), v_T6("v_T6");
    v.set_x_edge_field(&T1, v_T1);
    v.set_x_edge_field(&T2, v_T2);
    v.set_x_edge_field(&T3, v_T3);
    v.set_x_edge_field(&T4, v_T4);
    v.set_x_edge_field(&T5, v_T5);
    v.set_x_edge_field(&T6, v_T6);

    // clang-format off
    v.set_boundary_type(&T2, LocationType::Up,    PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T1, LocationType::Left,  PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T1, LocationType::Up,    PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T1, LocationType::Down,  PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T3, LocationType::Right, PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T3, LocationType::Up,    PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T3, LocationType::Down,  PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T4, LocationType::Left,  PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T4, LocationType::Right, PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T5, LocationType::Left,  PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T5, LocationType::Down,  PDEBoundaryType::Dirichlet);

    v.set_boundary_type(&T6, LocationType::Right, PDEBoundaryType::Neumann);
    v.set_boundary_type(&T6, LocationType::Up,    PDEBoundaryType::Dirichlet);
    v.set_boundary_type(&T6, LocationType::Down,  PDEBoundaryType::Dirichlet);
    // clang-format on

    Variable p("p");
    p.set_geometry(geo_tee);
    field2 p_T1("p_T1"), p_T2("p_T2"), p_T3("p_T3"), p_T4("p_T4"), p_T5("p_T5"), p_T6("p_T6");
    p.set_center_field(&T1, p_T1);
    p.set_center_field(&T2, p_T2);
    p.set_center_field(&T3, p_T3);
    p.set_center_field(&T4, p_T4);
    p.set_center_field(&T5, p_T5);
    p.set_center_field(&T6, p_T6);

    // clang-format off
    p.set_boundary_type(&T2, LocationType::Up,    PDEBoundaryType::Neumann);

    p.set_boundary_type(&T1, LocationType::Left,  PDEBoundaryType::Neumann);
    p.set_boundary_type(&T1, LocationType::Up,    PDEBoundaryType::Neumann);
    p.set_boundary_type(&T1, LocationType::Down,  PDEBoundaryType::Neumann);

    p.set_boundary_type(&T3, LocationType::Right, PDEBoundaryType::Neumann);
    p.set_boundary_type(&T3, LocationType::Up,    PDEBoundaryType::Neumann);
    p.set_boundary_type(&T3, LocationType::Down,  PDEBoundaryType::Neumann);

    p.set_boundary_type(&T4, LocationType::Left,  PDEBoundaryType::Neumann);
    p.set_boundary_type(&T4, LocationType::Right, PDEBoundaryType::Neumann);

    p.set_boundary_type(&T5, LocationType::Left,  PDEBoundaryType::Neumann);
    p.set_boundary_type(&T5, LocationType::Down,  PDEBoundaryType::Neumann);

    p.set_boundary_type(&T6, LocationType::Right, PDEBoundaryType::Neumann);
    p.set_boundary_type(&T6, LocationType::Up,    PDEBoundaryType::Neumann);
    p.set_boundary_type(&T6, LocationType::Down,  PDEBoundaryType::Neumann);
    // clang-format on

    ConcatNSSolver2D ns_solver(&u, &v, &p, time_config, physics_config, env_config);
    ns_solver.solve();
    ConcatPoissonSolver2D pe_solver(&v, env_config);
    pe_solver.solve();

    IO::field_to_csv(v_T1, "result/v_T1.txt");
    IO::field_to_csv(v_T2, "result/v_T2.txt");
    IO::field_to_csv(v_T3, "result/v_T3.txt");
    IO::field_to_csv(v_T4, "result/v_T4.txt");
    IO::field_to_csv(v_T5, "result/v_T5.txt");
    IO::field_to_csv(v_T6, "result/v_T6.txt");

    return 0;
}