#include "base/domain/geometry2d.h"
#include "base/domain/domain2d.h"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "base/domain/geometry_tree.hpp"

#include "base/location_boundary.h"

#include "pe/concat/concat_solver2d.h"

#include "io/csv_writer_2d.h"
#include "io/env_config.h"

int main(int argc, char* argv[])
{
    Geometry2D geo_tee;
    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes = true;
    env_config->showCurrentStep = true;

    // Grid parameters
    Domain2DUniform T2(10, 10, 1.0, 1.0, "T2"); // 中心
    Domain2DUniform T1("T1"); T1.set_nx(20); T1.set_lx(2.0);
    Domain2DUniform T3("T3"); T3.set_nx(30); T3.set_lx(3.0);
    Domain2DUniform T4("T4"); T4.set_ny(10); T4.set_ly(1.0);
    Domain2DUniform T5("T5"); T5.set_ny(20); T5.set_ly(2.0);
    Domain2DUniform T6("T6"); T6.set_nx(30); T6.set_lx(3.0);

    // Set boundary
    T2.set_boundary(LocationType::Up,    PDEBoundaryType::Dirichlet);

    T1.set_boundary(LocationType::Left,  PDEBoundaryType::Dirichlet);
    T1.set_boundary(LocationType::Up,    PDEBoundaryType::Dirichlet);
    T1.set_boundary(LocationType::Down,  PDEBoundaryType::Dirichlet);

    T3.set_boundary(LocationType::Right, PDEBoundaryType::Dirichlet);
    T3.set_boundary(LocationType::Up,    PDEBoundaryType::Dirichlet);
    T3.set_boundary(LocationType::Down,  PDEBoundaryType::Dirichlet);

    T4.set_boundary(LocationType::Left,  PDEBoundaryType::Dirichlet);
    T4.set_boundary(LocationType::Right, PDEBoundaryType::Dirichlet);

    T5.set_boundary(LocationType::Left,  PDEBoundaryType::Dirichlet);
    T5.set_boundary(LocationType::Down,  PDEBoundaryType::Dirichlet);

    T6.set_boundary(LocationType::Right, PDEBoundaryType::Dirichlet);
    T6.set_boundary(LocationType::Up,    PDEBoundaryType::Dirichlet);
    T6.set_boundary(LocationType::Down,  PDEBoundaryType::Dirichlet);

    // Construct geometry
    geo_tee.add_domain(T1);
    geo_tee.add_domain(T2);
    geo_tee.add_domain(T3);
    geo_tee.add_domain(T4);
    geo_tee.add_domain(T5);
    geo_tee.add_domain(T6);

    geo_tee.connect(T2, LocationType::Left,  T1);
    geo_tee.connect(T2, LocationType::Right, T3);
    geo_tee.connect(T2, LocationType::Down,  T4);
    geo_tee.connect(T4, LocationType::Down,  T5);
    geo_tee.connect(T5, LocationType::Right, T6);

    // geo_tee.check();
    // geo_tee.solve_prepare();
    // std::cout << "tee optimal tree:" << std::endl;
    // TreeUtils::printTreeMap(geo_tee.tree_root, geo_tee.tree_map);
    
    // if (geo_tee.tree_root)
    //     std::cout << "tee tree root: " << geo_tee.tree_root->name << std::endl;
    
    // if (!geo_tee.parent_map.empty())
    // {
    //     for (auto &[key, value] : geo_tee.parent_map)
    //     {
    //         std::cout << "parent of " << key->name << " is " << value.second->name << " on " << locationTypeToString(value.first) << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    Variable v("v");
    v.set_geometry(geo_tee);
    field2 v_T1("v_T1"), v_T2("v_T2"), v_T3("v_T3"), v_T4("v_T4"), v_T5("v_T5"), v_T6("v_T6");
    v.set_center_field(T1, v_T1);
    v.set_center_field(T2, v_T2);
    v.set_center_field(T3, v_T3);
    v.set_center_field(T4, v_T4);
    v.set_center_field(T5, v_T5);
    v.set_center_field(T6, v_T6);

    for (int i = 3; i < v_T4.get_nx() - 3; i++)
    {
        for (int j = 3; j < v_T4.get_ny() - 3; j++)
        {
            v_T4(i, j) = 1.0;
        }
    }


    ConcatSolver2D solver(v, env_config);
    solver.solve();

    IO::field_to_csv(v_T1, "result/v_T1.txt");
    IO::field_to_csv(v_T2, "result/v_T2.txt");
    IO::field_to_csv(v_T3, "result/v_T3.txt");
    IO::field_to_csv(v_T4, "result/v_T4.txt");
    IO::field_to_csv(v_T5, "result/v_T5.txt");
    IO::field_to_csv(v_T6, "result/v_T6.txt");

    return 0;
}