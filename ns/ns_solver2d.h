#pragma once

#include "base/pch.h"
#include "base/location_boundary.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable.h"
#include "base/domain/geometry_tree.hpp"

#include "pe/concat/concat_solver2d.h"
#include "io/config.h"

#include <vector>
#include <unordered_map>

class ConcatNSSolver2D
{
    //Simple: Only for single main domain geometry
public:
    Variable* u_var = nullptr, *v_var = nullptr, *p_var = nullptr;
    
    ConcatNSSolver2D(Variable* in_u_var, Variable* in_v_var, Variable* in_p_var, TimeAdvancingConfig* in_time_config, PhysicsConfig* in_physics_config, EnvironmentConfig* in_env_config = nullptr);
    ~ConcatNSSolver2D();

    // void init();
    void variable_check();
    void solve();
    

private:
    ConcatPoissonSolver2D* p_solver = nullptr;

    std::vector<Domain2DUniform*> domains;    
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, Domain2DUniform*>> adjacency;
    
    std::unordered_map<Domain2DUniform*, field2*> u_field_map, v_field_map, p_field_map;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, double*>> u_buffer_map, v_buffer_map, p_buffer_map;
    std::unordered_map<Domain2DUniform*, field2*> u_temp_field_map, v_temp_field_map;

    std::unordered_map<Domain2DUniform*, double>& left_up_corner_value_map;
    std::unordered_map<Domain2DUniform*, double>& right_down_corner_value_map;
    
    EnvironmentConfig* env_config;
    
    TimeAdvancingConfig* time_config;
    double dt;
    int num_it;

    PhysicsConfig* phy_config;
    double nu;

    void euler_conv_diff_inner();
    void euler_conv_diff_outer();
    void velocity_buffer_pass();

    void pressure_calculate();
    void velocity_div_calculate();
    void velocity_div_inner();
    void velocity_div_outer();
    void pressure_buffer_pass();
    void velocity_update();

    void boundary_init();
    void boundary_update();
};