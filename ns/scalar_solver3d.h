#pragma once

#include "base/pch.h"

#include "base/domain/variable3d.h"
#include "base/location_boundary.h"

#include <unordered_map>
#include <vector>

#include "base/scheme_type.h"

class ScalarSolver3D
{
    // Simple: Only for single main domain geometry
public:
    Variable3D *u_var = nullptr, *v_var = nullptr, *w_var = nullptr, *s_var = nullptr;

    ScalarSolver3D(Variable3D*          in_u_var,
                   Variable3D*          in_v_var,
                   Variable3D*          in_w_var,
                   Variable3D*          in_s_var,
                   double               _nr,
                   DifferenceSchemeType _scheme);

    void variable_check();
    void solve();

    void phys_boundary_update();
    void nondiag_shared_boundary_update();

    void conv_cd2nd_diff_cd2nd_inner();
    void conv_cd2nd_diff_cd2nd_outer();

    void conv_uw1st_diff_cd2nd_inner();
    void conv_uw1st_diff_cd2nd_outer();

    void conv_QUICK_diff_cd2nd_inner();
    void conv_QUICK_diff_cd2nd_outer();

private:
    std::vector<Domain3DUniform*>                                                            domains;
    std::unordered_map<Domain3DUniform*, std::unordered_map<LocationType, Domain3DUniform*>> adjacency;

    std::unordered_map<Domain3DUniform*, field3*> u_field_map, v_field_map, w_field_map, s_field_map;
    std::unordered_map<Domain3DUniform*, std::unordered_map<LocationType, field2*>> u_buffer_map, v_buffer_map,
        w_buffer_map, s_buffer_map;
    std::unordered_map<Domain3DUniform*, field3*> s_temp_field_map;

    double dt;
    double nr;

    DifferenceSchemeType scheme;
};