#pragma once

#include "base/pch.h"

#include "base/domain/variable2d.h"
#include "base/location_boundary.h"

#include <unordered_map>
#include <vector>

#include "base/scheme_type.h"

class ScalarSolver2D
{
    // Simple: Only for single main domain geometry
public:
    Variable2D *u_var = nullptr, *v_var = nullptr, *s_var = nullptr;

    ScalarSolver2D(Variable2D*          in_u_var,
                   Variable2D*          in_v_var,
                   Variable2D*          in_s_var,
                   double               _nr,
                   DifferenceSchemeType _scheme);

    void variable_check();
    void solve();

    void phys_boundary_update();
    void nondiag_shared_boundary_update();

    void conv_cd2nd_diff_cd2nd_inner();
    void conv_cd2nd_diff_cd2nd_outer();

    void conv_uw1st_diff_cd2nd_inner();
    void conv_uw1st_diff_cd2nd_outer_width1();

    void conv_QUICK_diff_cd2nd_inner();
    void conv_uw1st_diff_cd2nd_outer_width2();

    void conv_TVD_VanLeer_diff_cd2nd_inner();

private:
    /**
     * Helper: Van Leer Limiter Calculation
     * psi(r) = (r + |r|) / (1 + r)
     * If s_up2 == s_up (template insufficient), r = 0 -> psi = 0 -> First-Order Upwind.
     */
    inline double get_tvd_van_leer(double s_up2, double s_up, double s_down);

    std::vector<Domain2DUniform*>                                                            domains;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, Domain2DUniform*>> adjacency;

    std::unordered_map<Domain2DUniform*, field2*> u_field_map, v_field_map, s_field_map;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, double*>> u_buffer_map, v_buffer_map,
        s_buffer_map;
    std::unordered_map<Domain2DUniform*, field2*> s_temp_field_map;

    double dt;
    double nr;

    DifferenceSchemeType scheme;
};
