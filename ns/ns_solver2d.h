#pragma once

#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/geometry_tree.hpp"
#include "base/domain/variable2d.h"
#include "base/location_boundary.h"
#include "base/pch.h"

#include "io/config.h"
#include "pe/concat/concat_solver2d.h"

#include <memory>
#include <unordered_map>
#include <vector>

class MHDModule2D;

class ConcatNSSolver2D
{
    // Simple: Only for single main domain geometry
public:
    Variable2D *           u_var = nullptr, *v_var = nullptr, *p_var = nullptr;
    ConcatPoissonSolver2D* p_solver = nullptr;

    // Non-Newtonian fields
    Variable2D *mu_var = nullptr, *tau_xx_var = nullptr, *tau_yy_var = nullptr, *tau_xy_var = nullptr;

    ConcatNSSolver2D(Variable2D*          in_u_var,
                     Variable2D*          in_v_var,
                     Variable2D*          in_p_var,
                     TimeAdvancingConfig* in_time_config,
                     PhysicsConfig*       in_physics_config,
                     EnvironmentConfig*   in_env_config = nullptr);
    ~ConcatNSSolver2D();

    // void init();
    void variable_check();
    void solve();
    void normalize_pressure();

    // Non-Newtonian methods
    void init_nonnewton(Variable2D* in_mu_var,
                        Variable2D* in_tau_xx_var,
                        Variable2D* in_tau_yy_var,
                        Variable2D* in_tau_xy_var);
    void solve_nonnewton();
    void viscosity_update();
    void stress_update();
    void stress_buffer_update();
    void euler_conv_diff_inner_nonnewton();
    void euler_conv_diff_outer_nonnewton();

    /**
     * Physics boundary update.
     *
     * Step 1 of PISO.
     *
     * Iterate through each boundary that may require updating,
     * and determine whether updating is needed based on the connection conditions.
     *
     * If a boundary should be update, the updating strategy is:
     * A field-related row (or column) is denoted as fb, and a physical boundary is denoted as pb.
     * (1) When fb coincides with pb, compute fb based on the pb type and pb value.
     * (2) When fb and its corresponding buffer are symmetric with respect to pb, compute the buffer based on the pb
     * type, pb value, and fb.
     * For the first case, if the boundary condition is of Dirichlet type,
     * the assignment only needs to be done once, without repeating it in each iteration of the solution loop.
     * However, if the boundary condition is of Neumann type,
     * it will depend on the adjacent row/column near the boundary,
     * which in turn requires the solution of the Navier–Stokes equations.
     * Therefore, this type of boundary needs to be updated in every iteration of the solution loop.
     * For simplicity in programming, we will not distinguish
     * whether a boundary needs to be assigned only once or in every iteration,
     * since the computational cost of the boundary handling function is already very small.
     */
    void phys_boundary_update();
    /**
     * Non-diagonal shared boundary update.
     *
     * Step 2 of PISO.
     *
     * Iterate through each boundary that may require updating,
     * and determine whether updating is needed based on the connection conditions.
     *
     * If a boundary should be update, the updating strategy is:
     *
     * When a local buffer or corner has a physically coincident position pos in a non-diagonal neighboring field
     * (denoted as fn), assign fn(pos) to the local buffer or corner.
     */
    void nondiag_shared_boundary_update();

    /**
     * Diagonal shared boundary update.
     *
     * Step 3 of PISO.
     *
     * Iterate through each boundary that may require updating,
     * and determine whether updating is needed based on the connection conditions.
     *
     * Updating strategy is similar to nondiag_shared_boundary_update.
     *
     * The importance of diagonal shared boundary updating:
     * When performing NS calculations on shared boundary, if the calculation point is an endpoint of a column or row,
     * then certain buffer values ​​need to be taken.
     * Among these buffer dependencies, one buffer is special because it:
     * (1) When a diagonal domain exists, it originates from the diagonal domain.
     * (2) When a diagonal domain does not exist, it originates from the directly adjacent domain.
     *
     * In some cases, the domain might not use the updated buffer/corner at this step, but that's okay.
     * To avoid programming complexity, we won't determine whether to update based on whether it will be used.
     * We'll simply determine it based on the connection conditions.
     */
    void diag_shared_boundary_update();

    /**
     * Euler convection and diffusion term inner calculation.
     *
     * Step 4 of PISO.
     *
     * It is independent of the buffer/corner.
     */
    void euler_conv_diff_inner();

    /**
     * Euler convection and diffusion term outer calculation.
     *
     * Step 5 of PISO.
     *
     * It is dependent of the buffer/corner.
     */
    void euler_conv_diff_outer();

    void velocity_div_inner();
    void velocity_div_outer();
    void pressure_buffer_update();
    void add_pressure_gradient();

private:
    std::vector<Domain2DUniform*>                                                            domains;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, Domain2DUniform*>> adjacency;

    std::unordered_map<Domain2DUniform*, field2*> u_field_map, v_field_map, p_field_map;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, double*>> u_buffer_map, v_buffer_map,
        p_buffer_map;
    std::unordered_map<Domain2DUniform*, field2*> u_temp_field_map, v_temp_field_map;

    // Non-Newtonian field maps
    std::unordered_map<Domain2DUniform*, field2*> mu_field_map;
    std::unordered_map<Domain2DUniform*, field2*> tau_xx_field_map;
    std::unordered_map<Domain2DUniform*, field2*> tau_yy_field_map;
    std::unordered_map<Domain2DUniform*, field2*> tau_xy_field_map;

    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, double*>> tau_xx_buffer_map,
        tau_yy_buffer_map;

    std::unordered_map<Domain2DUniform*, double>& left_up_corner_value_map;
    std::unordered_map<Domain2DUniform*, double>& right_down_corner_value_map;

    EnvironmentConfig* env_config;

    std::unique_ptr<MHDModule2D> mhd_module;

    TimeAdvancingConfig* time_config;
    double               dt;
    int                  num_it;
    int                  corr_it;

    PhysicsConfig* phy_config;
    double         nu;
};