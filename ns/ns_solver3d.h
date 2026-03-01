#pragma once

#include "base/pch.h"

#include "pe/concat/concat_solver3d.h"

#include <unordered_map>
#include <vector>

class ConcatNSSolver3D
{
    // Simple: Only for single main domain geometry
public:
    Variable3D *           u_var = nullptr, *v_var = nullptr, *w_var = nullptr, *p_var = nullptr;
    ConcatPoissonSolver3D* p_solver = nullptr;

    // Non-Newtonian fields
    Variable3D *mu_var = nullptr, *tau_xx_var = nullptr, *tau_yy_var = nullptr, *tau_zz_var = nullptr,
               *tau_xy_var = nullptr, *tau_xz_var = nullptr, *tau_yz_var = nullptr;

    ConcatNSSolver3D(Variable3D*            in_u_var,
                     Variable3D*            in_v_var,
                     Variable3D*            in_w_var,
                     Variable3D*            in_p_var,
                     ConcatPoissonSolver3D* in_p_solver);

    // void init();
    void variable_check();
    void solve();
    void normalize_pressure();

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
    std::vector<Domain3DUniform*>                                                            domains;
    std::unordered_map<Domain3DUniform*, std::unordered_map<LocationType, Domain3DUniform*>> adjacency;

    std::unordered_map<Domain3DUniform*, field3*> u_field_map, v_field_map, w_field_map, p_field_map;
    std::unordered_map<Domain3DUniform*, std::unordered_map<LocationType, field2*>> u_buffer_map, v_buffer_map,
        w_buffer_map, p_buffer_map;
    std::unordered_map<Domain3DUniform*, field3*> u_temp_field_map, v_temp_field_map, w_temp_field_map;

    std::unordered_map<Domain3DUniform*, double*>&u_corner_value_map_y, u_corner_value_map_z;
    std::unordered_map<Domain3DUniform*, double*>&v_corner_value_map_x, v_corner_value_map_z;
    std::unordered_map<Domain3DUniform*, double*>&w_corner_value_map_x, w_corner_value_map_y;

    double dt;
    int    num_it;
    int    corr_it;
    double nu;
};