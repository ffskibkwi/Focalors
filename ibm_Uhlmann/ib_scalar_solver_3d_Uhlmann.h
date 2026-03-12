#pragma once

#include "base/domain/variable3d.h"
#include "ib_kernel.hpp"
#include "particle/particles_coordinate_3d.h"
#include "particles_ib_scalar.h"

#include <algorithm>
#include <functional>

// Markus Uhlmann. 2005. An immersed boundary method with direct forcing for the simulation of particulate flows. J.
// Comput. Phys. 209, 2 (1 November 2005), 448–476. https://doi.org/10.1016/j.jcp.2005.03.017

/**
 * @brief Immersed boundary method solver for scalar fields (e.g., concentration) in 3D.
 *
 * Grid space dx = dy = dz should be ensured.
 *
 * This solver handles scalar quantities that are cell-centered (e.g., concentration).
 * The IB point is at the grid cell center, and the scalar index matches the grid index.
 * Support domain: [-2, +2] in x, y, and z directions (5 points in each direction).
 */
class IBScalarSolver3D_Uhlmann
{
public:
    IBScalarSolver3D_Uhlmann(Variable3D* _scalar_var, std::unordered_map<Domain3DUniform*, PCoord3D*>& _coord_map);

    void solve();

    void calc_ib_scalar();
    void apply_ib_scalar();

    // Helper function to get scalar value from current or neighbor domain
    double& get_scalar_value(Domain3DUniform* domain, int iix, int iiy, int iiz);

    // Configure IB parameters (particle spacing and grid spacing)
    void set_parameters(double ib_spacing, double grid_spacing)
    {
        ib_h   = ib_spacing;
        grid_h = grid_spacing;
    }

    // Access IB data for a domain
    PIBScalar*                                        get_ib_data(Domain3DUniform* domain) { return ib_map[domain]; }
    std::unordered_map<Domain3DUniform*, PIBScalar*>& get_ib_map() { return ib_map; }

private:
    Variable3D* scalar_var;

    std::unordered_map<Domain3DUniform*, PCoord3D*>  coord_map;
    std::unordered_map<Domain3DUniform*, PIBScalar*> ib_map;

    double ib_h;
    double grid_h;

    // Context helper to access field and buffer for a given domain
    struct DomainContext
    {
        Domain3DUniform*                     domain;
        std::function<double(int, int, int)> get_scalar;
    };

    std::function<DomainContext(Domain3DUniform*)> get_domain_context;
};
