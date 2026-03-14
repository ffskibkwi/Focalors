#pragma once

#include "base/domain/variable2d.h"
#include "base/location_boundary.h"
#include "ib_kernel.hpp"
#include "particle/particles_coordinate_2d.h"
#include "particles_ib_scalar.h"

#include <functional>

// Markus Uhlmann. 2005. An immersed boundary method with direct forcing for the simulation of particulate flows. J.
// Comput. Phys. 209, 2 (1 November 2005), 448–476. https://doi.org/10.1016/j.jcp.2005.03.017

/**
 * @brief Immersed boundary method solver for scalar fields (e.g., concentration).
 *
 * Grid space dx = dy should be ensured.
 *
 * This solver handles scalar quantities that are cell-centered (e.g., concentration).
 * The IB point is at the grid cell center, and the scalar index matches the grid index.
 * Support domain: [-2, +2] in both x and y directions (5 points in each direction).
 *
 * For Neumann boundary, requires PIBNormal data containing normal vectors.
 */
class IBScalarSolver2D_Uhlmann
{
public:
    IBScalarSolver2D_Uhlmann(Variable2D*                                      _scalar_var,
                               std::unordered_map<Domain2DUniform*, PCoord2D*>&   _coord_map,
                               std::unordered_map<Domain2DUniform*, PIBNormal*>& _normal_map);

    void solve();

    void calc_ib_scalar();
    void apply_ib_scalar();

    // Set PDE type (Dirichlet or Neumann) using existing PDEBoundaryType
    void set_pde_type(PDEBoundaryType type) { pde_type = type; }

    // For Neumann boundary: set boundary condition value (dphi/dn = BC)
    void set_neumann_bc(double bc_value) { neumann_bc = bc_value; }

    // Helper function to get scalar value from current or neighbor domain
    double& get_scalar_value(Domain2DUniform* domain, int iix, int iiy);

    // Configure IB parameters (particle spacing and grid spacing)
    void set_parameters(double ib_spacing, double grid_spacing)
    {
        ib_h   = ib_spacing;
        grid_h = grid_spacing;
    }

    // Access IB data for a domain
    PIBScalar*                                        get_ib_data(Domain2DUniform* domain) { return ib_map[domain]; }
    std::unordered_map<Domain2DUniform*, PIBScalar*>& get_ib_map() { return ib_map; }

    // Access normal data for a domain
    PIBNormal*                           get_normal_data(Domain2DUniform* domain) { return normal_map[domain]; }
    std::unordered_map<Domain2DUniform*, PIBNormal*>& get_normal_map() { return normal_map; }

private:
    Variable2D* scalar_var;

    std::unordered_map<Domain2DUniform*, PCoord2D*>  coord_map;
    std::unordered_map<Domain2DUniform*, PIBScalar*> ib_map;
    std::unordered_map<Domain2DUniform*, PIBNormal*> normal_map;

    double ib_h;
    double grid_h;

    // PDE type for boundary condition
    PDEBoundaryType pde_type = PDEBoundaryType::Dirichlet;

    // Neumann boundary condition value
    double neumann_bc = 0.0;

    // Context helper to access field and buffer for a given domain
    struct DomainContext
    {
        Domain2DUniform*                domain;
        std::function<double(int, int)> get_scalar;
    };

    std::function<DomainContext(Domain2DUniform*)> get_domain_context;
};
