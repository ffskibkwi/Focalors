#pragma once

#include "base/domain/variable2d.h"
#include "ib_kernel.hpp"
#include "particles_coordinate_2d.h"
#include "particles_ib_2d.h"

#include <functional>

// Markus Uhlmann. 2005. An immersed boundary method with direct forcing for the simulation of particulate flows. J.
// Comput. Phys. 209, 2 (1 November 2005), 448–476. https://doi.org/10.1016/j.jcp.2005.03.017

/**
 * @brief Immersed boundary method solver.
 *
 * Grid space dx = dy should be ensured.
 *
 * We associate a discrete volume ΔV with each force point such that the union of
 * all these volumes forms a thin shell (of thickness equal to one mesh width) around each particle.
 * It means that IB point volume
 * ΔV = ib_h * ib_h * grid_h in 3D
 * ΔV = ib_h * grid_h in 2D
 */
class IBVelocitySolver2D_Uhlmann
{
public:
    IBVelocitySolver2D_Uhlmann(Variable2D*                                      _u_var,
                               Variable2D*                                      _v_var,
                               std::unordered_map<Domain2DUniform*, PCoord2D*>& _coord_map);

    void solve();

    void calc_ib_force();
    void apply_ib_force();

    // Helper function to get velocity reference from current or neighbor domain
    double& get_u_value(Domain2DUniform* domain, int iix, int iiy);
    double& get_v_value(Domain2DUniform* domain, int iix, int iiy);

    // Configure IB parameters (particle spacing and grid spacing)
    void set_parameters(double ib_spacing, double grid_spacing)
    {
        ib_h   = ib_spacing;
        grid_h = grid_spacing;
    }

    // Access IB data for a domain
    PIB2D*                                        get_ib_data(Domain2DUniform* domain) { return ib_map[domain]; }
    std::unordered_map<Domain2DUniform*, PIB2D*>& get_ib_map() { return ib_map; }

private:
    Variable2D* u_var;
    Variable2D* v_var;

    std::unordered_map<Domain2DUniform*, PCoord2D*> coord_map;
    std::unordered_map<Domain2DUniform*, PIB2D*>    ib_map;

    double ib_h;
    double grid_h;

    // Context helper to access field and buffer for a given domain
    struct DomainContext
    {
        Domain2DUniform*                domain;
        std::function<double(int, int)> get_u;
        std::function<double(int, int)> get_v;
    };

    std::function<DomainContext(Domain2DUniform*)> get_domain_context;
};