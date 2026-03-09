#pragma once

#include "ib_kernel.hpp"
#include "particles_coordinate_3d.h"
#include "particles_ib_3d.h"
#include "variable3d.h"

#include <algorithm>
#include <functional>

/**
 * @brief Immersed boundary method solver.
 *
 * Grid space dx = dy = dz should be ensured.
 *
 * We associate a discrete volume ΔV with each force point such that the union of
 * all these volumes forms a thin shell (of thickness equal to one mesh width) around each particle.
 * It means that ImmersedBoundary point volume
 * ΔV = ib_h * ib_h * grid_h in 3D
 * ΔV = ib_h * grid_h in 3D
 */
class ImmersedBoundarySolver3D
{
public:
    ImmersedBoundarySolver3D(Variable3D*                                      _u_var,
                             Variable3D*                                      _v_var,
                             Variable3D*                                      _w_var,
                             std::unordered_map<Domain3DUniform*, PCoord3D*>& _coord_map);

    void solve();

    void u3F();

    void apply_ib_force();

    // Helper function to get velocity value from current or neighbor domain
    double get_u_value(Domain3DUniform* domain, int iix, int iiy, int iiz);
    double get_v_value(Domain3DUniform* domain, int iix, int iiy, int iiz);
    double get_w_value(Domain3DUniform* domain, int iix, int iiy, int iiz);

private:
    Variable3D* u_var;
    Variable3D* v_var;
    Variable3D* w_var;

    std::unordered_map<Domain3DUniform*, PCoord3D*> coord_map;
    std::unordered_map<Domain3DUniform*, PIB3D*>    ib_map;

    double ib_h;
    double grid_h;

    // Context helper to access field and buffer for a given domain
    struct DomainContext
    {
        Domain3DUniform*                     domain;
        std::function<double(int, int, int)> get_u;
        std::function<double(int, int, int)> get_v;
        std::function<double(int, int, int)> get_w;
    };

    std::function<DomainContext(Domain3DUniform*)> get_domain_context;
};