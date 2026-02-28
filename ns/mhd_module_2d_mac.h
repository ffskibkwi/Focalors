#pragma once

#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "pe/concat/concat_solver2d.h"

#include <memory>
#include <unordered_map>
#include <vector>

/**
 * @brief MHD module for 2D quasi-static/induced electric potential method (MAC Grid)
 *
 * Variable2D Layout (Staggered MAC):
 * - phi: Cell Center (i, j)
 * - u:   Right Face  (i+1/2, j)
 * - v:   Top Face    (i, j+1/2)
 *
 * Derived Fields:
 * - Jx:  Right Face  (i+1/2, j) [Collocated with u]
 * - Jy:  Top Face    (i, j+1/2) [Collocated with v]
 * - Fx:  Right Face  (i+1/2, j) [Collocated with u]
 * - Fy:  Top Face    (i, j+1/2) [Collocated with v]
 *
 * Algorithm:
 * 1. Compute RHS = div(u x B) at Cell Centers using consistent Flux-Divergence.
 *    RHS = [ (v*Bz)_right - (v*Bz)_left ]/dx + [ (-u*Bz)_top - (-u*Bz)_bottom ]/dy
 *    Requires interpolating v to X-Faces and u to Y-Faces.
 * 2. Solve Poisson: div(grad phi) = RHS
 *    Enforces div(J) = 0.
 * 3. Update Current Density J = -grad(phi) + u x B at Faces.
 *    Jx = -dphi/dx + v_interp * Bz
 *    Jy = -dphi/dy - u_interp * Bz
 * 4. Calculate Lorentz Force F = J x B at Faces.
 *    Fx = Jy_interp * Bz
 *    Fy = -Jx_interp * Bz
 *    Interpolate J to the required face locations.
 */
class MHDModule2D
{
public:
    MHDModule2D(Variable2D* in_u_var, Variable2D* in_v_var);
    ~MHDModule2D();

    void init(Variable2D* phi_var = nullptr);

    void solveElectricPotential();
    void updateCurrentDensity();
    void applyLorentzForce();

private:
    void buffer_update_phi();
    void buffer_update_j();

    // Reference to NS velocity variable2ds
    Variable2D* m_uVar;
    Variable2D* m_vVar;

    // MHD-specific variable2ds
    Variable2D*                 m_phiVar   = nullptr; // Electric potential (Center, non-owning)
    std::unique_ptr<Variable2D> m_phiOwned = nullptr; // Owned phi when created internally
    // J stored at Faces to behave like velocity
    std::unique_ptr<Variable2D> m_jxVar; // Current density X (X-Face)
    std::unique_ptr<Variable2D> m_jyVar; // Current density Y (Y-Face)
    std::unique_ptr<Variable2D> m_jzVar; // Current density Z (Center - 2D special case)

    // Poisson solver for electric potential
    std::unique_ptr<ConcatPoissonSolver2D> m_phiSolver;

    // Cached parameters
    std::vector<Domain2DUniform*>                                                            m_domains;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, Domain2DUniform*>> m_adjacency;

    double m_Bx          = 0.0;
    double m_By          = 0.0;
    double m_Bz          = 0.0;
    double m_Ha          = 0.0;
    double m_Re          = 0.0;
    double m_dt          = 0.0;
    double m_lorentzCoef = 0.0;

    // Field storage
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_phiFieldStorage;
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_jxFieldStorage;
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_jyFieldStorage;
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_jzFieldStorage;

    // Field maps for quick access
    std::unordered_map<Domain2DUniform*, field2*> m_uFieldMap, m_vFieldMap;
    std::unordered_map<Domain2DUniform*, field2*> m_phiFieldMap, m_jxFieldMap, m_jyFieldMap, m_jzFieldMap;

    bool m_initialized = false;
};
