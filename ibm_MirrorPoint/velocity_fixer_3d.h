#pragma once

#include "base/domain/variable3d.h"
#include "shape3d.h"

#include <vector>
#include <unordered_map>

/**
 * @brief Velocity Fixer for Solid Points in 3D (Staggered Grid)
 *
 * This class identifies grid points inside immersed solid shapes and provides
 * a method to set their velocity to zero. This is used to prevent concentration
 * penetration into solids due to convective transport at high Reynolds numbers.
 *
 * Staggered grid layout:
 *   - u: XFace, located at (i, j+0.5, k+0.5), size nx*ny*nz
 *   - v: YFace, located at (i+0.5, j, k+0.5), size nx*ny*nz
 *   - w: ZFace, located at (i+0.5, j+0.5, k), size nx*ny*nz
 *
 * For each stored velocity component, we check if its physical location is inside
 * any shape. Only the field values are set to zero (not buffer values).
 *
 * Usage:
 *   1. Create the fixer with velocity pointers (u, v, w)
 *   2. Add shapes using add_shape()
 *   3. Call build() to identify solid points in all domains
 *   4. Call apply() before computing scalar RHS to zero out velocity in solids
 */
class SolidVelocityFixer3D
{
public:
    /**
     * @brief Constructor
     * @param u Velocity in x-direction (XFace)
     * @param v Velocity in y-direction (YFace)
     * @param w Velocity in z-direction (ZFace)
     */
    SolidVelocityFixer3D(Variable3D* u, Variable3D* v, Variable3D* w);

    /**
     * @brief Add a shape to the solver
     * @param shape Pointer to the shape
     */
    void add_shape(Shape3D* shape);

    /**
     * @brief Build the solid point list for all shapes and domains
     * Must be called after adding all shapes and before applying
     */
    void build();

    /**
     * @brief Set velocity to zero for all solid points
     * Should be called before computing scalar field RHS
     */
    void apply();

    /**
     * @brief Get the number of solid u points for a domain
     */
    size_t get_num_solid_u_points(Domain3DUniform* domain) const;

    /**
     * @brief Get the number of solid v points for a domain
     */
    size_t get_num_solid_v_points(Domain3DUniform* domain) const;

    /**
     * @brief Get the number of solid w points for a domain
     */
    size_t get_num_solid_w_points(Domain3DUniform* domain) const;

    /**
     * @brief Check if a domain has solid points
     */
    bool has_solid_points(Domain3DUniform* domain) const;

    /**
     * @brief Get the solid u point indices for a domain (linear indices)
     */
    const std::vector<int>& get_solid_u_points(Domain3DUniform* domain) const;

    /**
     * @brief Get the solid v point indices for a domain (linear indices)
     */
    const std::vector<int>& get_solid_v_points(Domain3DUniform* domain) const;

    /**
     * @brief Get the solid w point indices for a domain (linear indices)
     */
    const std::vector<int>& get_solid_w_points(Domain3DUniform* domain) const;

private:
    /**
     * @brief Check if a grid point is inside any shape
     */
    bool is_inside_any_shape(double x, double y, double z) const;

    Variable3D* u;
    Variable3D* v;
    Variable3D* w;

    std::vector<Shape3D*> shapes;

    // Cached data for fast apply
    struct DomainCache3D
    {
        double* u_data;
        double* v_data;
        double* w_data;
        int     nx;
        int     ny;
        int     nz;
        int     nyz;  // ny * nz
        int     stride; // nx * ny * nz
    };
    std::unordered_map<Domain3DUniform*, DomainCache3D> domain_cache_map;

    // Flattened solid points (linear indices) for u
    std::unordered_map<Domain3DUniform*, std::vector<int>> solid_u_idx_map;

    // Flattened solid points (linear indices) for v
    std::unordered_map<Domain3DUniform*, std::vector<int>> solid_v_idx_map;

    // Flattened solid points (linear indices) for w
    std::unordered_map<Domain3DUniform*, std::vector<int>> solid_w_idx_map;
};
