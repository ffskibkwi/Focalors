#pragma once

#include "base/domain/variable3d.h"
#include "shape3d.h"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Immersed Boundary Mirror Point Solver for 3D
 *
 * This solver implements the direct forcing method for immersed boundary problems.
 * It uses mirror points to enforce boundary conditions on scalar fields defined on grid points.
 *
 * For each interior point (inside the shape), we find its mirror point on the other side of the boundary.
 * The mirror point value is obtained via trilinear interpolation from surrounding grid points.
 *
 * Dirichlet: phi + phi_mirror = 2 * BC
 * Neumann: (phi_mirror - phi) / delta_l = BC
 */
class IBSolver3D_MirrorPoint
{
public:
    /**
     * @brief Constructor
     * @param var The variable to solve (e.g., temperature, concentration)
     * @param boundary_type The boundary type (Dirichlet or Neumann)
     * @param boundary_value The boundary value (BC)
     */
    IBSolver3D_MirrorPoint(Variable3D* var, PDEBoundaryType boundary_type, double boundary_value);

    /**
     * @brief Add a shape to the solver
     * @param shape Pointer to the shape
     */
    void add_shape(Shape3D* shape);

    /**
     * @brief Set the template radius for interior point detection
     * @param radius The radius (default is 2)
     */
    void set_template_radius(int radius) { template_radius = radius; }

    /**
     * @brief Get the template radius
     */
    int get_template_radius() const { return template_radius; }

    /**
     * @brief Build the interior point list for all shapes and domains
     * Must be called after adding all shapes and before solving
     */
    void build();

    /**
     * @brief Apply the mirror point boundary condition
     * This should be called after the field values are computed
     */
    void apply();

    /**
     * @brief Get the number of interior points for a domain
     */
    size_t get_num_interior_points(Domain3DUniform* domain) const;

    /**
     * @brief Check if a domain has interior points
     */
    bool has_interior_points(Domain3DUniform* domain) const;

    /**
     * @brief Get the interior point indices for a domain
     */
    const std::vector<std::tuple<int, int, int>>& get_interior_points(Domain3DUniform* domain) const;

private:
    /**
     * @brief Get the physical coordinates of a grid point
     */
    std::tuple<double, double, double> get_physical_location(Domain3DUniform* domain, int i, int j, int k) const;

    /**
     * @brief Check if a grid point is inside any shape
     */
    bool is_inside_any_shape(double x, double y, double z) const;

    /**
     * @brief Find the mirror point for an interior point
     * @param[in] x Physical x coordinate of the interior point
     * @param[in] y Physical y coordinate of the interior point
     * @param[in] z Physical z coordinate of the interior point
     * @param[out] mirror_x Physical x coordinate of the mirror point
     * @param[out] mirror_y Physical y coordinate of the mirror point
     * @param[out] mirror_z Physical z coordinate of the mirror point
     * @param[out] delta_l Distance from interior point to mirror point (wall normal distance)
     */
    void find_mirror_point(double x, double y, double z, double& mirror_x, double& mirror_y, double& mirror_z,
                          double& delta_l) const;

    /**
     * @brief Trilinear interpolation for mirror point value
     * @param domain The domain where the mirror point is located
     * @param mirror_x Physical x coordinate of mirror point
     * @param mirror_y Physical y coordinate of mirror point
     * @param mirror_z Physical z coordinate of mirror point
     * @return Interpolated value at mirror point
     */
    double interpolate_mirror_value(Domain3DUniform* domain, double mirror_x, double mirror_y, double mirror_z) const;

    /**
     * @brief Get field value with boundary handling
     * @param domain The domain
     * @param i Index in x direction
     * @param j Index in y direction
     * @param k Index in z direction
     * @return The field value (including buffer handling)
     */
    double get_field_value(Domain3DUniform* domain, int i, int j, int k) const;

    Variable3D*                              var;
    PDEBoundaryType                          boundary_type;
    double                                   boundary_value;
    int                                      template_radius = 2;

    std::vector<Shape3D*>                    shapes;

    // Interior points for each domain: domain -> vector of (i, j, k) indices
    std::unordered_map<Domain3DUniform*, std::vector<std::tuple<int, int, int>>> interior_points_map;

    // Mirror point info for each interior point: domain -> vector of mirror data
    struct MirrorInfo
    {
        int    i, j, k;    // Interior point index
        double mirror_x;    // Mirror point physical x
        double mirror_y;    // Mirror point physical y
        double mirror_z;    // Mirror point physical z
        double delta_l;    // Distance to mirror point
    };
    std::unordered_map<Domain3DUniform*, std::vector<MirrorInfo>> mirror_info_map;
};
