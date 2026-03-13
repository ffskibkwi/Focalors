#pragma once

#include <cmath>
#include <utility>

/**
 * @brief 3D Shape abstract base class
 */
class Shape3D
{
public:
    virtual ~Shape3D() = default;

    /**
     * @brief Get the closest point on the shape surface to a given point
     * @param point The point (x, y, z)
     * @return The closest point on the surface
     */
    virtual std::tuple<double, double, double> get_closest_point(double x, double y, double z) const = 0;

    /**
     * @brief Check if a point is inside the shape
     * @param point The point (x, y, z)
     * @return true if the point is inside, false otherwise
     */
    virtual bool is_inside(double x, double y, double z) const = 0;
};

/**
 * @brief Sphere shape in 3D
 */
class Sphere : public Shape3D
{
public:
    Sphere(double center_x, double center_y, double center_z, double radius)
        : cx(center_x)
        , cy(center_y)
        , cz(center_z)
        , r(radius)
    {}

    std::tuple<double, double, double> get_closest_point(double x, double y, double z) const override
    {
        double dx   = x - cx;
        double dy   = y - cy;
        double dz   = z - cz;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < 1e-14)
        {
            // Point is at center, return any point on the sphere surface
            return {cx + r, cy, cz};
        }

        // Normalize direction and scale to radius
        double nx = dx / dist;
        double ny = dy / dist;
        double nz = dz / dist;

        return {cx + nx * r, cy + ny * r, cz + nz * r};
    }

    bool is_inside(double x, double y, double z) const override
    {
        double dx = x - cx;
        double dy = y - cy;
        double dz = z - cz;
        return (dx * dx + dy * dy + dz * dz) < (r * r);
    }

    double get_center_x() const { return cx; }
    double get_center_y() const { return cy; }
    double get_center_z() const { return cz; }
    double get_radius() const { return r; }

private:
    double cx, cy, cz, r;
};
