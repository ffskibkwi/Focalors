#pragma once

#include <utility>

/**
 * @brief 2D Shape abstract base class
 */
class Shape2D
{
public:
    virtual ~Shape2D() = default;

    /**
     * @brief Get the closest point on the shape surface to a given point
     * @param point The point (x, y)
     * @return The closest point on the surface
     */
    virtual std::pair<double, double> get_closest_point(double x, double y) const = 0;

    /**
     * @brief Check if a point is inside the shape
     * @param point The point (x, y)
     * @return true if the point is inside, false otherwise
     */
    virtual bool is_inside(double x, double y) const = 0;
};

/**
 * @brief Circle shape in 2D
 */
class Circle : public Shape2D
{
public:
    Circle(double center_x, double center_y, double radius)
        : cx(center_x)
        , cy(center_y)
        , r(radius)
    {}

    std::pair<double, double> get_closest_point(double x, double y) const override
    {
        double dx = x - cx;
        double dy = y - cy;
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist < 1e-14)
        {
            // Point is at center, return any point on the circle
            return {cx + r, cy};
        }

        // Normalize direction and scale to radius
        double nx = dx / dist;
        double ny = dy / dist;

        return {cx + nx * r, cy + ny * r};
    }

    bool is_inside(double x, double y) const override
    {
        double dx = x - cx;
        double dy = y - cy;
        return (dx * dx + dy * dy) < (r * r);
    }

    double get_center_x() const { return cx; }
    double get_center_y() const { return cy; }
    double get_radius() const { return r; }

private:
    double cx, cy, r;
};
