#include "velocity_fixer_2d.h"

#include <stdexcept>

SolidVelocityFixer2D::SolidVelocityFixer2D(Variable2D* in_u, Variable2D* in_v)
    : u(in_u)
    , v(in_v)
{
    if (u == nullptr)
        throw std::runtime_error("SolidVelocityFixer2D: u is nullptr");
    if (v == nullptr)
        throw std::runtime_error("SolidVelocityFixer2D: v is nullptr");
    if (u->geometry == nullptr)
        throw std::runtime_error("SolidVelocityFixer2D: u has no geometry");
    if (v->geometry == nullptr)
        throw std::runtime_error("SolidVelocityFixer2D: v has no geometry");
}

void SolidVelocityFixer2D::add_shape(Shape2D* shape)
{
    shapes.push_back(shape);
}

void SolidVelocityFixer2D::build()
{
    if (shapes.empty())
        throw std::runtime_error("SolidVelocityFixer2D: no shapes added");

    auto& geometry = *u->geometry;

    for (auto* domain : geometry.domains)
    {
        auto* u_field = u->field_map.at(domain);
        auto* v_field = v->field_map.at(domain);

        int nx = u_field->get_nx();
        int ny = u_field->get_ny();

        std::vector<std::pair<int, int>> solid_u_points;
        std::vector<std::pair<int, int>> solid_v_points;

        // u: XFace, located at (i, j+0.5)
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                double x = domain->offset_x + i * domain->hx;
                double y = domain->offset_y + (j + 0.5) * domain->hy;

                if (is_inside_any_shape(x, y))
                {
                    solid_u_points.emplace_back(i, j);
                }
            }
        }

        // v: YFace, located at (i+0.5, j)
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                double x = domain->offset_x + (i + 0.5) * domain->hx;
                double y = domain->offset_y + j * domain->hy;

                if (is_inside_any_shape(x, y))
                {
                    solid_v_points.emplace_back(i, j);
                }
            }
        }

        solid_u_points_map[domain] = std::move(solid_u_points);
        solid_v_points_map[domain] = std::move(solid_v_points);
    }
}

void SolidVelocityFixer2D::apply()
{
    auto& geometry = *u->geometry;

    for (auto* domain : geometry.domains)
    {
        // Apply u
        auto u_it = solid_u_points_map.find(domain);
        if (u_it != solid_u_points_map.end())
        {
            const auto& solid_points = u_it->second;
            if (!solid_points.empty())
            {
                auto* u_field = u->field_map.at(domain);
                for (const auto& [i, j] : solid_points)
                {
                    (*u_field)(i, j) = 0.0;
                }
            }
        }

        // Apply v
        auto v_it = solid_v_points_map.find(domain);
        if (v_it != solid_v_points_map.end())
        {
            const auto& solid_points = v_it->second;
            if (!solid_points.empty())
            {
                auto* v_field = v->field_map.at(domain);
                for (const auto& [i, j] : solid_points)
                {
                    (*v_field)(i, j) = 0.0;
                }
            }
        }
    }
}

size_t SolidVelocityFixer2D::get_num_solid_u_points(Domain2DUniform* domain) const
{
    auto it = solid_u_points_map.find(domain);
    if (it == solid_u_points_map.end())
        return 0;
    return it->second.size();
}

size_t SolidVelocityFixer2D::get_num_solid_v_points(Domain2DUniform* domain) const
{
    auto it = solid_v_points_map.find(domain);
    if (it == solid_v_points_map.end())
        return 0;
    return it->second.size();
}

bool SolidVelocityFixer2D::has_solid_points(Domain2DUniform* domain) const
{
    return get_num_solid_u_points(domain) > 0 || get_num_solid_v_points(domain) > 0;
}

const std::vector<std::pair<int, int>>& SolidVelocityFixer2D::get_solid_u_points(Domain2DUniform* domain) const
{
    static std::vector<std::pair<int, int>> empty;
    auto it = solid_u_points_map.find(domain);
    if (it == solid_u_points_map.end())
        return empty;
    return it->second;
}

const std::vector<std::pair<int, int>>& SolidVelocityFixer2D::get_solid_v_points(Domain2DUniform* domain) const
{
    static std::vector<std::pair<int, int>> empty;
    auto it = solid_v_points_map.find(domain);
    if (it == solid_v_points_map.end())
        return empty;
    return it->second;
}

bool SolidVelocityFixer2D::is_inside_any_shape(double x, double y) const
{
    for (auto* shape : shapes)
    {
        if (shape->is_inside(x, y))
            return true;
    }
    return false;
}
