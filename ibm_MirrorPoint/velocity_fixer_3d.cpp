#include "velocity_fixer_3d.h"

#include <stdexcept>

SolidVelocityFixer3D::SolidVelocityFixer3D(Variable3D* in_u, Variable3D* in_v, Variable3D* in_w)
    : u(in_u)
    , v(in_v)
    , w(in_w)
{
    if (u == nullptr)
        throw std::runtime_error("SolidVelocityFixer3D: u is nullptr");
    if (v == nullptr)
        throw std::runtime_error("SolidVelocityFixer3D: v is nullptr");
    if (w == nullptr)
        throw std::runtime_error("SolidVelocityFixer3D: w is nullptr");
    if (u->geometry == nullptr)
        throw std::runtime_error("SolidVelocityFixer3D: u has no geometry");
    if (v->geometry == nullptr)
        throw std::runtime_error("SolidVelocityFixer3D: v has no geometry");
    if (w->geometry == nullptr)
        throw std::runtime_error("SolidVelocityFixer3D: w has no geometry");
}

void SolidVelocityFixer3D::add_shape(Shape3D* shape)
{
    shapes.push_back(shape);
}

void SolidVelocityFixer3D::build()
{
    if (shapes.empty())
        throw std::runtime_error("SolidVelocityFixer3D: no shapes added");

    auto& geometry = *u->geometry;

    for (auto* domain : geometry.domains)
    {
        auto* u_field = u->field_map.at(domain);
        auto* v_field = v->field_map.at(domain);
        auto* w_field = w->field_map.at(domain);

        int nx = u_field->get_nx();
        int ny = u_field->get_ny();
        int nz = u_field->get_nz();

        std::vector<std::tuple<int, int, int>> solid_u_points;
        std::vector<std::tuple<int, int, int>> solid_v_points;
        std::vector<std::tuple<int, int, int>> solid_w_points;

        // u: XFace, located at (i, j+0.5, k+0.5)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    double x = domain->offset_x + i * domain->hx;
                    double y = domain->offset_y + (j + 0.5) * domain->hy;
                    double z = domain->offset_z + (k + 0.5) * domain->hz;

                    if (is_inside_any_shape(x, y, z))
                    {
                        solid_u_points.emplace_back(i, j, k);
                    }
                }
            }
        }

        // v: YFace, located at (i+0.5, j, k+0.5)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    double x = domain->offset_x + (i + 0.5) * domain->hx;
                    double y = domain->offset_y + j * domain->hy;
                    double z = domain->offset_z + (k + 0.5) * domain->hz;

                    if (is_inside_any_shape(x, y, z))
                    {
                        solid_v_points.emplace_back(i, j, k);
                    }
                }
            }
        }

        // w: ZFace, located at (i+0.5, j+0.5, k)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    double x = domain->offset_x + (i + 0.5) * domain->hx;
                    double y = domain->offset_y + (j + 0.5) * domain->hy;
                    double z = domain->offset_z + k * domain->hz;

                    if (is_inside_any_shape(x, y, z))
                    {
                        solid_w_points.emplace_back(i, j, k);
                    }
                }
            }
        }

        solid_u_points_map[domain] = std::move(solid_u_points);
        solid_v_points_map[domain] = std::move(solid_v_points);
        solid_w_points_map[domain] = std::move(solid_w_points);
    }
}

void SolidVelocityFixer3D::apply()
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
                for (const auto& [i, j, k] : solid_points)
                {
                    (*u_field)(i, j, k) = 0.0;
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
                for (const auto& [i, j, k] : solid_points)
                {
                    (*v_field)(i, j, k) = 0.0;
                }
            }
        }

        // Apply w
        auto w_it = solid_w_points_map.find(domain);
        if (w_it != solid_w_points_map.end())
        {
            const auto& solid_points = w_it->second;
            if (!solid_points.empty())
            {
                auto* w_field = w->field_map.at(domain);
                for (const auto& [i, j, k] : solid_points)
                {
                    (*w_field)(i, j, k) = 0.0;
                }
            }
        }
    }
}

size_t SolidVelocityFixer3D::get_num_solid_u_points(Domain3DUniform* domain) const
{
    auto it = solid_u_points_map.find(domain);
    if (it == solid_u_points_map.end())
        return 0;
    return it->second.size();
}

size_t SolidVelocityFixer3D::get_num_solid_v_points(Domain3DUniform* domain) const
{
    auto it = solid_v_points_map.find(domain);
    if (it == solid_v_points_map.end())
        return 0;
    return it->second.size();
}

size_t SolidVelocityFixer3D::get_num_solid_w_points(Domain3DUniform* domain) const
{
    auto it = solid_w_points_map.find(domain);
    if (it == solid_w_points_map.end())
        return 0;
    return it->second.size();
}

bool SolidVelocityFixer3D::has_solid_points(Domain3DUniform* domain) const
{
    return get_num_solid_u_points(domain) > 0 || get_num_solid_v_points(domain) > 0
        || get_num_solid_w_points(domain) > 0;
}

const std::vector<std::tuple<int, int, int>>& SolidVelocityFixer3D::get_solid_u_points(Domain3DUniform* domain) const
{
    static std::vector<std::tuple<int, int, int>> empty;
    auto it = solid_u_points_map.find(domain);
    if (it == solid_u_points_map.end())
        return empty;
    return it->second;
}

const std::vector<std::tuple<int, int, int>>& SolidVelocityFixer3D::get_solid_v_points(Domain3DUniform* domain) const
{
    static std::vector<std::tuple<int, int, int>> empty;
    auto it = solid_v_points_map.find(domain);
    if (it == solid_v_points_map.end())
        return empty;
    return it->second;
}

const std::vector<std::tuple<int, int, int>>& SolidVelocityFixer3D::get_solid_w_points(Domain3DUniform* domain) const
{
    static std::vector<std::tuple<int, int, int>> empty;
    auto it = solid_w_points_map.find(domain);
    if (it == solid_w_points_map.end())
        return empty;
    return it->second;
}

bool SolidVelocityFixer3D::is_inside_any_shape(double x, double y, double z) const
{
    for (auto* shape : shapes)
    {
        if (shape->is_inside(x, y, z))
            return true;
    }
    return false;
}
