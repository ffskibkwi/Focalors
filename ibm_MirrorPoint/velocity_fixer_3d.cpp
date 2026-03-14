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
        int nyz = ny * nz;
        int stride = nx * nyz;

        // Cache for fast apply
        DomainCache3D cache;
        cache.u_data  = u_field->value;
        cache.v_data = v_field->value;
        cache.w_data = w_field->value;
        cache.nx     = nx;
        cache.ny     = ny;
        cache.nz     = nz;
        cache.nyz    = nyz;
        cache.stride = stride;
        domain_cache_map[domain] = cache;

        std::vector<int> solid_u_idx;
        std::vector<int> solid_v_idx;
        std::vector<int> solid_w_idx;
        solid_u_idx.reserve(stride / 8);
        solid_v_idx.reserve(stride / 8);
        solid_w_idx.reserve(stride / 8);

        // u: XFace, located at (i, j+0.5, k+0.5)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                int base = j * nz + k;
                for (int i = 0; i < nx; ++i)
                {
                    double x = domain->offset_x + i * domain->hx;
                    double y = domain->offset_y + (j + 0.5) * domain->hy;
                    double z = domain->offset_z + (k + 0.5) * domain->hz;

                    if (is_inside_any_shape(x, y, z))
                    {
                        // Store linear index: i * ny * nz + j * nz + k
                        solid_u_idx.push_back(i * nyz + base);
                    }
                }
            }
        }

        // v: YFace, located at (i+0.5, j, k+0.5)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                int base = j * nz + k;
                for (int i = 0; i < nx; ++i)
                {
                    double x = domain->offset_x + (i + 0.5) * domain->hx;
                    double y = domain->offset_y + j * domain->hy;
                    double z = domain->offset_z + (k + 0.5) * domain->hz;

                    if (is_inside_any_shape(x, y, z))
                    {
                        solid_v_idx.push_back(i * nyz + base);
                    }
                }
            }
        }

        // w: ZFace, located at (i+0.5, j+0.5, k)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                int base = j * nz + k;
                for (int i = 0; i < nx; ++i)
                {
                    double x = domain->offset_x + (i + 0.5) * domain->hx;
                    double y = domain->offset_y + (j + 0.5) * domain->hy;
                    double z = domain->offset_z + k * domain->hz;

                    if (is_inside_any_shape(x, y, z))
                    {
                        solid_w_idx.push_back(i * nyz + base);
                    }
                }
            }
        }

        solid_u_idx_map[domain] = std::move(solid_u_idx);
        solid_v_idx_map[domain] = std::move(solid_v_idx);
        solid_w_idx_map[domain] = std::move(solid_w_idx);
    }
}

void SolidVelocityFixer3D::apply()
{
    auto& geometry = *u->geometry;

    for (auto* domain : geometry.domains)
    {
        auto cache_it = domain_cache_map.find(domain);
        if (cache_it == domain_cache_map.end())
            continue;

        const auto& cache = cache_it->second;
        if (cache.u_data == nullptr || cache.v_data == nullptr || cache.w_data == nullptr)
            continue;

        // Apply u
        auto u_it = solid_u_idx_map.find(domain);
        if (u_it != solid_u_idx_map.end())
        {
            const auto& idx_vec = u_it->second;
            double* data = cache.u_data;
            for (int idx : idx_vec)
            {
                data[idx] = 0.0;
            }
        }

        // Apply v
        auto v_it = solid_v_idx_map.find(domain);
        if (v_it != solid_v_idx_map.end())
        {
            const auto& idx_vec = v_it->second;
            double* data = cache.v_data;
            for (int idx : idx_vec)
            {
                data[idx] = 0.0;
            }
        }

        // Apply w
        auto w_it = solid_w_idx_map.find(domain);
        if (w_it != solid_w_idx_map.end())
        {
            const auto& idx_vec = w_it->second;
            double* data = cache.w_data;
            for (int idx : idx_vec)
            {
                data[idx] = 0.0;
            }
        }
    }
}

size_t SolidVelocityFixer3D::get_num_solid_u_points(Domain3DUniform* domain) const
{
    auto it = solid_u_idx_map.find(domain);
    if (it == solid_u_idx_map.end())
        return 0;
    return it->second.size();
}

size_t SolidVelocityFixer3D::get_num_solid_v_points(Domain3DUniform* domain) const
{
    auto it = solid_v_idx_map.find(domain);
    if (it == solid_v_idx_map.end())
        return 0;
    return it->second.size();
}

size_t SolidVelocityFixer3D::get_num_solid_w_points(Domain3DUniform* domain) const
{
    auto it = solid_w_idx_map.find(domain);
    if (it == solid_w_idx_map.end())
        return 0;
    return it->second.size();
}

bool SolidVelocityFixer3D::has_solid_points(Domain3DUniform* domain) const
{
    return get_num_solid_u_points(domain) > 0 || get_num_solid_v_points(domain) > 0
        || get_num_solid_w_points(domain) > 0;
}

const std::vector<int>& SolidVelocityFixer3D::get_solid_u_points(Domain3DUniform* domain) const
{
    static std::vector<int> empty;
    auto it = solid_u_idx_map.find(domain);
    if (it == solid_u_idx_map.end())
        return empty;
    return it->second;
}

const std::vector<int>& SolidVelocityFixer3D::get_solid_v_points(Domain3DUniform* domain) const
{
    static std::vector<int> empty;
    auto it = solid_v_idx_map.find(domain);
    if (it == solid_v_idx_map.end())
        return empty;
    return it->second;
}

const std::vector<int>& SolidVelocityFixer3D::get_solid_w_points(Domain3DUniform* domain) const
{
    static std::vector<int> empty;
    auto it = solid_w_idx_map.find(domain);
    if (it == solid_w_idx_map.end())
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
