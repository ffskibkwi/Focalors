#include "ib_solver_3d_mirror_point.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

IBSolver3D_MirrorPoint::IBSolver3D_MirrorPoint(Variable3D* in_var, PDEBoundaryType in_boundary_type, double in_boundary_value)
    : var(in_var)
    , boundary_type(in_boundary_type)
    , boundary_value(in_boundary_value)
{
    if (var == nullptr)
        throw std::runtime_error("IBSolver3D_MirrorPoint: variable is nullptr");
    if (var->geometry == nullptr)
        throw std::runtime_error("IBSolver3D_MirrorPoint: variable has no geometry");
}

void IBSolver3D_MirrorPoint::add_shape(Shape3D* shape)
{
    shapes.push_back(shape);
}

void IBSolver3D_MirrorPoint::build()
{
    if (shapes.empty())
        throw std::runtime_error("IBSolver3D_MirrorPoint: no shapes added");

    auto& geometry = *var->geometry;

    for (auto* domain : geometry.domains)
    {
        auto* field = var->field_map.at(domain);
        int   nx    = field->get_nx();
        int   ny    = field->get_ny();
        int   nz    = field->get_nz();

        // Use unordered_map for deduplication: key = (i << 42) | (j << 21) | k
        std::unordered_map<long long, std::tuple<int, int, int>> interior_map;

        // Iterate over all grid points in this domain
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    auto [x, y, z] = get_physical_location(domain, i, j, k);

                    // Skip if this point is inside any shape
                    if (is_inside_any_shape(x, y, z))
                        continue;

                    // This is an exterior point, check its template neighbors
                    // Template can cross domain boundaries
                    // MUST check ALL template points - cannot skip once one interior point is found
                    for (int dk = -template_radius; dk <= template_radius; ++dk)
                    {
                        for (int dj = -template_radius; dj <= template_radius; ++dj)
                        {
                            for (int di = -template_radius; di <= template_radius; ++di)
                            {
                                if (di == 0 && dj == 0 && dk == 0)
                                    continue;

                                int ni = i + di;
                                int nj = j + dj;
                                int nk = k + dk;

                                // Determine which domain the template point belongs to
                                Domain3DUniform* target_domain = domain;
                                int              ti = ni;
                                int              tj = nj;
                                int              tk = nk;

                                // Check and adjust for domain boundaries in each direction
                                if (ni < 0 || ni >= nx)
                                {
                                    auto adj_it = geometry.adjacency.find(domain);
                                    if (adj_it != geometry.adjacency.end())
                                    {
                                        LocationType dir = (ni < 0) ? LocationType::XNegative : LocationType::XPositive;
                                        auto dir_it = adj_it->second.find(dir);
                                        if (dir_it != adj_it->second.end())
                                        {
                                            target_domain = dir_it->second;
                                            auto* neighbor_field = var->field_map.at(target_domain);
                                            int   neighbor_nx   = neighbor_field->get_nx();
                                            ti = (ni < 0) ? ni + neighbor_nx : ni - nx;
                                        }
                                    }
                                }

                                if (nj < 0 || nj >= ny)
                                {
                                    auto adj_it = geometry.adjacency.find(domain);
                                    if (adj_it != geometry.adjacency.end())
                                    {
                                        LocationType dir = (nj < 0) ? LocationType::YNegative : LocationType::YPositive;
                                        auto dir_it = adj_it->second.find(dir);
                                        if (dir_it != adj_it->second.end())
                                        {
                                            target_domain = dir_it->second;
                                            auto* neighbor_field = var->field_map.at(target_domain);
                                            int   neighbor_ny   = neighbor_field->get_ny();
                                            tj = (nj < 0) ? nj + neighbor_ny : nj - ny;
                                        }
                                    }
                                }

                                if (nk < 0 || nk >= nz)
                                {
                                    auto adj_it = geometry.adjacency.find(domain);
                                    if (adj_it != geometry.adjacency.end())
                                    {
                                        LocationType dir = (nk < 0) ? LocationType::ZNegative : LocationType::ZPositive;
                                        auto dir_it = adj_it->second.find(dir);
                                        if (dir_it != adj_it->second.end())
                                        {
                                            target_domain = dir_it->second;
                                            auto* neighbor_field = var->field_map.at(target_domain);
                                            int   neighbor_nz   = neighbor_field->get_nz();
                                            tk = (nk < 0) ? nk + neighbor_nz : nk - nz;
                                        }
                                    }
                                }

                                // Skip if target domain doesn't have field
                                if (var->field_map.find(target_domain) == var->field_map.end())
                                    continue;

                                // Get physical location of template point in target domain
                                auto [tx, ty, tz] = get_physical_location(target_domain, ti, tj, tk);

                                // If template point is inside any shape, mark current point as interior
                                // Must check ALL template points - add to set if found, continue checking others
                                if (is_inside_any_shape(tx, ty, tz))
                                {
                                    long long key = (static_cast<long long>(i) << 42) |
                                                    (static_cast<long long>(j) << 21) | static_cast<long long>(k);
                                    interior_map[key] = {i, j, k};
                                    // Continue checking other template points - we need to check ALL of them
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert map to vector
        auto& interior_points = interior_points_map[domain];
        interior_points.reserve(interior_map.size());
        for (auto& kv : interior_map)
        {
            interior_points.emplace_back(kv.second);
        }

        // Build mirror info for each interior point
        auto& mirror_infos = mirror_info_map[domain];
        mirror_infos.reserve(interior_points.size());

        for (const auto& [i, j, k] : interior_points)
        {
            auto [x, y, z] = get_physical_location(domain, i, j, k);

            double mirror_x, mirror_y, mirror_z, delta_l;
            find_mirror_point(x, y, z, mirror_x, mirror_y, mirror_z, delta_l);

            mirror_infos.push_back({i, j, k, mirror_x, mirror_y, mirror_z, delta_l});
        }
    }
}

void IBSolver3D_MirrorPoint::apply()
{
    for (auto* domain : var->geometry->domains)
    {
        auto* field       = var->field_map.at(domain);
        auto& mirror_list = mirror_info_map[domain];

        for (const auto& mirror : mirror_list)
        {
            int   i       = mirror.i;
            int   j       = mirror.j;
            int   k       = mirror.k;
            double phi    = (*field)(i, j, k);

            // Interpolate mirror point value
            double phi_mirror = interpolate_mirror_value(domain, mirror.mirror_x, mirror.mirror_y, mirror.mirror_z);

            double new_phi;

            if (boundary_type == PDEBoundaryType::Dirichlet)
            {
                // Dirichlet: phi + phi_mirror = 2 * BC
                new_phi = 2.0 * boundary_value - phi_mirror;
            }
            else if (boundary_type == PDEBoundaryType::Neumann)
            {
                // Neumann: (phi_mirror - phi) / delta_l = BC
                // => phi_mirror = phi + BC * delta_l
                // => phi = phi_mirror - BC * delta_l
                new_phi = phi_mirror - boundary_value * mirror.delta_l;
            }
            else
            {
                throw std::runtime_error("IBSolver3D_MirrorPoint: unsupported boundary type");
            }

            (*field)(i, j, k) = new_phi;
        }
    }
}

size_t IBSolver3D_MirrorPoint::get_num_interior_points(Domain3DUniform* domain) const
{
    auto it = interior_points_map.find(domain);
    if (it == interior_points_map.end())
        return 0;
    return it->second.size();
}

bool IBSolver3D_MirrorPoint::has_interior_points(Domain3DUniform* domain) const
{
    return interior_points_map.find(domain) != interior_points_map.end();
}

const std::vector<std::tuple<int, int, int>>& IBSolver3D_MirrorPoint::get_interior_points(Domain3DUniform* domain) const
{
    static std::vector<std::tuple<int, int, int>> empty;
    auto it = interior_points_map.find(domain);
    if (it == interior_points_map.end())
        return empty;
    return it->second;
}

std::tuple<double, double, double> IBSolver3D_MirrorPoint::get_physical_location(Domain3DUniform* domain, int i, int j,
                                                                                  int k) const
{
    double x, y, z;

    switch (var->position_type)
    {
        case VariablePositionType::Center:
            // Cell center: x = offset_x + (i + 0.5) * hx
            x = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
            y = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
            z = domain->get_offset_z() + (static_cast<double>(k) + 0.5) * domain->get_hz();
            break;
        case VariablePositionType::XFace:
            // X-face center: x = offset_x + i * hx
            x = domain->get_offset_x() + static_cast<double>(i) * domain->get_hx();
            y = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
            z = domain->get_offset_z() + (static_cast<double>(k) + 0.5) * domain->get_hz();
            break;
        case VariablePositionType::YFace:
            // Y-face center: x = offset_x + (i + 0.5) * hx
            x = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
            y = domain->get_offset_y() + static_cast<double>(j) * domain->get_hy();
            z = domain->get_offset_z() + (static_cast<double>(k) + 0.5) * domain->get_hz();
            break;
        case VariablePositionType::ZFace:
            // Z-face center: x = offset_x + (i + 0.5) * hx
            x = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
            y = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
            z = domain->get_offset_z() + static_cast<double>(k) * domain->get_hz();
            break;
        case VariablePositionType::Corner:
            // Node: x = offset_x + i * hx
            x = domain->get_offset_x() + static_cast<double>(i) * domain->get_hx();
            y = domain->get_offset_y() + static_cast<double>(j) * domain->get_hy();
            z = domain->get_offset_z() + static_cast<double>(k) * domain->get_hz();
            break;
        default:
            // Default to cell center
            x = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
            y = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
            z = domain->get_offset_z() + (static_cast<double>(k) + 0.5) * domain->get_hz();
            break;
    }

    return {x, y, z};
}

bool IBSolver3D_MirrorPoint::is_inside_any_shape(double x, double y, double z) const
{
    for (const auto* shape : shapes)
    {
        if (shape->is_inside(x, y, z))
            return true;
    }
    return false;
}

void IBSolver3D_MirrorPoint::find_mirror_point(double x, double y, double z, double& mirror_x, double& mirror_y,
                                              double& mirror_z, double& delta_l) const
{
    // Find the closest point on the shape surface
    // For multiple shapes, use the closest one
    double min_dist = 1e100;
    double closest_x = 0, closest_y = 0, closest_z = 0;
    double normal_x = 0, normal_y = 0, normal_z = 0;

    for (const auto* shape : shapes)
    {
        if (!shape->is_inside(x, y, z))
            continue;

        auto [surf_x, surf_y, surf_z] = shape->get_closest_point(x, y, z);

        // Vector from interior point to surface
        double dx = surf_x - x;
        double dy = surf_y - y;
        double dz = surf_z - z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < min_dist)
        {
            min_dist = dist;
            closest_x = surf_x;
            closest_y = surf_y;
            closest_z = surf_z;

            if (dist > 1e-14)
            {
                normal_x = dx / dist;
                normal_y = dy / dist;
                normal_z = dz / dist;
            }
            else
            {
                // Interior point is at the center, use radial direction
                normal_x = 1.0;
                normal_y = 0.0;
                normal_z = 0.0;
            }
        }
    }

    if (min_dist >= 1e100)
    {
        // Should not happen if called correctly
        mirror_x = x;
        mirror_y = y;
        mirror_z = z;
        delta_l = 0.0;
        return;
    }

    // Mirror point is on the other side of the boundary
    // Extend from the closest point along the outward normal
    mirror_x = x + normal_x * min_dist;
    mirror_y = y + normal_y * min_dist;
    mirror_z = z + normal_z * min_dist;
    delta_l = min_dist;
}

double IBSolver3D_MirrorPoint::interpolate_mirror_value(Domain3DUniform* domain, double mirror_x, double mirror_y,
                                                      double mirror_z) const
{
    auto* field = var->field_map.at(domain);
    int   nx    = field->get_nx();
    int   ny    = field->get_ny();
    int   nz    = field->get_nz();

    // Get domain physical range
    double hx = domain->get_hx();
    double hy = domain->get_hy();
    double hz = domain->get_hz();
    double ox = domain->get_offset_x();
    double oy = domain->get_offset_y();
    double oz = domain->get_offset_z();
    double x_max = ox + nx * hx;
    double y_max = oy + ny * hy;
    double z_max = oz + nz * hz;

    // Determine which domain to use for interpolation
    // The mirror point may be in a neighbor domain
    Domain3DUniform* target_domain = domain;
    double           target_mirror_x = mirror_x;
    double           target_mirror_y = mirror_y;
    double           target_mirror_z = mirror_z;

    // Check if mirror point is outside current domain
    bool is_outside = false;
    if (mirror_x < ox || mirror_x > x_max || mirror_y < oy || mirror_y > y_max || mirror_z < oz || mirror_z > z_max)
    {
        is_outside = true;
    }

    if (is_outside)
    {
        // Find the appropriate neighbor domain based on mirror point position
        auto adj_it = var->geometry->adjacency.find(domain);
        if (adj_it != var->geometry->adjacency.end())
        {
            // Determine which direction(s) the mirror point is outside
            bool outside_x_neg = mirror_x < ox;
            bool outside_x_pos = mirror_x > x_max;
            bool outside_y_neg = mirror_y < oy;
            bool outside_y_pos = mirror_y > y_max;
            bool outside_z_neg = mirror_z < oz;
            bool outside_z_pos = mirror_z > z_max;

            Domain3DUniform* found_domain = nullptr;

            // Check corner/edge cases systematically
            // First try face neighbors
            if (!found_domain)
            {
                if (outside_x_neg)
                {
                    auto dir_it = adj_it->second.find(LocationType::XNegative);
                    if (dir_it != adj_it->second.end())
                        found_domain = dir_it->second;
                }
                else if (outside_x_pos)
                {
                    auto dir_it = adj_it->second.find(LocationType::XPositive);
                    if (dir_it != adj_it->second.end())
                        found_domain = dir_it->second;
                }
                else if (outside_y_neg)
                {
                    auto dir_it = adj_it->second.find(LocationType::YNegative);
                    if (dir_it != adj_it->second.end())
                        found_domain = dir_it->second;
                }
                else if (outside_y_pos)
                {
                    auto dir_it = adj_it->second.find(LocationType::YPositive);
                    if (dir_it != adj_it->second.end())
                        found_domain = dir_it->second;
                }
                else if (outside_z_neg)
                {
                    auto dir_it = adj_it->second.find(LocationType::ZNegative);
                    if (dir_it != adj_it->second.end())
                        found_domain = dir_it->second;
                }
                else if (outside_z_pos)
                {
                    auto dir_it = adj_it->second.find(LocationType::ZPositive);
                    if (dir_it != adj_it->second.end())
                        found_domain = dir_it->second;
                }
            }

            // For edge and corner cases, we need to traverse through multiple domains
            // This is simplified: if multiple directions are outside, we try sequential lookup
            // A full implementation would need proper multi-dimensional adjacency

            if (found_domain != nullptr && var->field_map.find(found_domain) != var->field_map.end())
            {
                target_domain = found_domain;
                // Update field pointer and domain parameters
                field = var->field_map.at(target_domain);
                nx = field->get_nx();
                ny = field->get_ny();
                nz = field->get_nz();
                hx = target_domain->get_hx();
                hy = target_domain->get_hy();
                hz = target_domain->get_hz();
                ox = target_domain->get_offset_x();
                oy = target_domain->get_offset_y();
                oz = target_domain->get_offset_z();
            }
        }
    }

    // Convert physical coordinates to grid indices based on position type
    double gi, gj, gk;
    switch (var->position_type)
    {
        case VariablePositionType::Center:
            gi = (target_mirror_x - ox) / hx - 0.5;
            gj = (target_mirror_y - oy) / hy - 0.5;
            gk = (target_mirror_z - oz) / hz - 0.5;
            break;
        case VariablePositionType::XFace:
            gi = (target_mirror_x - ox) / hx;
            gj = (target_mirror_y - oy) / hy - 0.5;
            gk = (target_mirror_z - oz) / hz - 0.5;
            break;
        case VariablePositionType::YFace:
            gi = (target_mirror_x - ox) / hx - 0.5;
            gj = (target_mirror_y - oy) / hy;
            gk = (target_mirror_z - oz) / hz - 0.5;
            break;
        case VariablePositionType::ZFace:
            gi = (target_mirror_x - ox) / hx - 0.5;
            gj = (target_mirror_y - oy) / hy - 0.5;
            gk = (target_mirror_z - oz) / hz;
            break;
        case VariablePositionType::Corner:
            gi = (target_mirror_x - ox) / hx;
            gj = (target_mirror_y - oy) / hy;
            gk = (target_mirror_z - oz) / hz;
            break;
        default:
            gi = (target_mirror_x - ox) / hx - 0.5;
            gj = (target_mirror_y - oy) / hy - 0.5;
            gk = (target_mirror_z - oz) / hz - 0.5;
            break;
    }

    int i0 = static_cast<int>(std::floor(gi));
    int j0 = static_cast<int>(std::floor(gj));
    int k0 = static_cast<int>(std::floor(gk));
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    int k1 = k0 + 1;

    double sx = gi - static_cast<double>(i0);
    double sy = gj - static_cast<double>(j0);
    double sz = gk - static_cast<double>(k0);

    // Helper lambda to get value with boundary handling for target domain
    auto get_val = [&](int i, int j, int k) -> double {
        // Check if within target domain
        if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            return (*field)(i, j, k);
        }

        // Need to handle out-of-bounds - use buffer if available
        auto buffer_it = var->buffer_map.find(target_domain);
        if (buffer_it != var->buffer_map.end())
        {
            const auto& buffer = buffer_it->second;

            // X negative buffer
            if (i < 0 && j >= 0 && j < ny && k >= 0 && k < nz)
            {
                auto buf_it = buffer.find(LocationType::XNegative);
                if (buf_it != buffer.end())
                    return buf_it->second->operator()(j, k);
            }
            // X positive buffer
            if (i >= nx && j >= 0 && j < ny && k >= 0 && k < nz)
            {
                auto buf_it = buffer.find(LocationType::XPositive);
                if (buf_it != buffer.end())
                    return buf_it->second->operator()(j, k);
            }
            // Y negative buffer
            if (j < 0 && i >= 0 && i < nx && k >= 0 && k < nz)
            {
                auto buf_it = buffer.find(LocationType::YNegative);
                if (buf_it != buffer.end())
                    return buf_it->second->operator()(i, k);
            }
            // Y positive buffer
            if (j >= ny && i >= 0 && i < nx && k >= 0 && k < nz)
            {
                auto buf_it = buffer.find(LocationType::YPositive);
                if (buf_it != buffer.end())
                    return buf_it->second->operator()(i, k);
            }
            // Z negative buffer
            if (k < 0 && i >= 0 && i < nx && j >= 0 && j < ny)
            {
                auto buf_it = buffer.find(LocationType::ZNegative);
                if (buf_it != buffer.end())
                    return buf_it->second->operator()(i, j);
            }
            // Z positive buffer
            if (k >= nz && i >= 0 && i < nx && j >= 0 && j < ny)
            {
                auto buf_it = buffer.find(LocationType::ZPositive);
                if (buf_it != buffer.end())
                    return buf_it->second->operator()(i, j);
            }
        }

        // Fall back to single-side extrapolation (clamp to boundary)
        if (i < 0)
            i = 0;
        if (i >= nx)
            i = nx - 1;
        if (j < 0)
            j = 0;
        if (j >= ny)
            j = ny - 1;
        if (k < 0)
            k = 0;
        if (k >= nz)
            k = nz - 1;

        return (*field)(i, j, k);
    };

    // Get the eight corner values for trilinear interpolation
    double v000 = get_val(i0, j0, k0);
    double v100 = get_val(i1, j0, k0);
    double v010 = get_val(i0, j1, k0);
    double v110 = get_val(i1, j1, k0);
    double v001 = get_val(i0, j0, k1);
    double v101 = get_val(i1, j0, k1);
    double v011 = get_val(i0, j1, k1);
    double v111 = get_val(i1, j1, k1);

    // Trilinear interpolation
    double result = (1.0 - sx) * (1.0 - sy) * (1.0 - sz) * v000 + sx * (1.0 - sy) * (1.0 - sz) * v100 +
                    (1.0 - sx) * sy * (1.0 - sz) * v010 + sx * sy * (1.0 - sz) * v110 +
                    (1.0 - sx) * (1.0 - sy) * sz * v001 + sx * (1.0 - sy) * sz * v101 +
                    (1.0 - sx) * sy * sz * v011 + sx * sy * sz * v111;

    return result;
}

double IBSolver3D_MirrorPoint::get_field_value(Domain3DUniform* domain, int i, int j, int k) const
{
    auto* field = var->field_map.at(domain);
    int   nx    = field->get_nx();
    int   ny    = field->get_ny();
    int   nz    = field->get_nz();

    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        return (*field)(i, j, k);
    }

    // Use buffer if available
    auto buffer_it = var->buffer_map.find(domain);
    if (buffer_it != var->buffer_map.end())
    {
        const auto& buffer = buffer_it->second;

        if (i < 0 && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            auto buf_it = buffer.find(LocationType::XNegative);
            if (buf_it != buffer.end())
                return buf_it->second->operator()(j, k);
        }
        if (i >= nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            auto buf_it = buffer.find(LocationType::XPositive);
            if (buf_it != buffer.end())
                return buf_it->second->operator()(j, k);
        }
        if (j < 0 && i >= 0 && i < nx && k >= 0 && k < nz)
        {
            auto buf_it = buffer.find(LocationType::YNegative);
            if (buf_it != buffer.end())
                return buf_it->second->operator()(i, k);
        }
        if (j >= ny && i >= 0 && i < nx && k >= 0 && k < nz)
        {
            auto buf_it = buffer.find(LocationType::YPositive);
            if (buf_it != buffer.end())
                return buf_it->second->operator()(i, k);
        }
        if (k < 0 && i >= 0 && i < nx && j >= 0 && j < ny)
        {
            auto buf_it = buffer.find(LocationType::ZNegative);
            if (buf_it != buffer.end())
                return buf_it->second->operator()(i, j);
        }
        if (k >= nz && i >= 0 && i < nx && j >= 0 && j < ny)
        {
            auto buf_it = buffer.find(LocationType::ZPositive);
            if (buf_it != buffer.end())
                return buf_it->second->operator()(i, j);
        }
    }

    // Fallback to boundary value
    return boundary_value;
}
