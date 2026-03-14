#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "ibm_MirrorPoint/ib_solver_2d_mirror_point.h"
#include "poisson_base/base/math/compare.h"

#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

// 简单初始化速度场：u = sin(x), v = cos(y)
static void init_velocity_sin(Variable2D& u, Variable2D& v)
{
    u.set_value([](double x, double /*y*/) { return std::sin(x); });
    v.set_value([](double /*x*/, double y) { return std::cos(y); });
}

// 将某个 Variable2D 的所有域的场复制到另一个 Variable2D（假定几何一致）
static void copy_velocity(const Variable2D& src_u, const Variable2D& src_v, Variable2D& dst_u, Variable2D& dst_v)
{
    for (auto& kv : src_u.field_map)
    {
        Domain2DUniform* d     = kv.first;
        const field2&    src_f = *kv.second;
        field2&          dst_f = *dst_u.field_map[d];
        for (int i = 0; i < src_f.get_nx(); ++i)
            for (int j = 0; j < src_f.get_ny(); ++j)
                dst_f(i, j) = src_f(i, j);
    }

    for (auto& kv : src_v.field_map)
    {
        Domain2DUniform* d     = kv.first;
        const field2&    src_f = *kv.second;
        field2&          dst_f = *dst_v.field_map[d];
        for (int i = 0; i < src_f.get_nx(); ++i)
            for (int j = 0; j < src_f.get_ny(); ++j)
                dst_f(i, j) = src_f(i, j);
    }
}

// 生成圆柱边界上的采样点
static std::vector<std::tuple<double, double>>
generate_cylinder_surface_points(double cx, double cy, double r, int n_points)
{
    std::vector<std::tuple<double, double>> points;
    for (int i = 0; i < n_points; i++)
    {
        double theta = 2.0 * M_PI * i / n_points;
        double x     = cx + r * std::cos(theta);
        double y     = cy + r * std::sin(theta);
        points.push_back({x, y});
    }
    return points;
}

// 打印几何信息
static void print_geometry_info(const Geometry2D& geo, const std::string& label)
{
    std::cout << "[" << label << "] Geometry info:\n";
    std::cout << "    Domain count: " << geo.domains.size() << "\n";
    for (auto* d : geo.domains)
    {
        std::cout << "    Domain '" << d->name << "':\n";
        std::cout << "        Offset: (" << d->get_offset_x() << "," << d->get_offset_y() << ")\n";
        std::cout << "        Size: (" << d->get_lx() << "," << d->get_ly() << ")\n";
        std::cout << "        Grid: (" << d->get_nx() << "," << d->get_ny() << ")\n";
        std::cout << "        Spacing: (" << d->get_hx() << "," << d->get_hy() << ")\n";
    }
}

// 在给定物理坐标 (x,y) 上，从 Variable2D 中采样 u(x,y)（x-face centered）/ v(x,y)（y-face centered）
static bool sample_u_at(const Variable2D& u, double x, double y, double& value)
{
    Geometry2D* geo = u.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double hx = d->get_hx();
        double hy = d->get_hy();

        // u 是 x-face centered: x = ox + i*hx, y = oy + (j+0.5)*hy
        double eps = 1e-10;
        if (x < ox - eps || x > ox + lx + eps || y < oy - eps || y > oy + ly + eps)
            continue;

        int i = static_cast<int>(std::round((x - ox) / hx));
        int j = static_cast<int>(std::round((y - oy) / hy - 0.5));

        auto& f = *u.field_map.at(d);
        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny())
        {
            value = f(i, j);
            return true;
        }
    }
    return false;
}

static bool sample_v_at(const Variable2D& v, double x, double y, double& value)
{
    Geometry2D* geo = v.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double hx = d->get_hx();
        double hy = d->get_hy();

        // v 是 y-face centered: x = ox + (i+0.5)*hx, y = oy + j*hy
        double eps = 1e-10;
        if (x < ox - eps || x > ox + lx + eps || y < oy - eps || y > oy + ly + eps)
            continue;

        int i = static_cast<int>(std::round((x - ox) / hx - 0.5));
        int j = static_cast<int>(std::round((y - oy) / hy));

        auto& f = *v.field_map.at(d);
        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny())
        {
            value = f(i, j);
            return true;
        }
    }
    return false;
}

// 采样并对比IBM前后的速度值，输出不相等的个数
static size_t sample_and_compare_velocity(const Variable2D&  u_before,
                                          const Variable2D&  v_before,
                                          const Variable2D&  u_after,
                                          const Variable2D&  v_after,
                                          double             cylinder_cx,
                                          double             cylinder_cy,
                                          double             cylinder_r,
                                          double            abs_eps,
                                          double            rel_eps)
{
    auto sample_points = generate_cylinder_surface_points(cylinder_cx, cylinder_cy, cylinder_r, 10);
    size_t mismatch_count = 0;

    for (const auto& [x, y] : sample_points)
    {
        double u_before_val = 0.0, v_before_val = 0.0;
        double u_after_val = 0.0, v_after_val = 0.0;
        bool   u_ok_before = sample_u_at(u_before, x, y, u_before_val);
        bool   v_ok_before = sample_v_at(v_before, x, y, v_before_val);
        bool   u_ok_after = sample_u_at(u_after, x, y, u_after_val);
        bool   v_ok_after = sample_v_at(v_after, x, y, v_after_val);

        if (u_ok_before && u_ok_after && !approximatelyEqualAbsRel(u_before_val, u_after_val, abs_eps, rel_eps))
            mismatch_count++;
        if (v_ok_before && v_ok_after && !approximatelyEqualAbsRel(v_before_val, v_after_val, abs_eps, rel_eps))
            mismatch_count++;
    }

    std::cout << "    IBM before/after comparison: " << mismatch_count << " mismatches out of "
              << sample_points.size() * 2 << " samples\n";
    return mismatch_count;
}

// 真正对比：单域 (u_single,v_single) 与 多域 (u_multi,v_multi) 在同一物理位置上的速度
// 输出相等和不等的个数
static void compare_velocity_fields_phys(const Variable2D& u_single,
                                         const Variable2D& v_single,
                                         const Variable2D& u_multi,
                                         const Variable2D& v_multi,
                                         double            abs_eps,
                                         double            rel_eps)
{
    std::cout << "[MirrorPoint 2D] Comparing velocities between single-domain and two-domain setups (by physical coords)...\n";

    Geometry2D* geo_single = u_single.geometry;
    size_t equal_count = 0;
    size_t mismatch_count = 0;

    for (auto* d : geo_single->domains)
    {
        auto& fu = *u_single.field_map.at(d);
        auto& fv = *v_single.field_map.at(d);

        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double hx = d->get_hx();
        double hy = d->get_hy();

        // u: x-face centered
        for (int i = 0; i < fu.get_nx(); ++i)
        {
            for (int j = 0; j < fu.get_ny(); ++j)
            {
                double x = ox + i * hx;
                double y = oy + (j + 0.5) * hy;

                double a = fu(i, j);
                double b = 0.0;
                if (sample_u_at(u_multi, x, y, b))
                {
                    if (approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                        equal_count++;
                    else
                        mismatch_count++;
                }
            }
        }

        // v: y-face centered
        for (int i = 0; i < fv.get_nx(); ++i)
        {
            for (int j = 0; j < fv.get_ny(); ++j)
            {
                double x = ox + (i + 0.5) * hx;
                double y = oy + j * hy;

                double a = fv(i, j);
                double b = 0.0;
                if (sample_v_at(v_multi, x, y, b))
                {
                    if (approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                        equal_count++;
                    else
                        mismatch_count++;
                }
            }
        }
    }

    std::cout << "    Single vs Multi domain comparison: " << equal_count << " equal, " << mismatch_count
              << " mismatched\n";
}

int main(int /*argc*/, char* /*argv*/[])
{
    // 统一配置
    constexpr int    NX_TOTAL = 8;
    constexpr int    NY_TOTAL = 8;
    constexpr double LX       = 1.0;
    constexpr double LY       = 1.0;

    const double hx = LX / NX_TOTAL;
    const double hy = LY / NY_TOTAL;

    // MirrorPoint 圆柱：在整个正方形中间放一个圆柱
    const double cx = 0.5 * LX;
    const double cy = 0.5 * LY;
    const double r  = 0.15 * LX;

    std::cout << "=== MirrorPoint 2D Cylinder Validation ===\n";
    std::cout << "Computational domain: [" << 0 << "," << LX << "] x [" << 0 << "," << LY << "]\n";
    std::cout << "Grid size: " << NX_TOTAL << " x " << NY_TOTAL << "\n";
    std::cout << "Grid spacing: (" << hx << "," << hy << ")\n";
    std::cout << "MirrorPoint cylinder: center=(" << cx << "," << cy << "), radius=" << r << "\n";
    std::cout << "Boundary condition: Dirichlet (no-slip, BC=0)\n\n";

    // ------------------ 情况 1：单个 domain ------------------
    Geometry2D geo_single;

    Domain2DUniform d_single(NX_TOTAL, NY_TOTAL, LX, LY, "Single");
    geo_single.add_domain(&d_single);
    geo_single.set_global_spatial_step(hx, hy);

    print_geometry_info(geo_single, "Case 1: Single Domain");

    Variable2D u_single("u_single"), v_single("v_single");
    u_single.set_geometry(geo_single);
    v_single.set_geometry(geo_single);

    field2 u_single_f, v_single_f;
    u_single.set_x_edge_field(&d_single, u_single_f);
    v_single.set_y_edge_field(&d_single, v_single_f);

    init_velocity_sin(u_single, v_single);

    // 创建 MirrorPoint 求解器
    Circle circle(cx, cy, r);

    // 为 u 创建 solver（u 在 x-face 上）
    IBSolver2D_MirrorPoint solver_u_single(&u_single, PDEBoundaryType::Dirichlet, 0.0);
    solver_u_single.add_shape(&circle);
    solver_u_single.build();

    // 为 v 创建 solver（v 在 y-face 上）
    IBSolver2D_MirrorPoint solver_v_single(&v_single, PDEBoundaryType::Dirichlet, 0.0);
    solver_v_single.add_shape(&circle);
    solver_v_single.build();

    // Keep a copy before IBM solve for comparison
    Variable2D u_single_before("u_single_before"), v_single_before("v_single_before");
    u_single_before.set_geometry(geo_single);
    v_single_before.set_geometry(geo_single);
    field2 u_single_before_f, v_single_before_f;
    u_single_before.set_x_edge_field(&d_single, u_single_before_f);
    v_single_before.set_y_edge_field(&d_single, v_single_before_f);
    copy_velocity(u_single, v_single, u_single_before, v_single_before);

    // Apply MirrorPoint
    solver_u_single.apply();
    solver_v_single.apply();

    // Compare before/after
    sample_and_compare_velocity(u_single_before, v_single_before, u_single, v_single, cx, cy, r, 1e-10, 1e-8);

    // ------------------ 情况 2：两个 domain 拼接 ------------------
    Geometry2D geo_multi;

    Domain2DUniform d_left(NX_TOTAL / 2, NY_TOTAL, LX / 2.0, LY, "Left");
    Domain2DUniform d_right(NX_TOTAL / 2, NY_TOTAL, LX / 2.0, LY, "Right");

    geo_multi.add_domain({&d_left, &d_right});
    geo_multi.connect(&d_left, LocationType::XPositive, &d_right);
    geo_multi.set_global_spatial_step(hx, hy);

    // Set the axis to automatically compute offsets
    geo_multi.axis(&d_left, LocationType::XNegative);

    print_geometry_info(geo_multi, "Case 2: Multi Domain");

    Variable2D u_multi("u_multi"), v_multi("v_multi");
    u_multi.set_geometry(geo_multi);
    v_multi.set_geometry(geo_multi);

    field2 u_left_f, u_right_f;
    field2 v_left_f, v_right_f;

    u_multi.set_x_edge_field(&d_left, u_left_f);
    u_multi.set_x_edge_field(&d_right, u_right_f);
    v_multi.set_y_edge_field(&d_left, v_left_f);
    v_multi.set_y_edge_field(&d_right, v_right_f);

    init_velocity_sin(u_multi, v_multi);

    // 为 u 创建 solver
    IBSolver2D_MirrorPoint solver_u_multi(&u_multi, PDEBoundaryType::Dirichlet, 0.0);
    solver_u_multi.add_shape(&circle);
    solver_u_multi.build();

    // 为 v 创建 solver
    IBSolver2D_MirrorPoint solver_v_multi(&v_multi, PDEBoundaryType::Dirichlet, 0.0);
    solver_v_multi.add_shape(&circle);
    solver_v_multi.build();

    // Keep a copy before IBM solve for comparison
    Variable2D u_multi_before("u_multi_before"), v_multi_before("v_multi_before");
    u_multi_before.set_geometry(geo_multi);
    v_multi_before.set_geometry(geo_multi);
    field2 u_multi_before_left_f, v_multi_before_left_f;
    field2 u_multi_before_right_f, v_multi_before_right_f;
    u_multi_before.set_x_edge_field(&d_left, u_multi_before_left_f);
    u_multi_before.set_x_edge_field(&d_right, u_multi_before_right_f);
    v_multi_before.set_y_edge_field(&d_left, v_multi_before_left_f);
    v_multi_before.set_y_edge_field(&d_right, v_multi_before_right_f);
    copy_velocity(u_multi, v_multi, u_multi_before, v_multi_before);

    // Apply MirrorPoint
    solver_u_multi.apply();
    solver_v_multi.apply();

    // Compare before/after
    sample_and_compare_velocity(u_multi_before, v_multi_before, u_multi, v_multi, cx, cy, r, 1e-10, 1e-8);

    // 按物理坐标真正对比单域结果和双域结果
    compare_velocity_fields_phys(u_single, v_single, u_multi, v_multi, 1e-10, 1e-8);

    std::cout << "[MirrorPoint 2D] Validation finished.\n";

    return 0;
}
