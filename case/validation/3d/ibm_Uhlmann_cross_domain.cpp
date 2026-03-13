#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "ibm_Uhlmann/ib_velocity_solver_3d_Uhlmann.h"
#include "particle/particles_coordinate_map_3d.h"
#include "poisson_base/base/math/compare.h"

#include <cmath>
#include <iostream>

// 初始化三维速度场：u = sin(x), v = cos(y), w = sin(z)
static void init_velocity_sin(Variable3D& u, Variable3D& v, Variable3D& w)
{
    u.set_value([](double x, double /*y*/, double /*z*/) { return std::sin(x); });
    v.set_value([](double /*x*/, double y, double /*z*/) { return std::cos(y); });
    w.set_value([](double /*x*/, double /*y*/, double z) { return std::sin(z); });
}

// 生成球体边界上的采样点
static std::vector<std::tuple<double, double, double>>
generate_sphere_surface_points(double cx, double cy, double cz, double r, int n_points)
{
    std::vector<std::tuple<double, double, double>> points;
    // 使用斐波那契球面分布生成均匀分布的点
    double phi = M_PI * (3.0 - std::sqrt(5.0)); // 黄金角

    for (int i = 0; i < n_points; i++)
    {
        double y      = 1.0 - (i / static_cast<double>(n_points - 1)) * 2.0; // y 从 1 到 -1
        double radius = std::sqrt(1.0 - y * y);                              // 半径在 y 处

        double theta = phi * i; // 黄金角增量

        double x = std::cos(theta) * radius;
        double z = std::sin(theta) * radius;

        // 缩放到球体表面
        points.push_back({cx + x * r, cy + y * r, cz + z * r});
    }
    return points;
}

// 打印几何信息
static void print_geometry_info(const Geometry3D& geo, const std::string& label)
{
    std::cout << "[" << label << "] Geometry info:\n";
    std::cout << "    Domain count: " << geo.domains.size() << "\n";
    for (auto* d : geo.domains)
    {
        std::cout << "    Domain '" << d->name << "':\n";
        std::cout << "        Offset: (" << d->get_offset_x() << "," << d->get_offset_y() << "," << d->get_offset_z()
                  << ")\n";
        std::cout << "        Size: (" << d->get_lx() << "," << d->get_ly() << "," << d->get_lz() << ")\n";
        std::cout << "        Grid: (" << d->get_nx() << "," << d->get_ny() << "," << d->get_nz() << ")\n";
        std::cout << "        Spacing: (" << d->get_hx() << "," << d->get_hy() << "," << d->get_hz() << ")\n";
    }
}

// 在给定物理坐标 (x,y,z) 上，从 Variable3D 中采样 u/v/w（面心变量）
static bool sample_u_at(const Variable3D& u, double x, double y, double z, double& value)
{
    Geometry3D* geo = u.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // u: x-face centered, x = ox + i*hx, y = oy + (j+0.5)*hy, z = oz + (k+0.5)*hz
        // 对于边界点，需要特殊处理：允许 x 恰好等于右边界（因为 u 存储在边界上）
        double eps = 1e-10;
        if (x < ox - eps || x > ox + lx + eps || y < oy - eps || y > oy + ly + eps || z < oz - eps || z > oz + lz + eps)
            continue;

        int i = static_cast<int>(std::round((x - ox) / hx));
        int j = static_cast<int>(std::round((y - oy) / hy - 0.5));
        int k = static_cast<int>(std::round((z - oz) / hz - 0.5));

        auto& f = *u.field_map.at(d);
        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            return true;
        }
    }
    return false;
}

static bool sample_v_at(const Variable3D& v, double x, double y, double z, double& value)
{
    Geometry3D* geo = v.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // v: y-face centered, x = ox + (i+0.5)*hx, y = oy + j*hy, z = oz + (k+0.5)*hz
        double eps = 1e-10;
        if (x < ox - eps || x > ox + lx + eps || y < oy - eps || y > oy + ly + eps || z < oz - eps || z > oz + lz + eps)
            continue;

        int i = static_cast<int>(std::round((x - ox) / hx - 0.5));
        int j = static_cast<int>(std::round((y - oy) / hy));
        int k = static_cast<int>(std::round((z - oz) / hz - 0.5));

        auto& f = *v.field_map.at(d);
        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            return true;
        }
    }
    return false;
}

static bool sample_w_at(const Variable3D& w, double x, double y, double z, double& value)
{
    Geometry3D* geo = w.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // w: z-face centered, x = ox + (i+0.5)*hx, y = oy + (j+0.5)*hy, z = oz + k*hz
        double eps = 1e-10;
        if (x < ox - eps || x > ox + lx + eps || y < oy - eps || y > oy + ly + eps || z < oz - eps || z > oz + lz + eps)
            continue;

        int i = static_cast<int>(std::round((x - ox) / hx - 0.5));
        int j = static_cast<int>(std::round((y - oy) / hy - 0.5));
        int k = static_cast<int>(std::round((z - oz) / hz));

        auto& f = *w.field_map.at(d);
        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            return true;
        }
    }
    return false;
}

// 将某个 Variable3D 的所有域的场复制到另一个 Variable3D（假定几何一致）
static void copy_velocity(const Variable3D& src_u, const Variable3D& src_v, const Variable3D& src_w,
                          Variable3D& dst_u, Variable3D& dst_v, Variable3D& dst_w)
{
    for (auto& kv : src_u.field_map)
    {
        Domain3DUniform* d     = kv.first;
        const field3&    src_f = *kv.second;
        field3&          dst_f = *dst_u.field_map.at(d);
        for (int i = 0; i < src_f.get_nx(); ++i)
            for (int j = 0; j < src_f.get_ny(); ++j)
                for (int k = 0; k < src_f.get_nz(); ++k)
                    dst_f(i, j, k) = src_f(i, j, k);
    }

    for (auto& kv : src_v.field_map)
    {
        Domain3DUniform* d     = kv.first;
        const field3&    src_f = *kv.second;
        field3&          dst_f = *dst_v.field_map.at(d);
        for (int i = 0; i < src_f.get_nx(); ++i)
            for (int j = 0; j < src_f.get_ny(); ++j)
                for (int k = 0; k < src_f.get_nz(); ++k)
                    dst_f(i, j, k) = src_f(i, j, k);
    }

    for (auto& kv : src_w.field_map)
    {
        Domain3DUniform* d     = kv.first;
        const field3&    src_f = *kv.second;
        field3&          dst_f = *dst_w.field_map.at(d);
        for (int i = 0; i < src_f.get_nx(); ++i)
            for (int j = 0; j < src_f.get_ny(); ++j)
                for (int k = 0; k < src_f.get_nz(); ++k)
                    dst_f(i, j, k) = src_f(i, j, k);
    }
}

// 采样并对比IBM前后的速度值，输出不相等的个数
static size_t sample_and_compare_velocity(const Variable3D&  u_before,
                                          const Variable3D&  v_before,
                                          const Variable3D&  w_before,
                                          const Variable3D&  u_after,
                                          const Variable3D&  v_after,
                                          const Variable3D&  w_after,
                                          double             sphere_cx,
                                          double             sphere_cy,
                                          double             sphere_cz,
                                          double             sphere_r,
                                          double            abs_eps,
                                          double            rel_eps)
{
    auto sample_points = generate_sphere_surface_points(sphere_cx, sphere_cy, sphere_cz, sphere_r, 10);
    size_t mismatch_count = 0;

    for (const auto& [x, y, z] : sample_points)
    {
        double u_before_val = 0.0, v_before_val = 0.0, w_before_val = 0.0;
        double u_after_val = 0.0, v_after_val = 0.0, w_after_val = 0.0;
        bool   u_ok_before = sample_u_at(u_before, x, y, z, u_before_val);
        bool   v_ok_before = sample_v_at(v_before, x, y, z, v_before_val);
        bool   w_ok_before = sample_w_at(w_before, x, y, z, w_before_val);
        bool   u_ok_after = sample_u_at(u_after, x, y, z, u_after_val);
        bool   v_ok_after = sample_v_at(v_after, x, y, z, v_after_val);
        bool   w_ok_after = sample_w_at(w_after, x, y, z, w_after_val);

        if (u_ok_before && u_ok_after && !approximatelyEqualAbsRel(u_before_val, u_after_val, abs_eps, rel_eps))
            mismatch_count++;
        if (v_ok_before && v_ok_after && !approximatelyEqualAbsRel(v_before_val, v_after_val, abs_eps, rel_eps))
            mismatch_count++;
        if (w_ok_before && w_ok_after && !approximatelyEqualAbsRel(w_before_val, w_after_val, abs_eps, rel_eps))
            mismatch_count++;
    }

    std::cout << "    IBM before/after comparison: " << mismatch_count << " mismatches out of "
              << sample_points.size() * 3 << " samples\n";
    return mismatch_count;
}

// 真正对比：单域 (u_single,v_single,w_single) 与 多域 (u_multi,v_multi,w_multi)
// 输出相等和不等的个数
static void compare_velocity_fields_phys(const Variable3D& u_single,
                                         const Variable3D& v_single,
                                         const Variable3D& w_single,
                                         const Variable3D& u_multi,
                                         const Variable3D& v_multi,
                                         const Variable3D& w_multi,
                                         double            abs_eps,
                                         double            rel_eps)
{
    std::cout << "[IBM 3D] Comparing velocities between single-domain and two-domain setups (by physical coords)...\n";

    Geometry3D* geo_single = u_single.geometry;
    size_t equal_count = 0;
    size_t mismatch_count = 0;

    for (auto* d : geo_single->domains)
    {
        auto& fu = *u_single.field_map.at(d);
        auto& fv = *v_single.field_map.at(d);
        auto& fw = *w_single.field_map.at(d);

        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // u: x-face centered
        for (int i = 0; i < fu.get_nx(); ++i)
        {
            for (int j = 0; j < fu.get_ny(); ++j)
            {
                for (int k = 0; k < fu.get_nz(); ++k)
                {
                    double x = ox + i * hx;
                    double y = oy + (j + 0.5) * hy;
                    double z = oz + (k + 0.5) * hz;

                    double a = fu(i, j, k);
                    double b = 0.0;
                    if (sample_u_at(u_multi, x, y, z, b))
                    {
                        if (approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                            equal_count++;
                        else
                            mismatch_count++;
                    }
                }
            }
        }

        // v: y-face centered
        for (int i = 0; i < fv.get_nx(); ++i)
        {
            for (int j = 0; j < fv.get_ny(); ++j)
            {
                for (int k = 0; k < fv.get_nz(); ++k)
                {
                    double x = ox + (i + 0.5) * hx;
                    double y = oy + j * hy;
                    double z = oz + (k + 0.5) * hz;

                    double a = fv(i, j, k);
                    double b = 0.0;
                    if (sample_v_at(v_multi, x, y, z, b))
                    {
                        if (approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                            equal_count++;
                        else
                            mismatch_count++;
                    }
                }
            }
        }

        // w: z-face centered
        for (int i = 0; i < fw.get_nx(); ++i)
        {
            for (int j = 0; j < fw.get_ny(); ++j)
            {
                for (int k = 0; k < fw.get_nz(); ++k)
                {
                    double x = ox + (i + 0.5) * hx;
                    double y = oy + (j + 0.5) * hy;
                    double z = oz + k * hz;

                    double a = fw(i, j, k);
                    double b = 0.0;
                    if (sample_w_at(w_multi, x, y, z, b))
                    {
                        if (approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                            equal_count++;
                        else
                            mismatch_count++;
                    }
                }
            }
        }
    }

    std::cout << "    Single vs Multi domain comparison: " << equal_count << " equal, " << mismatch_count
              << " mismatched\n";
}

int main(int /*argc*/, char* /*argv*/[])
{
    constexpr int    NX_TOTAL = 8;
    constexpr int    NY_TOTAL = 8;
    constexpr int    NZ_TOTAL = 8;
    constexpr double LX       = 1.0;
    constexpr double LY       = 1.0;
    constexpr double LZ       = 1.0;

    const double hx = LX / NX_TOTAL;
    const double hy = LY / NY_TOTAL;
    const double hz = LZ / NZ_TOTAL;

    // IBM 粒子：在立方体中间放一个球
    const double cx = 0.5 * LX;
    const double cy = 0.5 * LY;
    const double cz = 0.5 * LZ;
    const double r  = 0.15 * LX;

    std::cout << "=== IBM 3D Sphere Validation ===\n";
    std::cout << "Computational domain: [" << 0 << "," << LX << "] x [" << 0 << "," << LY << "] x [" << 0 << "," << LZ
              << "]\n";
    std::cout << "Grid size: " << NX_TOTAL << " x " << NY_TOTAL << " x " << NZ_TOTAL << "\n";
    std::cout << "Grid spacing: (" << hx << "," << hy << "," << hz << ")\n";
    std::cout << "IBM sphere: center=(" << cx << "," << cy << "," << cz << "), radius=" << r << "\n";
    std::cout << "Number of IBM particles: 400\n\n";

    // ------------------ 情况 1：单个 domain ------------------
    Geometry3D geo_single;

    Domain3DUniform d_single(NX_TOTAL, NY_TOTAL, NZ_TOTAL, LX, LY, LZ, "Single3D");
    geo_single.add_domain(&d_single);
    geo_single.set_global_spatial_step(hx, hy, hz);

    print_geometry_info(geo_single, "Case 1: Single Domain");

    Variable3D u_single("u_single"), v_single("v_single"), w_single("w_single");
    u_single.set_geometry(geo_single);
    v_single.set_geometry(geo_single);
    w_single.set_geometry(geo_single);

    field3 u_single_f, v_single_f, w_single_f;
    u_single.set_x_face_center_field(&d_single, u_single_f);
    v_single.set_y_face_center_field(&d_single, v_single_f);
    w_single.set_z_face_center_field(&d_single, w_single_f);

    init_velocity_sin(u_single, v_single, w_single);

    PCoordMap3D coord_map_single;
    coord_map_single.add_sphere(hx, r, cx, cy, cz);
    coord_map_single.generate_map(&geo_single);

    auto coord_map_single_raw = coord_map_single.get_map();

    IBVelocitySolver3D_Uhlmann ibm_single(&u_single, &v_single, &w_single, coord_map_single_raw);
    ibm_single.set_parameters(coord_map_single.get_h(), hx);

    // Keep a copy before IBM solve for comparison
    Variable3D u_single_before("u_single_before"), v_single_before("v_single_before"), w_single_before("w_single_before");
    u_single_before.set_geometry(geo_single);
    v_single_before.set_geometry(geo_single);
    w_single_before.set_geometry(geo_single);
    field3 u_single_before_f, v_single_before_f, w_single_before_f;
    u_single_before.set_x_face_center_field(&d_single, u_single_before_f);
    v_single_before.set_y_face_center_field(&d_single, v_single_before_f);
    w_single_before.set_z_face_center_field(&d_single, w_single_before_f);
    copy_velocity(u_single, v_single, w_single, u_single_before, v_single_before, w_single_before);

    // Apply IBM
    ibm_single.solve();

    // Compare before/after
    sample_and_compare_velocity(u_single_before, v_single_before, w_single_before,
                                 u_single, v_single, w_single, cx, cy, cz, r, 1e-10, 1e-8);

    // ------------------ 情况 2：两个 domain 拼接（沿 x 方向）------------------
    Geometry3D geo_multi;

    Domain3DUniform d_left(NX_TOTAL / 2, NY_TOTAL, NZ_TOTAL, LX / 2.0, LY, LZ, "Left3D");
    Domain3DUniform d_right(NX_TOTAL / 2, NY_TOTAL, NZ_TOTAL, LX / 2.0, LY, LZ, "Right3D");

    geo_multi.add_domain(&d_left);
    geo_multi.add_domain(&d_right);
    geo_multi.connect(&d_left, LocationType::XPositive, &d_right);
    geo_multi.set_global_spatial_step(hx, hy, hz);

    // Set the axis to automatically compute offsets
    geo_multi.axis(&d_left, LocationType::XNegative);

    print_geometry_info(geo_multi, "Case 2: Multi Domain");

    Variable3D u_multi("u_multi"), v_multi("v_multi"), w_multi("w_multi");
    u_multi.set_geometry(geo_multi);
    v_multi.set_geometry(geo_multi);
    w_multi.set_geometry(geo_multi);

    field3 u_left_f, u_right_f;
    field3 v_left_f, v_right_f;
    field3 w_left_f, w_right_f;

    u_multi.set_x_face_center_field(&d_left, u_left_f);
    u_multi.set_x_face_center_field(&d_right, u_right_f);
    v_multi.set_y_face_center_field(&d_left, v_left_f);
    v_multi.set_y_face_center_field(&d_right, v_right_f);
    w_multi.set_z_face_center_field(&d_left, w_left_f);
    w_multi.set_z_face_center_field(&d_right, w_right_f);

    init_velocity_sin(u_multi, v_multi, w_multi);

    PCoordMap3D coord_map_multi;
    coord_map_multi.add_sphere(hx, r, cx, cy, cz);
    coord_map_multi.generate_map(&geo_multi);

    auto coord_map_multi_raw = coord_map_multi.get_map();

    IBVelocitySolver3D_Uhlmann ibm_multi(&u_multi, &v_multi, &w_multi, coord_map_multi_raw);
    ibm_multi.set_parameters(coord_map_multi.get_h(), hx);

    // Keep a copy before IBM solve for comparison
    Variable3D u_multi_before("u_multi_before"), v_multi_before("v_multi_before"), w_multi_before("w_multi_before");
    u_multi_before.set_geometry(geo_multi);
    v_multi_before.set_geometry(geo_multi);
    w_multi_before.set_geometry(geo_multi);
    field3 u_multi_before_left_f, v_multi_before_left_f, w_multi_before_left_f;
    field3 u_multi_before_right_f, v_multi_before_right_f, w_multi_before_right_f;
    u_multi_before.set_x_face_center_field(&d_left, u_multi_before_left_f);
    u_multi_before.set_x_face_center_field(&d_right, u_multi_before_right_f);
    v_multi_before.set_y_face_center_field(&d_left, v_multi_before_left_f);
    v_multi_before.set_y_face_center_field(&d_right, v_multi_before_right_f);
    w_multi_before.set_z_face_center_field(&d_left, w_multi_before_left_f);
    w_multi_before.set_z_face_center_field(&d_right, w_multi_before_right_f);
    copy_velocity(u_multi, v_multi, w_multi, u_multi_before, v_multi_before, w_multi_before);

    // Apply IBM
    ibm_multi.solve();

    // Compare before/after
    sample_and_compare_velocity(u_multi_before, v_multi_before, w_multi_before,
                                 u_multi, v_multi, w_multi, cx, cy, cz, r, 1e-10, 1e-8);

    // 按物理坐标真正对比单域结果和双域结果
    compare_velocity_fields_phys(u_single, v_single, w_single, u_multi, v_multi, w_multi, 1e-10, 1e-8);

    std::cout << "[IBM 3D] Validation finished.\n";

    return 0;
}
