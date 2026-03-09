#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "ibm/ib_solver_3d.h"
#include "ibm/particles_coordinate_map_3d.h"
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
    static int debug_count = 0;
    bool debug_this = (debug_count < 5);

    Geometry3D* geo = u.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();

        if (x < ox || x > ox + lx || y < oy || y > oy + ly || z < oz || z > oz + lz)
            continue;

        auto&  f  = *u.field_map.at(d);
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // u: x-face centered, x = ox + i*hx, y = oy + (j+0.5)*hy, z = oz + (k+0.5)*hz
        int i = static_cast<int>(std::round((x - ox) / hx));
        int j = static_cast<int>(std::round((y - oy) / hy - 0.5));
        int k = static_cast<int>(std::round((z - oz) / hz - 0.5));

        if (debug_this)
        {
            std::cout << "[DEBUG sample_u] Point (" << x << "," << y << "," << z << ") in domain '" << d->name << "'\n";
            std::cout << "    Indices: i=" << i << ", j=" << j << ", k=" << k << "\n";
            std::cout << "    Field size: nx=" << f.get_nx() << ", ny=" << f.get_ny() << ", nz=" << f.get_nz() << "\n";
            debug_count++;
        }

        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            if (debug_this)
                std::cout << "    SUCCESS: value=" << value << "\n";
            return true;
        }
        if (debug_this)
            std::cout << "    FAILED: indices out of bounds\n";
    }
    if (debug_this)
        std::cout << "[DEBUG sample_u] Point (" << x << "," << y << "," << z << ") NOT in any domain\n";
    return false;
}

static bool sample_v_at(const Variable3D& v, double x, double y, double z, double& value)
{
    static int debug_count = 0;
    bool debug_this = (debug_count < 5);

    Geometry3D* geo = v.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();

        if (x < ox || x > ox + lx || y < oy || y > oy + ly || z < oz || z > oz + lz)
            continue;

        auto&  f  = *v.field_map.at(d);
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // v: y-face centered, x = ox + (i+0.5)*hx, y = oy + j*hy, z = oz + (k+0.5)*hz
        int i = static_cast<int>(std::round((x - ox) / hx - 0.5));
        int j = static_cast<int>(std::round((y - oy) / hy));
        int k = static_cast<int>(std::round((z - oz) / hz - 0.5));

        if (debug_this)
        {
            std::cout << "[DEBUG sample_v] Point (" << x << "," << y << "," << z << ") in domain '" << d->name << "'\n";
            std::cout << "    Indices: i=" << i << ", j=" << j << ", k=" << k << "\n";
            std::cout << "    Field size: nx=" << f.get_nx() << ", ny=" << f.get_ny() << ", nz=" << f.get_nz() << "\n";
            debug_count++;
        }

        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            if (debug_this)
                std::cout << "    SUCCESS: value=" << value << "\n";
            return true;
        }
        if (debug_this)
            std::cout << "    FAILED: indices out of bounds\n";
    }
    if (debug_this)
        std::cout << "[DEBUG sample_v] Point (" << x << "," << y << "," << z << ") NOT in any domain\n";
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

        if (x < ox || x > ox + lx || y < oy || y > oy + ly || z < oz || z > oz + lz)
            continue;

        auto&  f  = *w.field_map.at(d);
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // w: z-face centered, x = ox + (i+0.5)*hx, y = oy + (j+0.5)*hy, z = oz + k*hz
        int i = static_cast<int>(std::round((x - ox) / hx - 0.5));
        int j = static_cast<int>(std::round((y - oy) / hy - 0.5));
        int k = static_cast<int>(std::round((z - oz) / hz));

        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            return true;
        }
    }
    return false;
}

// 采样并输出球体边界上的速度值
static void sample_and_print_velocity(const Variable3D&  u,
                                      const Variable3D&  v,
                                      const Variable3D&  w,
                                      const std::string& label,
                                      double             sphere_cx,
                                      double             sphere_cy,
                                      double             sphere_cz,
                                      double             sphere_r)
{
    std::cout << "[" << label << "] Velocity samples on sphere surface (r=" << sphere_r << "):\n";
    auto sample_points = generate_sphere_surface_points(sphere_cx, sphere_cy, sphere_cz, sphere_r, 10);

    for (const auto& [x, y, z] : sample_points)
    {
        double u_val = 0.0, v_val = 0.0, w_val = 0.0;
        bool   u_ok = sample_u_at(u, x, y, z, u_val);
        bool   v_ok = sample_v_at(v, x, y, z, v_val);
        bool   w_ok = sample_w_at(w, x, y, z, w_val);
        std::cout << "    (" << x << "," << y << "," << z << ") u=" << (u_ok ? std::to_string(u_val) : "N/A")
                  << " v=" << (v_ok ? std::to_string(v_val) : "N/A") << " w=" << (w_ok ? std::to_string(w_val) : "N/A")
                  << "\n";
    }
}

// 真正对比：单域 (u_single,v_single,w_single) 与 多域 (u_multi,v_multi,w_multi)
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
                    if (!sample_u_at(u_multi, x, y, z, b) || !approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                    {
                        std::cout << "[u] mismatch at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << ", multi=" << b << "\n";
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
                    if (!sample_v_at(v_multi, x, y, z, b) || !approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                    {
                        std::cout << "[v] mismatch at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << ", multi=" << b << "\n";
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
                    if (!sample_w_at(w_multi, x, y, z, b) || !approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                    {
                        std::cout << "[w] mismatch at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << ", multi=" << b << "\n";
                    }
                }
            }
        }
    }
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
    coord_map_single.add_sphere(400, r, cx, cy, cz);
    coord_map_single.generate_map(&geo_single);

    auto coord_map_single_raw = coord_map_single.get_map();

    // Debug: check particle data
    for (auto* domain : geo_single.domains)
    {
        auto* particles = coord_map_single_raw[domain];
        std::cout << "[DEBUG] Domain '" << domain->name << "' has " << particles->cur_n << " particles\n";
        if (particles->cur_n > 0)
        {
            EXPOSE_PCOORD3D(particles)
            std::cout << "    First particle position: (" << X[0] << "," << Y[0] << "," << Z[0] << ")\n";
        }
    }

    ImmersedBoundarySolver3D ibm_single(&u_single, &v_single, &w_single, coord_map_single_raw);
    ibm_single.set_parameters(coord_map_single.get_h(), hx);

    // Sample before IBM solve
    sample_and_print_velocity(u_single, v_single, w_single, "Case 1 (Before IBM)", cx, cy, cz, r);

    ibm_single.solve();

    // Sample after IBM solve
    sample_and_print_velocity(u_single, v_single, w_single, "Case 1 (After IBM)", cx, cy, cz, r);

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
    coord_map_multi.add_sphere(400, r, cx, cy, cz);
    coord_map_multi.generate_map(&geo_multi);

    auto coord_map_multi_raw = coord_map_multi.get_map();

    ImmersedBoundarySolver3D ibm_multi(&u_multi, &v_multi, &w_multi, coord_map_multi_raw);
    ibm_multi.set_parameters(coord_map_multi.get_h(), hx);

    // Sample before IBM solve
    sample_and_print_velocity(u_multi, v_multi, w_multi, "Case 2 (Before IBM)", cx, cy, cz, r);

    ibm_multi.solve();

    // Sample after IBM solve
    sample_and_print_velocity(u_multi, v_multi, w_multi, "Case 2 (After IBM)", cx, cy, cz, r);

    // 按物理坐标真正对比单域结果和双域结果
    compare_velocity_fields_phys(u_single, v_single, w_single, u_multi, v_multi, w_multi, 1e-10, 1e-8);

    std::cout << "[IBM 3D] Validation finished.\n";

    return 0;
}
