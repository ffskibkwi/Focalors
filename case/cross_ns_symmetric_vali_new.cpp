#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"

#include "ns/ns_solver2d.h"

#include "io/config.h"
#include "io/csv_writer_2d.h"

#include "pe/concat/concat_solver2d.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

static double max_abs(const field2& f)
{
    double m = 0.0;
    for (int i = 0; i < f.get_nx(); ++i)
        for (int j = 0; j < f.get_ny(); ++j)
            m = std::max(m, std::abs(f(i, j)));
    return m;
}

static void calc_diff_with_two_field_along_y(field2& src1, field2& src2, field2& diff)
{
    // diff(i,j) = src1(i,j) - src2(nx-1-i, j)
    int nx = std::min(src1.get_nx(), src2.get_nx());
    int ny = std::min(src1.get_ny(), src2.get_ny());
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            diff(i, j) = src1(i, j) - src2(src2.get_nx() - 1 - i, j);
}

static void calc_diff_with_two_field_reversed_along_y(field2& src1, field2& src2, field2& diff)
{
    int nx = std::min(src1.get_nx(), src2.get_nx());
    int ny = std::min(src1.get_ny(), src2.get_ny());
    for (int i = 0; i < nx - 1; ++i)
        for (int j = 0; j < ny; ++j)
            diff(i, j) = src1(i + 1, j) + src2(nx - 1 - i, j);
}

static void calc_diff_with_two_field_along_x(field2& src1, field2& src2, field2& diff)
{
    // diff(i,j) = src1(i,j+1) - src2(i, ny-1-j)
    int nx = std::min(src1.get_nx(), src2.get_nx());
    int ny = std::min(src1.get_ny(), src2.get_ny());
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            diff(i, j) = src1(i, j) - src2(i, src2.get_ny() - 1 - j);
}

static void calc_diff_with_two_field_reversed_along_x(field2& src1, field2& src2, field2& diff)
{
    // diff(i,j) = src1(i,j+1) + src2(i, ny-1-j)
    int nx = std::min(src1.get_nx(), src2.get_nx());
    int ny = std::min(src1.get_ny(), src2.get_ny());
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny - 1; ++j)
            diff(i, j) = src1(i, j + 1) + src2(i, src2.get_ny() - 1 - j);
}

// pretty print a field in matrix-aligned form (top-to-bottom), fixed 3 dp
static void print_field_aligned(const field2& f, const std::string& name)
{
    std::ios old(nullptr);
    old.copyfmt(std::cout);

    std::cout << "Field: " << name << " (" << f.get_nx() << " x " << f.get_ny() << ")\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int j = f.get_ny() - 1; j >= 0; --j)
    {
        for (int i = 0; i < f.get_nx(); ++i)
            std::cout << std::setw(10) << f(i, j) << " ";
        std::cout << "\n";
    }
    std::cout << std::string(80, '-') << "\n";

    std::cout.copyfmt(old);
}
std::string to_string(PDEBoundaryType type)
{
    switch (type)
    {
        case PDEBoundaryType::Dirichlet:
            return "Dirichlet";
        case PDEBoundaryType::Neumann:
            return "Neumann";
        case PDEBoundaryType::Periodic:
            return "Periodic";
        case PDEBoundaryType::Adjacented:
            return "Adjacented";
        default:
            return "Unknown";
    }
}
std::string to_string(LocationType loc)
{
    switch (loc)
    {
        case LocationType::Left:
            return "Left";
        case LocationType::Right:
            return "Right";
        case LocationType::Down:
            return "Down";
        case LocationType::Up:
            return "Up";
        case LocationType::Front:
            return "Front";
        case LocationType::Back:
            return "Back";
        default:
            return "Unknown";
    }
}
int main(int argc, char* argv[])
{
    // Geometry: Cross shape
    Geometry2D geo_cross;

    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes      = true;
    env_config->showCurrentStep   = true;

    TimeAdvancingConfig* time_config = new TimeAdvancingConfig();
    time_config->dt                  = 0.001;
    time_config->num_iterations      = 1; // one NS step for validation

    PhysicsConfig* physics_config = new PhysicsConfig();
    physics_config->nu            = 0.01;

    // Center domain
    Domain2DUniform A2(6, 6, 1.0, 1.0, "A2");

    // Left / Right arms (ensure same ny as A2 after connect)
    Domain2DUniform A1("A1");
    A1.set_nx(6);
    A1.set_lx(1.0);
    Domain2DUniform A3("A3");
    A3.set_nx(6);
    A3.set_lx(1.0);

    // Down / Up arms (ensure same nx as A2 after connect)
    Domain2DUniform A4("A4");
    A4.set_ny(6);
    A4.set_ly(1.0);
    Domain2DUniform A5("A5");
    A5.set_ny(6);
    A5.set_ly(1.0);

    geo_cross.add_domain(A1);
    geo_cross.add_domain(A2);
    geo_cross.add_domain(A3);
    geo_cross.add_domain(A4);
    geo_cross.add_domain(A5);

    // Construct cross connectivity
    geo_cross.connect(A2, LocationType::Left, A1);
    geo_cross.connect(A2, LocationType::Right, A3);
    geo_cross.connect(A2, LocationType::Down, A4);
    geo_cross.connect(A2, LocationType::Up, A5);

    // Variables
    Variable u("u"), v("v"), p("p");
    u.set_geometry(geo_cross);
    v.set_geometry(geo_cross);
    p.set_geometry(geo_cross);

    // Fields on each domain
    field2 u_A1("u_A1"), u_A2("u_A2"), u_A3("u_A3"), u_A4("u_A4"), u_A5("u_A5");
    field2 v_A1("v_A1"), v_A2("v_A2"), v_A3("v_A3"), v_A4("v_A4"), v_A5("v_A5");
    field2 p_A1("p_A1"), p_A2("p_A2"), p_A3("p_A3"), p_A4("p_A4"), p_A5("p_A5");

    u.set_x_edge_field(&A1, u_A1);
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    u.set_x_edge_field(&A4, u_A4);
    u.set_x_edge_field(&A5, u_A5);
    v.set_y_edge_field(&A1, v_A1);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    v.set_y_edge_field(&A4, v_A4);
    v.set_y_edge_field(&A5, v_A5);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);
    p.set_center_field(&A5, p_A5);

    // Helper setters
    auto set_dirichlet_zero = [](Variable& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto is_adjacented = [&](Domain2DUniform* d, LocationType loc) {
        return geo_cross.adjacency.count(d) && geo_cross.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};
    std::vector<LocationType> dirs = {LocationType::Left, LocationType::Right, LocationType::Down, LocationType::Up};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(d, loc))
                continue; // internal boundaries handled automatically
            // velocity: default wall (Dirichlet 0)
            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            // pressure: default Neumann (zero gradient)
            set_neumann_zero(p, d, loc);
        }
    }

    // Inlet profiles for symmetry validation (Poiseuille)
    const double U0 = 1.0;

    u.set_boundary_type(&A1, LocationType::Left, PDEBoundaryType::Dirichlet);
    u.has_boundary_value_map[&A1][LocationType::Left] = true;
    set_dirichlet_zero(v, &A1, LocationType::Left);
    // A1 Left: u(y_norm) = +6*U0*y_norm*(1-y_norm)
    for (int j = 0; j < u_A1.get_ny(); ++j)
    {
        double y_norm                                    = (j + 0.5) / static_cast<double>(u_A1.get_ny());
        double u_val                                     = 6.0 * U0 * y_norm * (1.0 - y_norm);
        u.boundary_value_map[&A1][LocationType::Left][j] = u_val;
    }
    set_dirichlet_zero(v, &A1, LocationType::Left);
    // A3 Right: u(y_norm) = -6*U0*y_norm*(1-y_norm)
    u.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Dirichlet);
    u.has_boundary_value_map[&A3][LocationType::Right] = true;
    set_dirichlet_zero(v, &A3, LocationType::Right);
    for (int j = 0; j < u_A3.get_ny(); ++j)
    {
        double y_norm                                     = (j + 0.5) / static_cast<double>(u_A3.get_ny());
        double u_val                                      = -6.0 * U0 * y_norm * (1.0 - y_norm);
        u.boundary_value_map[&A3][LocationType::Right][j] = u_val;
    }

    // A4 Down: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A4, LocationType::Down);
    set_neumann_zero(v, &A4, LocationType::Down);

    // A5 Up: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A5, LocationType::Up);
    set_neumann_zero(v, &A5, LocationType::Up);

    int nx = u_A2.get_nx();
    int ny = u_A2.get_ny();
    // A1
    for (int i = 1; i < u_A1.get_nx(); i++)
    {
        for (int j = 0; j < (u_A1.get_ny() / 2); j++)
        {
            double val = i + j;

            u_A1(i, j)                     = val;
            u_A1(i, u_A1.get_ny() - 1 - j) = val;
        }
    }
    for (int i = 1; i < v_A1.get_nx(); i++)
    {
        for (int j = 1; j < v_A1.get_ny(); j++)
        {
            double val = (j < ny / 2) ? (i + j) : -(i + ny - j);
            v_A1(i, j) = -val;
        }
    }
    for (int i = 1; i < v_A1.get_nx(); i++)
    {
        v_A1(i, v_A1.get_nx() / 2) = 0;
    }

    // A4
    for (int i = 1; i < u_A4.get_nx(); i++)
    {
        for (int j = 0; j < u_A4.get_ny(); j++)
        {
            if (i == u_A4.get_nx() / 2)
            {
                for (int j = 0; j < ny; ++j)
                {
                    u_A4(i, j) = 0;
                }
                continue;
            }
            // 左半部分为正，右半部分为负
            double val = (i < nx / 2) ? i * (j + 1) : -(nx - i) * (j + 1);
            u_A4(i, j) = val;
        }
    }
    for (int i = 0; i < v_A4.get_nx() / 2; i++)
    {
        for (int j = 1; j < v_A4.get_ny(); j++)
        {
            double val = i + j;

            v_A4(i, j)                     = -val;
            v_A4(v_A4.get_nx() - 1 - i, j) = -val;
        }
    }
    // u_A2: 上下对称，左右反对称
    for (int i = 1; i < nx; ++i)
    {
        for (int j = 0; j < ny / 2; ++j)
        {
            if (i == u_A2.get_nx() / 2)
            {
                for (int j = 0; j < ny / 2; ++j)
                {
                    u_A2(i, j)          = 0;
                    u_A2(i, ny - 1 - j) = 0;
                }
                continue;
            }
            // 左半部分为正，右半部分为负
            double val = (i < nx / 2) ? i * (j + 1) : -(nx - i) * (j + 1);
            u_A2(i, j) = val;
            // 上下对称
            u_A2(i, ny - 1 - j) = val;
        }
    }
    for (int j = 0; j < u_A2.get_ny() / 2; j++)
    {
        u_A2(0, j)                     = j + 1;
        u_A2(0, u_A2.get_ny() - 1 - j) = j + 1;
    }

    // v_A2: 上下反对称，左右对称
    for (int i = 0; i < nx / 2; ++i)
    {
        for (int j = 1; j < ny; ++j)
        {
            if (j == v_A2.get_nx() / 2)
            {
                for (int i = 0; i < nx; ++i)
                {
                    v_A2(i, j) = 0;
                }
                continue;
            }
            // 下半部分为负，上半部分为正
            double val          = (j < ny / 2) ? -(i + 1) * j : (i + 1) * (ny - j);
            v_A2(i, j)          = val;
            v_A2(nx - i - 1, j) = val;
        }
    }
    for (int i = 0; i < nx / 2; ++i)
    {
        v_A2(i, 0)          = -i;
        v_A2(nx - 1 - i, 0) = -i;
    }
    // u3 copy from u1
    for (int i = 0; i < u_A1.get_nx() - 1; i++)
    {
        for (int j = 0; j < u_A1.get_ny(); j++)
        {
            u_A3(u_A3.get_nx() - 1 - i, j) = -u_A1(i + 1, j);
        }
    }
    for (int j = 0; j < u_A3.get_ny(); j++)
    {
        u_A3(0, j) = -u_A2(0, j);
    }
    // v3 copy from v1
    for (int i = 0; i < v_A1.get_nx(); i++)
    {
        for (int j = 1; j < v_A1.get_ny(); j++)
        {
            v_A3(v_A3.get_nx() - 1 - i, j) = v_A1(i, j);
        }
    }
    // u5 copy from u4
    for (int i = 1; i < u_A4.get_nx(); i++)
    {
        for (int j = 0; j < u_A4.get_ny(); j++)
        {
            u_A5(i, u_A5.get_ny() - 1 - j) = u_A4(i, j);
        }
    }
    // v5 copy from v4
    for (int i = 0; i < v_A4.get_nx(); i++)
    {
        for (int j = 0; j < v_A4.get_ny() - 1; j++)
        {
            v_A5(i, v_A5.get_ny() - 1 - j) = -v_A4(i, j + 1);
        }
    }
    for (int i = 0; i < v_A4.get_nx(); i++)
    {
        v_A5(i, 0) = -v_A2(i, 0);
    }
    // 这里直接用 .at() 获取指针，若不存在则为nullptr
    auto get_buffer_ptr = [](auto& map, Domain2DUniform* dom, LocationType loc) -> double* {
        try
        {
            return map.at(dom).at(loc);
        }
        catch (...)
        {
            return nullptr;
        }
    };

    // Solve
    ConcatNSSolver2D solver(&u, &v, &p, time_config, physics_config, env_config);
    solver.variable_check();
    solver.phys_boundary_update();
    solver.nondiag_shared_boundary_update();
    solver.diag_shared_boundary_update();
    // 输出的u为结束时刻的u 输出buffer只能得到前一dt的值，无法和field做比较
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, double*>> u_buffer_map, v_buffer_map,
        p_buffer_map;
    std::unordered_map<Domain2DUniform*, double>& left_up_corner_value_map    = v.left_up_corner_value_map;
    std::unordered_map<Domain2DUniform*, double>& right_down_corner_value_map = u.right_down_corner_value_map;
    u_buffer_map                                                              = u.buffer_map;
    v_buffer_map                                                              = v.buffer_map;
    p_buffer_map                                                              = p.buffer_map;
    double* v1_left_buffer  = get_buffer_ptr(v_buffer_map, &A1, LocationType::Left);
    double* u1_left_buffer  = get_buffer_ptr(u_buffer_map, &A1, LocationType::Left);
    double* v1_right_buffer = get_buffer_ptr(v_buffer_map, &A1, LocationType::Right);
    double* u1_right_buffer = get_buffer_ptr(u_buffer_map, &A1, LocationType::Right);
    double* u1_down_buffer  = get_buffer_ptr(u_buffer_map, &A1, LocationType::Down);
    double* v1_down_buffer  = get_buffer_ptr(v_buffer_map, &A1, LocationType::Down);
    double* u1_up_buffer    = get_buffer_ptr(u_buffer_map, &A1, LocationType::Up);
    double* v1_up_buffer    = get_buffer_ptr(v_buffer_map, &A1, LocationType::Up);

    double* v2_left_buffer  = get_buffer_ptr(v_buffer_map, &A2, LocationType::Left);
    double* u2_left_buffer  = get_buffer_ptr(u_buffer_map, &A2, LocationType::Left);
    double* v2_right_buffer = get_buffer_ptr(v_buffer_map, &A2, LocationType::Right);
    double* u2_right_buffer = get_buffer_ptr(u_buffer_map, &A2, LocationType::Right);
    double* u2_down_buffer  = get_buffer_ptr(u_buffer_map, &A2, LocationType::Down);
    double* v2_down_buffer  = get_buffer_ptr(v_buffer_map, &A2, LocationType::Down);
    double* u2_up_buffer    = get_buffer_ptr(u_buffer_map, &A2, LocationType::Up);
    double* v2_up_buffer    = get_buffer_ptr(v_buffer_map, &A2, LocationType::Up);

    double* v3_left_buffer  = get_buffer_ptr(v_buffer_map, &A3, LocationType::Left);
    double* u3_left_buffer  = get_buffer_ptr(u_buffer_map, &A3, LocationType::Left);
    double* v3_right_buffer = get_buffer_ptr(v_buffer_map, &A3, LocationType::Right);
    double* u3_right_buffer = get_buffer_ptr(u_buffer_map, &A3, LocationType::Right);
    double* u3_down_buffer  = get_buffer_ptr(u_buffer_map, &A3, LocationType::Down);
    double* v3_down_buffer  = get_buffer_ptr(v_buffer_map, &A3, LocationType::Down);
    double* u3_up_buffer    = get_buffer_ptr(u_buffer_map, &A3, LocationType::Up);
    double* v3_up_buffer    = get_buffer_ptr(v_buffer_map, &A3, LocationType::Up);

    double* v4_left_buffer  = get_buffer_ptr(v_buffer_map, &A4, LocationType::Left);
    double* u4_left_buffer  = get_buffer_ptr(u_buffer_map, &A4, LocationType::Left);
    double* v4_right_buffer = get_buffer_ptr(v_buffer_map, &A4, LocationType::Right);
    double* u4_right_buffer = get_buffer_ptr(u_buffer_map, &A4, LocationType::Right);
    double* v4_down_buffer  = get_buffer_ptr(v_buffer_map, &A4, LocationType::Down);
    double* u4_down_buffer  = get_buffer_ptr(u_buffer_map, &A4, LocationType::Down);
    double* v4_up_buffer    = get_buffer_ptr(v_buffer_map, &A4, LocationType::Up);
    double* u4_up_buffer    = get_buffer_ptr(u_buffer_map, &A4, LocationType::Up);

    double* v5_left_buffer  = get_buffer_ptr(v_buffer_map, &A5, LocationType::Left);
    double* u5_left_buffer  = get_buffer_ptr(u_buffer_map, &A5, LocationType::Left);
    double* v5_right_buffer = get_buffer_ptr(v_buffer_map, &A5, LocationType::Right);
    double* u5_right_buffer = get_buffer_ptr(u_buffer_map, &A5, LocationType::Right);
    double* u5_down_buffer  = get_buffer_ptr(u_buffer_map, &A5, LocationType::Down);
    double* v5_down_buffer  = get_buffer_ptr(v_buffer_map, &A5, LocationType::Down);
    double* u5_up_buffer    = get_buffer_ptr(u_buffer_map, &A5, LocationType::Up);
    double* v5_up_buffer    = get_buffer_ptr(v_buffer_map, &A5, LocationType::Up);

    // Symmetry validation
    field2 u_diff_r_1_3(u_A1.get_nx() - 1, u_A1.get_ny(), "u_diff_r_1_3");
    field2 v_diff_1_3(v_A1.get_nx(), v_A1.get_ny(), "v_diff_1_3");
    field2 u_diff_4_5(u_A4.get_nx(), u_A4.get_ny(), "u_diff_4_5");
    field2 v_diff_r_4_5(v_A4.get_nx(), v_A4.get_ny() - 1, "v_diff_r_4_5");

    auto print_buffer_info = [&]() {
        for (int j = 0; j < 5; ++j)
        {
            std::cout << "u_A1(0, " << j << "): " << u_A1(0, j) << std::endl;
            std::cout << "u3_right_buffer[" << j << "]: " << u3_right_buffer[j] << std::endl;
            std::cout << "u_A1(0, " << j << ") + u3_right_buffer[" << j << "]: " << u_A1(0, j) + u3_right_buffer[j]
                      << std::endl;
            std::cout << std::endl;

            std::cout << "u_A3(0, " << j << "): " << u_A3(0, j) << std::endl;
            std::cout << "u1_right_buffer[" << j << "]: " << u1_right_buffer[j] << std::endl;
            std::cout << "u1_right_buffer[" << j << "] + u_A3(0, " << j << "): " << u1_right_buffer[j] + u_A3(0, j)
                      << std::endl;
            std::cout << "u1_right_buffer[" << j << "] - u_A2(0, " << j << "): " << u1_right_buffer[j] - u_A2(0, j)
                      << std::endl;
            std::cout << std::endl;

            std::cout << "u2_right_buffer[" << j << "]: " << u2_right_buffer[j] << std::endl;
            std::cout << "u2_right_buffer[" << j << "] - u_A3(0, " << j << "): " << u2_right_buffer[j] - u_A3(0, j)
                      << std::endl;
            std::cout << "u2_right_buffer[" << j << "] + u_A2(0, " << j << "): " << u2_right_buffer[j] + u_A2(0, j)
                      << std::endl;
            std::cout << std::endl;

            std::cout << "u_A4(0, " << j << "): " << u_A4(0, j) << std::endl;
            std::cout << "u4_right_buffer[" << j << "]: " << u4_right_buffer[j] << std::endl;
            std::cout << "u_A4(0, " << j << ") + u5_left_buffer[" << j << "]: " << u_A4(0, j) + u5_left_buffer[j]
                      << std::endl;
            std::cout << std::endl;

            std::cout << "u_A5(0, " << j << "): " << u_A5(0, j) << std::endl;
            std::cout << "u5_right_buffer[" << j << "]: " << u5_right_buffer[j] << std::endl;
            std::cout << "u_A5(0, " << j << ") + u5_right_buffer[" << j << "]: " << u_A5(0, j) + u5_right_buffer[j]
                      << std::endl;
            std::cout << std::endl;
        }

        for (int i = 0; i < 5; ++i)
        {
            std::cout << "v_A4(" << i << ", 0): " << v_A4(i, 0) << std::endl;
            std::cout << "v5_up_buffer[" << i << "]: " << v5_up_buffer[i] << std::endl;
            std::cout << "v_A4(" << i << ", 0) + v5_up_buffer[" << i << "]: " << v_A4(i, 0) + v5_up_buffer[i]
                      << std::endl;
            std::cout << std::endl;

            std::cout << "v_A5(" << i << ", 0): " << v_A5(i, 0) << std::endl;
            std::cout << "v4_up_buffer[" << i << "]: " << v4_up_buffer[i] << std::endl;
            std::cout << "v_A5(" << i << ", 0) + v4_up_buffer[" << i << "]: " << v_A5(i, 0) + v4_up_buffer[i]
                      << std::endl;
            std::cout << "v_A2(" << i << ", 0) - v4_up_buffer[" << i << "]: " << v_A2(i, 0) - v4_up_buffer[i]
                      << std::endl;
            std::cout << "v_A2(" << i << ", 0) + v2_up_buffer[" << i << "]: " << v_A2(i, 0) + v2_up_buffer[i]
                      << std::endl;
            std::cout << std::endl;

            std::cout << "v_A1(" << i << ", 0): " << v_A1(i, 0) << std::endl;
            std::cout << "v1_up_buffer[" << i << "]: " << v1_up_buffer[i] << std::endl;
            std::cout << "v_A1(" << i << ", 0) + v1_up_buffer[" << i << "]: " << v_A1(i, 0) + v1_up_buffer[i]
                      << std::endl;
            std::cout << std::endl;

            std::cout << "v_A3(" << i << ", 0): " << v_A3(i, 0) << std::endl;
            std::cout << "v3_up_buffer[" << i << "]: " << v3_up_buffer[i] << std::endl;
            std::cout << "v_A3(" << i << ", 0) + v3_up_buffer[" << i << "]: " << v_A3(i, 0) + v3_up_buffer[i]
                      << std::endl;
            std::cout << std::endl;
        }
        std::cout << "right_down_corner_value_map[&A1]"
                  << " : " << right_down_corner_value_map[&A1] << std::endl;
        std::cout << "u_A4(0, ny - 1) : " << u_A4(0, ny - 1) << std::endl;
        std::cout << "right_down_corner_value_map[&A1] - u_A4(0, ny - 1) : "
                  << right_down_corner_value_map[&A1] - u_A4(0, ny - 1) << std::endl;
        std::cout << std::endl;

        std::cout << "left_up_corner_value_map[&A2]"
                  << " : " << left_up_corner_value_map[&A2] << std::endl;
        std::cout << "v1_up_buffer[nx - 1]: " << v1_up_buffer[nx - 1] << std::endl;
        std::cout << "left_up_corner_value_map[&A2] - v1_up_buffer[nx - 1]: "
                  << left_up_corner_value_map[&A2] - v1_up_buffer[nx - 1] << std::endl;
        std::cout << std::endl;

        std::cout << "right_down_corner_value_map[&A2]"
                  << " : " << right_down_corner_value_map[&A2] << std::endl;
        std::cout << "u4_right_buffer[ny - 1]"
                  << " : " << u4_right_buffer[ny - 1] << std::endl;
        std::cout << "right_down_corner_value_map[&A2] - u4_right_buffer[ny - 1]: "
                  << right_down_corner_value_map[&A2] - u4_right_buffer[ny - 1] << std::endl;
        std::cout << std::endl;

        std::cout << "left_up_corner_value_map[&A3]"
                  << " : " << left_up_corner_value_map[&A3] << std::endl;
        std::cout << "v2_up_buffer[nx - 1]"
                  << " : " << v2_up_buffer[nx - 1] << std::endl;
        std::cout << "left_up_corner_value_map[&A3] - v2_up_buffer[nx - 1]: "
                  << left_up_corner_value_map[&A3] - v2_up_buffer[nx - 1] << std::endl;
        std::cout << std::endl;

        std::cout << "left_up_corner_value_map[&A4]"
                  << " : " << left_up_corner_value_map[&A4] << std::endl;
        std::cout << "v_A1(nx - 1, 0)"
                  << " : " << v_A1(nx - 1, 0) << std::endl;
        std::cout << "left_up_corner_value_map[&A4] - v3_up_buffer[ny - 1]: "
                  << left_up_corner_value_map[&A4] - v3_up_buffer[ny - 1] << std::endl;
        std::cout << std::endl;

        std::cout << "right_down_corner_value_map[&A5]"
                  << " : " << right_down_corner_value_map[&A5] << std::endl;
        std::cout << "u2_right_buffer[ny - 1]"
                  << " : " << u2_right_buffer[ny - 1] << std::endl;
        std::cout << "right_down_corner_value_map[&A5] - u2_right_buffer[ny - 1]: "
                  << right_down_corner_value_map[&A5] - u2_right_buffer[ny - 1] << std::endl;
        std::cout << std::endl;
    };
    auto print_all_field = [&]() {
        std::cout << "\n[Field Values]\n";
        print_field_aligned(u_A1, "u_A1");
        print_field_aligned(u_A3, "u_A3");
        print_field_aligned(u_A5, "u_A5");
        print_field_aligned(u_A2, "u_A2");
        print_field_aligned(u_A4, "u_A4");
        print_field_aligned(v_A1, "v_A1");
        print_field_aligned(v_A3, "v_A3");
        print_field_aligned(v_A5, "v_A5");
        print_field_aligned(v_A2, "v_A2");
        print_field_aligned(v_A4, "v_A4");
        std::cout << std::setprecision(6) << std::scientific;
        std::cout << "\n[ Symmetry Check]\n";
        std::cout << "[NS Symmetry] L_inf(u_1 + u_3^R) = " << max_abs(u_diff_r_1_3) << "\n";
        std::cout << "[NS Symmetry] L_inf(v_1 - v_3^R) = " << max_abs(v_diff_1_3) << "\n";
        std::cout << "[NS Symmetry] L_inf(u_4 - u_5^R) = " << max_abs(u_diff_4_5) << "\n";
        std::cout << "[NS Symmetry] L_inf(v_4 + v_5^R) = " << max_abs(v_diff_r_4_5) << "\n";

        // Pretty matrices for visual inspection
        std::cout << "\n[Symmetry matrices]\n";
        print_field_aligned(u_diff_r_1_3, "u_diff_r_1_3 = u_A1 + ReverseY(u_A3)");
        print_field_aligned(v_diff_1_3, "v_diff_1_3   = v_A1 - ReverseY(v_A3)");
        print_field_aligned(u_diff_4_5, "u_diff_4_5   = u_A4 - ReverseX(u_A5)");
        print_field_aligned(v_diff_r_4_5, "v_diff_r_4_5 = v_A4 + ReverseX(v_A5)");
    };
    calc_diff_with_two_field_reversed_along_y(u_A1, u_A3, u_diff_r_1_3); // expect near 0
    calc_diff_with_two_field_along_y(v_A1, v_A3, v_diff_1_3);            // expect near 0
    calc_diff_with_two_field_along_x(u_A4, u_A5, u_diff_4_5);            // expect near 0
    calc_diff_with_two_field_reversed_along_x(v_A4, v_A5, v_diff_r_4_5); // expect near 0

    std::cout << "--------print before solve--------" << std::endl;
    print_buffer_info();
    print_all_field();
    IO::field_to_csv(u_A1, "result/cross_ns_sym/u_A1_init");
    IO::field_to_csv(v_A1, "result/cross_ns_sym/v_A1_init");
    IO::field_to_csv(u_A2, "result/cross_ns_sym/u_A2_init");
    IO::field_to_csv(v_A2, "result/cross_ns_sym/v_A2_init");
    IO::field_to_csv(u_A3, "result/cross_ns_sym/u_A3_init");
    IO::field_to_csv(v_A3, "result/cross_ns_sym/v_A3_init");
    IO::field_to_csv(u_A4, "result/cross_ns_sym/u_A4_init");
    IO::field_to_csv(v_A4, "result/cross_ns_sym/v_A4_init");
    IO::field_to_csv(u_A5, "result/cross_ns_sym/u_A5_init");
    IO::field_to_csv(v_A5, "result/cross_ns_sym/v_A5_init");

    solver.solve();
    // Symmetry validation
    calc_diff_with_two_field_reversed_along_y(u_A1, u_A3, u_diff_r_1_3); // expect near 0
    calc_diff_with_two_field_along_y(v_A1, v_A3, v_diff_1_3);            // expect near 0
    calc_diff_with_two_field_along_x(u_A4, u_A5, u_diff_4_5);            // expect near 0
    calc_diff_with_two_field_reversed_along_x(v_A4, v_A5, v_diff_r_4_5); // expect near 0
    std::cout << "--------print after solve--------" << std::endl;
    print_all_field();

    solver.phys_boundary_update();
    solver.nondiag_shared_boundary_update();
    solver.diag_shared_boundary_update();

    // todo 检查corner buffer

    print_buffer_info();
    // Optional CSV outputs (uncomment if needed)
    IO::field_to_csv(u_A1, "result/cross_ns_sym/u_A1");
    IO::field_to_csv(u_A2, "result/cross_ns_sym/u_A2");
    IO::field_to_csv(u_A3, "result/cross_ns_sym/u_A3");
    IO::field_to_csv(u_A4, "result/cross_ns_sym/u_A4");
    IO::field_to_csv(u_A5, "result/cross_ns_sym/u_A5");
    IO::field_to_csv(v_A1, "result/cross_ns_sym/v_A1");
    IO::field_to_csv(v_A2, "result/cross_ns_sym/v_A2");
    IO::field_to_csv(v_A3, "result/cross_ns_sym/v_A3");
    IO::field_to_csv(v_A4, "result/cross_ns_sym/v_A4");
    IO::field_to_csv(v_A5, "result/cross_ns_sym/v_A5");

    // check boundary types
    // for (auto* d : domains)
    // {
    //     for (auto loc : dirs)
    //     {
    //         auto& bound_type_map   = u.boundary_type_map[d];
    //         auto& bound_type_map_v = v.boundary_type_map[d];
    //         auto& bound_type_map_p = p.boundary_type_map[d];
    //         std::cout << "Domain: " << d->name << ", Location: " << to_string(loc) << "\n";
    //         std::cout << "  u boundary type: ";
    //         if (bound_type_map.count(loc))
    //             std::cout << to_string(bound_type_map[loc]) << "\n";
    //         else
    //             std::cout << "Not Set\n";
    //         std::cout << "  v boundary type: ";
    //         if (bound_type_map_v.count(loc))
    //             std::cout << to_string(bound_type_map_v[loc]) << "\n";
    //         else
    //             std::cout << "Not Set\n";
    //         std::cout << "  p boundary type: ";
    //         if (bound_type_map_p.count(loc))
    //             std::cout << to_string(bound_type_map_p[loc]) << "\n";
    //         else
    //             std::cout << "Not Set\n";
    //     }
    // }
    return 0;
}