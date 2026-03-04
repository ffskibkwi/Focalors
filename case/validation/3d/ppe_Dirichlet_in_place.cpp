#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "base/math/compare.h"
#include "io/csv_handler.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver3d.h"
#include "ns/physical_pe_solver3d.h"
#include "pe/concat/concat_solver3d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <cmath>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Error argument! Usage: program rank[int > 0]" << std::endl;
        return 0;
    }

    int rank = std::stoi(argv[1]);

    Geometry3D geo;

    double Lx  = 1.0;
    double Ly  = 2.0;
    double Lz  = 3.0;
    double rho = 2.0;

    Domain3DUniform A1(rank, rank, rank, Lx, Ly, Lz, "A1");

    double hx = A1.get_hx();
    double hy = A1.get_hy();
    double hz = A1.get_hz();

    geo.add_domain(&A1);

    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);
    geo.axis(&A1, LocationType::ZNegative);

    Variable3D p("p");
    p.set_geometry(geo);

    field3 p_A1;

    p.set_center_field(&A1, p_A1);

    auto calc_u = [&](double x, double y, double z) { return std::exp(x) * std::exp(y) * z; };
    auto calc_v = [&](double x, double y, double z) { return std::exp(x) * std::exp(y) * z; };
    auto calc_w = [&](double x, double y, double z) { return -std::exp(x) * std::exp(y) * z * z; };

    auto calc_dudx = [&](double x, double y, double z) { return std::exp(x) * std::exp(y) * z; };
    auto calc_dudy = [&](double x, double y, double z) { return std::exp(x) * std::exp(y) * z; };
    auto calc_dudz = [&](double x, double y, double z) { return std::exp(x) * std::exp(y); };

    auto calc_dvdx = [&](double x, double y, double z) { return std::exp(x) * std::exp(y) * z; };
    auto calc_dvdy = [&](double x, double y, double z) { return std::exp(x) * std::exp(y) * z; };
    auto calc_dvdz = [&](double x, double y, double z) { return std::exp(x) * std::exp(y); };

    auto calc_dwdx = [&](double x, double y, double z) { return -std::exp(x) * std::exp(y) * z * z; };
    auto calc_dwdy = [&](double x, double y, double z) { return -std::exp(x) * std::exp(y) * z * z; };
    auto calc_dwdz = [&](double x, double y, double z) { return -std::exp(x) * std::exp(y) * 2 * z; };

    auto calc_neg_rho_div_u_grad_u = [&](double x, double y, double z) {
        return -4.0 * rho * z * z * std::exp(2 * x + 2 * y);
    };

    auto calc_p = [&](double x, double y, double z) { return std::exp(x) * std::exp(y) * std::exp(z); };

    auto calc_laplacian_p = [&](double x, double y, double z) { return 3.0 * std::exp(x) * std::exp(y) * std::exp(z); };

    auto calc_rho_div_f = [&](double x, double y, double z) {
        return 3.0 * std::exp(x + y + z) + 4.0 * rho * z * z * std::exp(2 * x + 2 * y);
    };

    auto calc_error_centered =
        [&](const std::string& name, field3& target, std::function<double(double, double, double)> f) {
            double error = 0.0;
            for (int i = 0; i < target.get_nx(); i++)
            {
                for (int j = 0; j < target.get_ny(); j++)
                {
                    for (int k = 0; k < target.get_nz(); k++)
                    {
                        double x    = (i + 0.5) * hx;
                        double y    = (j + 0.5) * hy;
                        double z    = (k + 0.5) * hz;
                        double diff = target(i, j, k) - f(x, y, z);
                        error += diff * diff;
                    }
                }
            }
            error = std::sqrt(error / target.get_size_n());
            std::cout << "error " << name << " = " << error << std::endl;
        };

    p.set_boundary_type(PDEBoundaryType::Dirichlet);
    p.set_buffer(calc_p);

    ConcatPoissonSolver3D p_solver(&p);

    p.set_value(calc_laplacian_p);
    p_solver.solve();
    calc_error_centered("p_from_exact_rhs", p_A1, calc_p);

    field3 dudx(A1.nx, A1.ny, A1.nz);
    field3 dudy(A1.nx, A1.ny, A1.nz);
    field3 dudz(A1.nx, A1.ny, A1.nz);

    field3 dvdx(A1.nx, A1.ny, A1.nz);
    field3 dvdy(A1.nx, A1.ny, A1.nz);
    field3 dvdz(A1.nx, A1.ny, A1.nz);

    field3 dwdx(A1.nx, A1.ny, A1.nz);
    field3 dwdy(A1.nx, A1.ny, A1.nz);
    field3 dwdz(A1.nx, A1.ny, A1.nz);

    auto u = [&](int i, int j, int k) { return calc_u(i * hx, (j + 0.5) * hy, (k + 0.5) * hz); };
    auto v = [&](int i, int j, int k) { return calc_v((i + 0.5) * hx, j * hy, (k + 0.5) * hz); };
    auto w = [&](int i, int j, int k) { return calc_w((i + 0.5) * hx, (j + 0.5) * hy, k * hz); };

    for (int i = 0; i < p_A1.get_nx(); i++)
    {
        for (int j = 0; j < p_A1.get_ny(); j++)
        {
            for (int k = 0; k < p_A1.get_nz(); k++)
            {
                std::array<std::array<double, 3>, 3> L;

                double u_ijk = u(i, j, k);
                double u_im1 = u(i - 1, j, k);
                double u_ip1 = u(i + 1, j, k);
                double u_jm1 = u(i, j - 1, k);
                double u_jp1 = u(i, j + 1, k);
                double u_km1 = u(i, j, k - 1);
                double u_kp1 = u(i, j, k + 1);

                double u_ip1_jm1 = u(i + 1, j - 1, k);
                double u_ip1_jp1 = u(i + 1, j + 1, k);
                double u_ip1_km1 = u(i + 1, j, k - 1);
                double u_ip1_kp1 = u(i + 1, j, k + 1);

                // dudy $(u_{i, j+1, k} + u_{i+1, j+1, k}) - (u_{i, j-1, k} + u_{i+1, j-1, k})$
                // dudz $(u_{i, j, k+1} + u_{i+1, j, k+1}) - (u_{i, j, k-1} + u_{i+1, j, k-1})$

                L[0][0] = (u_ip1 - u_ijk) / hx;
                L[0][1] = (u_jp1 + u_ip1_jp1 - u_jm1 - u_ip1_jm1) / 4.0 / hy;
                L[0][2] = (u_kp1 + u_ip1_kp1 - u_km1 - u_ip1_km1) / 4.0 / hz;

                // debug
                dudx(i, j, k) = L[0][0];
                dudy(i, j, k) = L[0][1];
                dudz(i, j, k) = L[0][2];

                double v_ijk = v(i, j, k);
                double v_im1 = v(i - 1, j, k);
                double v_ip1 = v(i + 1, j, k);
                double v_jm1 = v(i, j - 1, k);
                double v_jp1 = v(i, j + 1, k);
                double v_km1 = v(i, j, k - 1);
                double v_kp1 = v(i, j, k + 1);

                double v_im1_jp1 = v(i - 1, j + 1, k);
                double v_ip1_jp1 = v(i + 1, j + 1, k);
                double v_jp1_km1 = v(i, j + 1, k - 1);
                double v_jp1_kp1 = v(i, j + 1, k + 1);

                // dvdx $(v_{i+1, j, k} + v_{i+1, j+1, k}) - (v_{i-1, j, k} + v_{i-1, j+1, k})$
                // dvdz $(v_{i, j, k+1} + v_{i, j+1, k+1}) - (v_{i, j, k-1} + v_{i, j+1, k-1})$

                L[1][0] = (v_ip1 + v_ip1_jp1 - v_im1 - v_im1_jp1) / 4.0 / hx;
                L[1][1] = (v_jp1 - v_ijk) / hy;
                L[1][2] = (v_kp1 + v_jp1_kp1 - v_km1 - v_jp1_km1) / 4.0 / hz;

                // debug
                dvdx(i, j, k) = L[1][0];
                dvdy(i, j, k) = L[1][1];
                dvdz(i, j, k) = L[1][2];

                double w_ijk = w(i, j, k);
                double w_im1 = w(i - 1, j, k);
                double w_ip1 = w(i + 1, j, k);
                double w_jm1 = w(i, j - 1, k);
                double w_jp1 = w(i, j + 1, k);
                double w_km1 = w(i, j, k - 1);
                double w_kp1 = w(i, j, k + 1);

                double w_im1_kp1 = w(i - 1, j, k + 1);
                double w_ip1_kp1 = w(i + 1, j, k + 1);
                double w_jm1_kp1 = w(i, j - 1, k + 1);
                double w_jp1_kp1 = w(i, j + 1, k + 1);

                // dzdx $(w_{i+1, j, k} + w_{i+1, j, k+1}) - (w_{i-1, j, k} + w_{i-1, j, k+1})$
                // dzdy $(w_{i, j+1, k} + w_{i, j+1, k+1}) - (w_{i, j-1, k} + w_{i, j-1, k+1})$

                L[2][0] = (w_ip1 + w_ip1_kp1 - w_im1 - w_im1_kp1) / 4.0 / hx;
                L[2][1] = (w_jp1 + w_jp1_kp1 - w_jm1 - w_jm1_kp1) / 4.0 / hy;
                L[2][2] = (w_kp1 - w_ijk) / hz;

                // debug
                dwdx(i, j, k) = L[2][0];
                dwdy(i, j, k) = L[2][1];
                dwdz(i, j, k) = L[2][2];

                p_A1(i, j, k) = 0.0;
                for (int m = 0; m < 3; m++)
                    for (int n = 0; n < 3; n++)
                        p_A1(i, j, k) += L[m][n] * L[n][m];
                p_A1(i, j, k) *= -rho;
            }
        }
    }

    calc_error_centered("rho_div_u_grad_u", p_A1, calc_neg_rho_div_u_grad_u);

    for (int i = 0; i < p_A1.get_nx(); i++)
    {
        for (int j = 0; j < p_A1.get_ny(); j++)
        {
            for (int k = 0; k < p_A1.get_nz(); k++)
            {
                double x = (i + 0.5) * hx;
                double y = (j + 0.5) * hy;
                double z = (k + 0.5) * hz;
                p_A1(i, j, k) += calc_rho_div_f(x, y, z);
            }
        }
    }

    calc_error_centered("rhs", p_A1, calc_laplacian_p);

    p_solver.solve();

    calc_error_centered("dudx", dudx, calc_dudx);
    calc_error_centered("dudy", dudy, calc_dudy);
    calc_error_centered("dudz", dudz, calc_dudz);
    calc_error_centered("dvdx", dvdx, calc_dvdx);
    calc_error_centered("dvdy", dvdy, calc_dvdy);
    calc_error_centered("dvdz", dvdz, calc_dvdz);
    calc_error_centered("dwdx", dwdx, calc_dwdx);
    calc_error_centered("dwdy", dwdy, calc_dwdy);
    calc_error_centered("dwdz", dwdz, calc_dwdz);

    double error_laplacian_p = 0.0;
    for (int i = 1; i < p_A1.get_nx() - 1; i++)
    {
        for (int j = 1; j < p_A1.get_ny() - 1; j++)
        {
            for (int k = 1; k < p_A1.get_nz() - 1; k++)
            {
                double x           = (i + 0.5) * hx;
                double y           = (j + 0.5) * hy;
                double z           = (k + 0.5) * hz;
                double laplacian_p = (p_A1(i + 1, j, k) + p_A1(i - 1, j, k) - 2.0 * p_A1(i, j, k)) / hx / hx +
                                     (p_A1(i, j + 1, k) + p_A1(i, j - 1, k) - 2.0 * p_A1(i, j, k)) / hy / hy +
                                     (p_A1(i, j, k + 1) + p_A1(i, j, k - 1) - 2.0 * p_A1(i, j, k)) / hz / hz;
                double diff = laplacian_p - calc_laplacian_p(x, y, z);
                error_laplacian_p += diff * diff;
            }
        }
    }
    error_laplacian_p = std::sqrt(error_laplacian_p / (p_A1.get_nx() - 1) / (p_A1.get_ny() - 1) / (p_A1.get_nz() - 1));
    std::cout << "error_laplacian_p = " << error_laplacian_p << std::endl;

    double sum_num = 0.0, sum_exact = 0.0;
    sum_num = p_A1.sum();
    for (int i = 0; i < p_A1.get_nx(); i++)
    {
        for (int j = 0; j < p_A1.get_ny(); j++)
        {
            for (int k = 0; k < p_A1.get_nz(); k++)
            {
                double x = (i + 0.5) * hx;
                double y = (j + 0.5) * hy;
                double z = (k + 0.5) * hz;
                sum_exact += calc_p(x, y, z);
            }
        }
    }
    double avg_diff = (sum_num - sum_exact) / p_A1.get_size_n();
    p_A1 -= avg_diff;
    calc_error_centered("p_from_sol_rhs", p_A1, calc_p);
}