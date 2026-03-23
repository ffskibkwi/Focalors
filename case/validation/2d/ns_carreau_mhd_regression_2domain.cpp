#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/case_base.hpp"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr double kDiffusionDtSafety = 0.20;
    constexpr double kMagneticDtSafety  = 0.50;
    constexpr double kSmall             = 1.0e-12;

    struct RegressionCase : public CaseBase
    {
        static constexpr double kDefaultCarreauN = 0.3568;
        static constexpr double kDefaultMuRef    = 0.00596;

        explicit RegressionCase(int argc, char* argv[])
            : CaseBase(argc, argv)
        {}

        void read_paras() override
        {
            CaseBase::read_paras();

            IO::read_number(para_map, "nx", nx);
            IO::read_number(para_map, "ny", ny);
            IO::read_number(para_map, "half_height", half_height);
            IO::read_number(para_map, "length_over_h", length_over_h);
            IO::read_bool(para_map, "split_domain", split_domain);

            IO::read_number(para_map, "Re", Re);
            IO::read_number(para_map, "Ha", Ha);
            IO::read_number(para_map, "dp_dx", dp_dx);
            IO::read_number(para_map, "Bx", Bx);
            IO::read_number(para_map, "By", By);
            IO::read_number(para_map, "Bz", Bz);

            IO::read_number(para_map, "dt_factor", dt_factor);
            IO::read_number(para_map, "corr_iter", corr_iter);
            IO::read_number(para_map, "pv_output_step", pv_output_step);

            IO::read_number(para_map, "gmres_m", gmres_m);
            IO::read_number(para_map, "gmres_tol", gmres_tol);
            IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

            IO::read_number(para_map, "model_type", model_type);
            IO::read_number(para_map, "n_index", n_index);
            IO::read_number(para_map, "mu_0", mu_0);
            IO::read_number(para_map, "mu_inf", mu_inf);
            IO::read_number(para_map, "lambda", lambda);
            IO::read_number(para_map, "a", a);
            IO::read_number(para_map, "mu_min_pl", mu_min_pl);
            IO::read_number(para_map, "mu_max_pl", mu_max_pl);
            IO::read_number(para_map, "mu_ref", mu_ref);
            IO::read_bool(para_map, "use_dimensionless_viscosity", use_dimensionless_viscosity);

            if (para_map.find("max_step") == para_map.end())
                max_step = 600;
            if (para_map.find("step_to_save") == para_map.end())
                step_to_save = max_step;

            validate();
        }

        bool record_paras() override
        {
            if (!CaseBase::record_paras())
                return false;

            paras_record.record("nx", nx)
                .record("ny", ny)
                .record("half_height", half_height)
                .record("length_over_h", length_over_h)
                .record("Lx", getLx())
                .record("Ly", getLy())
                .record("split_domain", split_domain ? 1 : 0)
                .record("topology", std::string("split"))
                .record("Re", Re)
                .record("Ha", Ha)
                .record("dp_dx", dp_dx)
                .record("Bx", Bx)
                .record("By", By)
                .record("Bz", Bz)
                .record("dt_factor", dt_factor)
                .record("corr_iter", corr_iter)
                .record("pv_output_step", pv_output_step)
                .record("gmres_m", gmres_m)
                .record("gmres_tol", gmres_tol)
                .record("gmres_max_iter", gmres_max_iter)
                .record("model_type", model_type)
                .record("n_index", n_index)
                .record("mu_0", mu_0)
                .record("mu_inf", mu_inf)
                .record("lambda", lambda)
                .record("a", a)
                .record("mu_min_pl", mu_min_pl)
                .record("mu_max_pl", mu_max_pl)
                .record("mu_ref", mu_ref)
                .record("use_dimensionless_viscosity", use_dimensionless_viscosity ? 1 : 0);

            return true;
        }

        std::array<int, 3> checkpoint_steps() const
        {
            if (pv_output_step <= 0)
                throw std::runtime_error("pv_output_step must be positive.");

            const int step_1 = pv_output_step;
            const int step_2 = 2 * pv_output_step;
            const int step_3 = max_step;

            if (step_2 >= step_3)
                throw std::runtime_error("Regression case requires max_step > 2 * pv_output_step.");

            return {step_1, step_2, step_3};
        }

        double getLx() const { return length_over_h * half_height; }

        double getLy() const { return 2.0 * half_height; }

        void validate() const
        {
            if (!split_domain)
                throw std::runtime_error("This regression case only supports split_domain=1.");
            if (nx < 4 || ny < 4)
                throw std::runtime_error("Invalid grid: nx and ny must both be >= 4.");
            if (max_step <= 0)
                throw std::runtime_error("max_step must be positive.");
            if (model_type != 2)
                throw std::runtime_error("This regression case is fixed to Carreau model_type=2.");
            if (std::abs(Ha) <= 0.0)
                throw std::runtime_error("Ha must be non-zero for this MHD regression case.");
            if (std::abs(Bz) <= 0.0)
                throw std::runtime_error("Bz must be non-zero to exercise the external phi MHD path.");
        }

        int    nx                         = 80;
        int    ny                         = 40;
        double half_height                = 1.0;
        double length_over_h              = 20.0;
        bool   split_domain               = true;

        double Re    = 20.0;
        double Ha    = 10.0;
        double dp_dx = -1.0;
        double Bx    = 0.0;
        double By    = 0.0;
        double Bz    = 1.0;

        double dt_factor      = 0.10;
        int    corr_iter      = 2;
        int    pv_output_step = 200;

        int    gmres_m        = 30;
        double gmres_tol      = 1.0e-7;
        int    gmres_max_iter = 1000;

        int    model_type                  = 2;
        double n_index                     = kDefaultCarreauN;
        double mu_0                        = 0.056;
        double mu_inf                      = 0.00345;
        double lambda                      = 3.313;
        double a                           = 2.0;
        double mu_min_pl                   = 0.00345;
        double mu_max_pl                   = 0.056;
        double mu_ref                      = kDefaultMuRef;
        bool   use_dimensionless_viscosity = true;
    };

    struct TimeStepSelection
    {
        double convective_dt      = 0.0;
        double diffusion_dt_limit = std::numeric_limits<double>::infinity();
        double magnetic_dt_limit  = std::numeric_limits<double>::infinity();
        double selected_dt        = 0.0;
        double viscosity_upper    = 0.0;
        double magnetic_factor_sq = 0.0;
    };

    struct DomainBundle
    {
        Geometry2D                                    geometry;
        std::vector<std::unique_ptr<Domain2DUniform>> holders;
        std::vector<Domain2DUniform*>                 domains;
        Domain2DUniform*                              left_domain  = nullptr;
        Domain2DUniform*                              right_domain = nullptr;
        int                                           nx_left      = 0;
        int                                           nx_right     = 0;
        double                                        hx           = 0.0;
        double                                        hy           = 0.0;
    };

    struct SolverState
    {
        Variable2D u_var;
        Variable2D v_var;
        Variable2D p_var;

        std::unique_ptr<Variable2D> phi_var;
        std::unique_ptr<Variable2D> mu_var;
        std::unique_ptr<Variable2D> tau_xx_var;
        std::unique_ptr<Variable2D> tau_yy_var;
        std::unique_ptr<Variable2D> tau_xy_var;

        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> u_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> v_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> p_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> phi_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> mu_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> tau_xx_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> tau_yy_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> tau_xy_fields;

        explicit SolverState(Geometry2D& geometry)
            : u_var("u")
            , v_var("v")
            , p_var("p")
        {
            u_var.set_geometry(geometry);
            v_var.set_geometry(geometry);
            p_var.set_geometry(geometry);
        }
    };

    TimeStepSelection select_time_step(const RegressionCase& case_param, const PhysicsConfig& physics_cfg)
    {
        TimeStepSelection selection;
        const double      h_min =
            std::min(case_param.getLx() / static_cast<double>(case_param.nx), case_param.getLy() / static_cast<double>(case_param.ny));

        selection.convective_dt   = case_param.dt_factor * h_min;
        selection.viscosity_upper = std::max(physics_cfg.mu_max, case_param.mu_0);
        selection.magnetic_factor_sq = case_param.Bx * case_param.Bx + case_param.By * case_param.By + case_param.Bz * case_param.Bz;

        if (selection.viscosity_upper > 0.0)
        {
            selection.diffusion_dt_limit =
                kDiffusionDtSafety * h_min * h_min / std::max(selection.viscosity_upper, kSmall);
        }

        if (std::abs(case_param.Ha) > 0.0 && selection.magnetic_factor_sq > 0.0)
        {
            selection.magnetic_dt_limit =
                kMagneticDtSafety * case_param.Re / (case_param.Ha * case_param.Ha * selection.magnetic_factor_sq);
        }

        selection.selected_dt =
            std::min(selection.convective_dt, std::min(selection.diffusion_dt_limit, selection.magnetic_dt_limit));

        if (!std::isfinite(selection.selected_dt) || selection.selected_dt <= 0.0)
            throw std::runtime_error("Failed to determine a positive time step for the regression case.");

        return selection;
    }

    DomainBundle build_domains(const RegressionCase& case_param)
    {
        DomainBundle bundle;

        bundle.nx_left  = case_param.nx / 2;
        bundle.nx_right = case_param.nx - bundle.nx_left;
        if (bundle.nx_left < 2 || bundle.nx_right < 2)
            throw std::runtime_error("Split-domain mode requires at least 2 cells in each sub-domain.");

        const double lx       = case_param.getLx();
        const double ly       = case_param.getLy();
        const double hx       = lx / static_cast<double>(case_param.nx);
        const double lx_left  = hx * static_cast<double>(bundle.nx_left);
        const double lx_right = hx * static_cast<double>(bundle.nx_right);

        bundle.holders.emplace_back(new Domain2DUniform(bundle.nx_left, case_param.ny, lx_left, ly, "D0"));
        bundle.holders.emplace_back(new Domain2DUniform(bundle.nx_right, case_param.ny, lx_right, ly, "D1"));

        bundle.domains.push_back(bundle.holders[0].get());
        bundle.domains.push_back(bundle.holders[1].get());
        bundle.left_domain  = bundle.domains.front();
        bundle.right_domain = bundle.domains.back();

        bundle.geometry.connect(bundle.left_domain, LocationType::XPositive, bundle.right_domain);
        bundle.geometry.axis(bundle.left_domain, LocationType::XNegative);
        bundle.geometry.axis(bundle.left_domain, LocationType::YNegative);
        bundle.geometry.check();
        bundle.geometry.solve_prepare();

        bundle.hx = bundle.left_domain->get_hx();
        bundle.hy = bundle.left_domain->get_hy();
        return bundle;
    }

    void add_field_for_all_domains(Variable2D&                                                    var,
                                   const std::vector<Domain2DUniform*>&                           domains,
                                   std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>>& storages,
                                   VariablePositionType                                           pos)
    {
        for (auto* domain : domains)
        {
            storages[domain] = std::unique_ptr<field2>(new field2(var.name + "_" + domain->name));

            if (pos == VariablePositionType::XFace)
                var.set_x_edge_field(domain, *storages[domain]);
            else if (pos == VariablePositionType::YFace)
                var.set_y_edge_field(domain, *storages[domain]);
            else if (pos == VariablePositionType::Center)
                var.set_center_field(domain, *storages[domain]);
            else if (pos == VariablePositionType::Corner)
                var.set_corner_field(domain, *storages[domain]);
            else
                throw std::runtime_error("Unsupported field position.");
        }
    }

    void build_state(DomainBundle& bundle, SolverState& state)
    {
        add_field_for_all_domains(state.u_var, bundle.domains, state.u_fields, VariablePositionType::XFace);
        add_field_for_all_domains(state.v_var, bundle.domains, state.v_fields, VariablePositionType::YFace);
        add_field_for_all_domains(state.p_var, bundle.domains, state.p_fields, VariablePositionType::Center);

        state.phi_var = std::make_unique<Variable2D>("phi");
        state.phi_var->set_geometry(bundle.geometry);
        add_field_for_all_domains(*state.phi_var, bundle.domains, state.phi_fields, VariablePositionType::Center);
        state.phi_var->set_boundary_type(PDEBoundaryType::Neumann);

        state.mu_var     = std::make_unique<Variable2D>("mu");
        state.tau_xx_var = std::make_unique<Variable2D>("tau_xx");
        state.tau_yy_var = std::make_unique<Variable2D>("tau_yy");
        state.tau_xy_var = std::make_unique<Variable2D>("tau_xy");

        state.mu_var->set_geometry(bundle.geometry);
        state.tau_xx_var->set_geometry(bundle.geometry);
        state.tau_yy_var->set_geometry(bundle.geometry);
        state.tau_xy_var->set_geometry(bundle.geometry);

        add_field_for_all_domains(*state.mu_var, bundle.domains, state.mu_fields, VariablePositionType::Corner);
        add_field_for_all_domains(*state.tau_xx_var, bundle.domains, state.tau_xx_fields, VariablePositionType::Center);
        add_field_for_all_domains(*state.tau_yy_var, bundle.domains, state.tau_yy_fields, VariablePositionType::Center);
        add_field_for_all_domains(*state.tau_xy_var, bundle.domains, state.tau_xy_fields, VariablePositionType::Corner);

        state.mu_var->set_boundary_type(PDEBoundaryType::Neumann);
        state.tau_xy_var->set_boundary_type(PDEBoundaryType::Neumann);
    }

    void set_dirichlet(Variable2D& var, Domain2DUniform* domain, LocationType loc, double value)
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(domain, loc, value);
    }

    void set_neumann(Variable2D& var, Domain2DUniform* domain, LocationType loc, double value = 0.0)
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Neumann);
        var.set_boundary_value(domain, loc, value);
    }

    void setup_boundary_conditions(const DomainBundle& bundle,
                                   Variable2D&         u,
                                   Variable2D&         v,
                                   Variable2D&         p,
                                   Variable2D&         phi)
    {
        for (auto* domain : bundle.domains)
        {
            set_dirichlet(u, domain, LocationType::YNegative, 0.0);
            set_dirichlet(u, domain, LocationType::YPositive, 0.0);
            set_dirichlet(v, domain, LocationType::YNegative, 0.0);
            set_dirichlet(v, domain, LocationType::YPositive, 0.0);
            set_neumann(p, domain, LocationType::YNegative, 0.0);
            set_neumann(p, domain, LocationType::YPositive, 0.0);

            set_neumann(phi, domain, LocationType::YNegative, 0.0);
            set_neumann(phi, domain, LocationType::YPositive, 0.0);
        }

        set_neumann(u, bundle.left_domain, LocationType::XNegative, 0.0);
        set_neumann(u, bundle.right_domain, LocationType::XPositive, 0.0);
        set_dirichlet(v, bundle.left_domain, LocationType::XNegative, 0.0);
        set_dirichlet(v, bundle.right_domain, LocationType::XPositive, 0.0);
        set_neumann(p, bundle.left_domain, LocationType::XNegative, 0.0);
        set_neumann(p, bundle.right_domain, LocationType::XPositive, 0.0);

        set_dirichlet(phi, bundle.left_domain, LocationType::XNegative, 0.0);
        set_dirichlet(phi, bundle.right_domain, LocationType::XPositive, 0.0);
    }

    void initialize_fields(const RegressionCase& case_param, const DomainBundle& bundle, SolverState& state)
    {
        for (auto* domain : bundle.domains)
        {
            state.u_var.field_map[domain]->clear(0.0);
            state.v_var.field_map[domain]->clear(0.0);
            state.p_var.field_map[domain]->clear(0.0);
            state.phi_var->field_map[domain]->clear(0.0);
            state.mu_var->field_map[domain]->clear(case_param.mu_0);
            state.tau_xx_var->field_map[domain]->clear(0.0);
            state.tau_yy_var->field_map[domain]->clear(0.0);
            state.tau_xy_var->field_map[domain]->clear(0.0);
        }
    }

    void apply_streamwise_pressure_gradient_force(const RegressionCase& case_param, SolverState& state)
    {
        const double delta_u = TimeAdvancingConfig::Get().dt * (-case_param.dp_dx);
        if (std::abs(delta_u) <= kSmall)
            return;

        for (auto* domain : state.u_var.geometry->domains)
        {
            field2& u_field = *state.u_var.field_map[domain];
            for (int i = 0; i < u_field.get_nx(); ++i)
            {
                for (int j = 0; j < u_field.get_ny(); ++j)
                    u_field(i, j) += delta_u;
            }
        }
    }

    void ensure_finite_field(const Variable2D& var, int step)
    {
        for (auto* domain : var.geometry->domains)
        {
            const field2& field = *var.field_map.at(domain);
            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    if (!std::isfinite(field(i, j)))
                    {
                        throw std::runtime_error("Non-finite value detected in " + var.name + " at step=" +
                                                 std::to_string(step) + ", domain=" + domain->name + ", i=" +
                                                 std::to_string(i) + ", j=" + std::to_string(j));
                    }
                }
            }
        }
    }

    void write_face_field_with_positive_buffer(const Variable2D& var, const std::string& output_stem)
    {
        for (auto* domain : var.geometry->domains)
        {
            const auto field_it = var.field_map.find(domain);
            const auto buffer_map_it = var.buffer_map.find(domain);
            if (field_it == var.field_map.end() || buffer_map_it == var.buffer_map.end())
                throw std::runtime_error("Missing field or buffer storage for " + var.name + " on domain " + domain->name);

            const auto& positive_buffers = buffer_map_it->second;
            if (var.position_type == VariablePositionType::XFace)
            {
                const auto buffer_it = positive_buffers.find(LocationType::XPositive);
                if (buffer_it == positive_buffers.end())
                    throw std::runtime_error("Missing XPositive buffer for " + var.name + " on domain " + domain->name);
                IO::write_csv(*field_it->second, buffer_it->second, output_stem + "_" + domain->name, VariablePositionType::XFace);
            }
            else if (var.position_type == VariablePositionType::YFace)
            {
                const auto buffer_it = positive_buffers.find(LocationType::YPositive);
                if (buffer_it == positive_buffers.end())
                    throw std::runtime_error("Missing YPositive buffer for " + var.name + " on domain " + domain->name);
                IO::write_csv(*field_it->second, buffer_it->second, output_stem + "_" + domain->name, VariablePositionType::YFace);
            }
            else
            {
                throw std::runtime_error("write_face_field_with_positive_buffer only supports face-centered variables.");
            }
        }
    }

    void write_checkpoint_fields(const RegressionCase& case_param, int step, SolverState& state)
    {
        write_face_field_with_positive_buffer(state.u_var, case_param.root_dir + "/u/u_" + std::to_string(step));
        write_face_field_with_positive_buffer(state.v_var, case_param.root_dir + "/v/v_" + std::to_string(step));
    }

    void write_final_fields(const RegressionCase& case_param, int final_step, SolverState& state)
    {
        write_face_field_with_positive_buffer(state.u_var, case_param.root_dir + "/final/u_" + std::to_string(final_step));
        write_face_field_with_positive_buffer(state.v_var, case_param.root_dir + "/final/v_" + std::to_string(final_step));
    }

    void write_summary_csv(const RegressionCase&     case_param,
                           const DomainBundle&       bundle,
                           const TimeStepSelection&  dt_selection,
                           const std::array<int, 3>& checkpoints,
                           int                       final_step)
    {
        std::ofstream out(case_param.root_dir + "/regression_summary.csv");
        if (!out.is_open())
            throw std::runtime_error("Failed to open regression_summary.csv for writing.");

        out << std::setprecision(16);
        out << "topology,split_domain,num_domains,nx_total,ny,nx_left,nx_right,Lx,Ly,hx,hy,Re,Ha,Bx,By,Bz,"
               "model_type,n_index,mu_0,mu_inf,lambda,a,mu_ref,use_dimensionless_viscosity,dt,dt_convective,"
               "dt_diffusion_limit,dt_magnetic_limit,viscosity_upper_bound,magnetic_factor_sq,max_step,pv_output_step,"
               "step_1,step_2,step_3,final_step,pressure_drive_realization\n";
        out << "split,"
            << (case_param.split_domain ? 1 : 0) << ","
            << bundle.domains.size() << ","
            << case_param.nx << ","
            << case_param.ny << ","
            << bundle.nx_left << ","
            << bundle.nx_right << ","
            << case_param.getLx() << ","
            << case_param.getLy() << ","
            << bundle.hx << ","
            << bundle.hy << ","
            << case_param.Re << ","
            << case_param.Ha << ","
            << case_param.Bx << ","
            << case_param.By << ","
            << case_param.Bz << ","
            << case_param.model_type << ","
            << case_param.n_index << ","
            << case_param.mu_0 << ","
            << case_param.mu_inf << ","
            << case_param.lambda << ","
            << case_param.a << ","
            << case_param.mu_ref << ","
            << (case_param.use_dimensionless_viscosity ? 1 : 0) << ","
            << TimeAdvancingConfig::Get().dt << ","
            << dt_selection.convective_dt << ","
            << dt_selection.diffusion_dt_limit << ","
            << dt_selection.magnetic_dt_limit << ","
            << dt_selection.viscosity_upper << ","
            << dt_selection.magnetic_factor_sq << ","
            << case_param.max_step << ","
            << case_param.pv_output_step << ","
            << checkpoints[0] << ","
            << checkpoints[1] << ","
            << checkpoints[2] << ","
            << final_step << ","
            << "equivalent_uniform_source_from_dpdx\n";
    }
} // namespace

int main(int argc, char* argv[])
{
    try
    {
        RegressionCase case_param(argc, argv);
        case_param.read_paras();
        case_param.record_paras();
        const std::array<int, 3> checkpoints = case_param.checkpoint_steps();

        EnvironmentConfig& env_cfg    = EnvironmentConfig::Get();
        TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
        PhysicsConfig&       physics_cfg = PhysicsConfig::Get();

        env_cfg.showGmresRes    = false;
        env_cfg.showCurrentStep = false;

        physics_cfg.set_Re(case_param.Re);
        physics_cfg.set_model_type(case_param.model_type);
        physics_cfg.set_carreau_dimensionless(case_param.mu_0,
                                              case_param.mu_inf,
                                              case_param.a,
                                              case_param.lambda,
                                              case_param.n_index,
                                              case_param.Re,
                                              case_param.mu_ref,
                                              case_param.use_dimensionless_viscosity,
                                              case_param.mu_min_pl,
                                              case_param.mu_max_pl);
        physics_cfg.set_enable_mhd(true);
        physics_cfg.set_Ha(case_param.Ha);
        physics_cfg.set_magnetic_field(case_param.Bx, case_param.By, case_param.Bz);

        const TimeStepSelection dt_selection = select_time_step(case_param, physics_cfg);

        time_cfg.set_dt(dt_selection.selected_dt);
        time_cfg.set_num_iterations(case_param.max_step);
        time_cfg.set_t_max(case_param.max_step * dt_selection.selected_dt);
        time_cfg.set_corr_iter(case_param.corr_iter);

        case_param.paras_record.record("dt", time_cfg.dt)
            .record("dt_convective", dt_selection.convective_dt)
            .record("dt_diffusion_limit", dt_selection.diffusion_dt_limit)
            .record("dt_magnetic_limit", dt_selection.magnetic_dt_limit)
            .record("viscosity_upper_bound", dt_selection.viscosity_upper)
            .record("magnetic_factor_sq", dt_selection.magnetic_factor_sq)
            .record("pressure_drive_realization", std::string("equivalent_uniform_source_from_dpdx"))
            .record("step_1", checkpoints[0])
            .record("step_2", checkpoints[1])
            .record("step_3", checkpoints[2])
            .record("final_step", checkpoints[2]);

        DomainBundle bundle = build_domains(case_param);
        SolverState  state(bundle.geometry);
        build_state(bundle, state);
        setup_boundary_conditions(bundle, state.u_var, state.v_var, state.p_var, *state.phi_var);
        initialize_fields(case_param, bundle, state);

        ConcatPoissonSolver2D p_solver(&state.p_var);
        ConcatNSSolver2D      ns_solver(&state.u_var, &state.v_var, &state.p_var, &p_solver);
        ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);
        ns_solver.init_nonnewton(
            state.mu_var.get(), state.tau_xx_var.get(), state.tau_yy_var.get(), state.tau_xy_var.get(), state.phi_var.get());

        std::cout << "2-domain NS+Carreau+MHD regression case" << std::endl;
        std::cout << "  Root dir: " << case_param.root_dir << std::endl;
        std::cout << "  Grid: nx=" << case_param.nx << ", ny=" << case_param.ny << ", split_domain=1" << std::endl;
        std::cout << "  Physics: Re=" << case_param.Re << ", Ha=" << case_param.Ha << ", B=(" << case_param.Bx << ","
                  << case_param.By << "," << case_param.Bz << "), model_type=" << case_param.model_type << std::endl;
        std::cout << "  Time stepping: dt=" << time_cfg.dt << ", max_step=" << case_param.max_step
                  << ", corr_iter=" << case_param.corr_iter << std::endl;
        std::cout << "  Checkpoints: " << checkpoints[0] << ", " << checkpoints[1] << ", " << checkpoints[2] << std::endl;

        for (int step = 1; step <= case_param.max_step; ++step)
        {
            ns_solver.solve_nonnewton();
            apply_streamwise_pressure_gradient_force(case_param, state);
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();

            ensure_finite_field(state.u_var, step);
            ensure_finite_field(state.v_var, step);

            if (step == checkpoints[0] || step == checkpoints[1] || step == checkpoints[2])
            {
                write_checkpoint_fields(case_param, step, state);
                std::cout << "  Saved checkpoint step=" << step << std::endl;
            }
        }

        ns_solver.phys_boundary_update();
        ns_solver.nondiag_shared_boundary_update();
        ns_solver.diag_shared_boundary_update();
        write_final_fields(case_param, checkpoints[2], state);
        write_summary_csv(case_param, bundle, dt_selection, checkpoints, checkpoints[2]);

        std::cout << "Finished. Final fields saved under " << case_param.root_dir + "/final" << std::endl;
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return 2;
    }
}
