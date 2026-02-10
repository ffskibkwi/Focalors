#include "csv_writer_2d.h"
#include "common.h"

#include <fstream>

namespace IO
{
    namespace
    {
        const char* position_type_to_string(VariablePositionType type)
        {
            switch (type)
            {
                case VariablePositionType::Center:
                    return "Center";
                case VariablePositionType::XFace:
                    return "XFace";
                case VariablePositionType::YFace:
                    return "YFace";
                case VariablePositionType::Corner:
                    return "Corner";
                default:
                    return "Null";
            }
        }

        void position_shift(VariablePositionType type, double& shift_x, double& shift_y)
        {
            switch (type)
            {
                case VariablePositionType::Center:
                    shift_x = 0.5;
                    shift_y = 0.5;
                    break;
                case VariablePositionType::XFace:
                    shift_x = 0.0;
                    shift_y = 0.5;
                    break;
                case VariablePositionType::YFace:
                    shift_x = 0.5;
                    shift_y = 0.0;
                    break;
                case VariablePositionType::Corner:
                    shift_x = 0.0;
                    shift_y = 0.0;
                    break;
                default:
                    shift_x = 0.0;
                    shift_y = 0.0;
                    break;
            }
        }
    } // namespace

    bool write_csv(double** value, int nx, int ny, const std::string& filename)
    {
        fs::path path(filename);
        fs::path dir = path.parent_path();
        IO::create_directory(dir);

        std::ofstream outfile(filename + ".csv");

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (j == (ny - 1))
                {
                    outfile << value[i][j] << std::endl;
                }
                else
                {
                    outfile << value[i][j] << ",";
                }
            }
        }
        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }

    bool write_csv(field2& field, const std::string& filename)
    {
        fs::path path(filename);
        fs::path dir = path.parent_path();
        IO::create_directory(dir);

        std::ofstream outfile(filename + ".csv");

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }

        for (int i = 0; i < field.get_nx(); i++)
        {
            for (int j = 0; j < field.get_ny(); j++)
            {
                if (j == (field.get_ny() - 1))
                {
                    outfile << field(i, j) << std::endl;
                }
                else
                {
                    outfile << field(i, j) << ",";
                }
            }
        }
        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }

    bool read_csv(field2& field, const std::string& filename)
    {
        std::ifstream infile(filename + ".csv");

        if (!infile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }

        std::string line;
        int         i = 0;

        while (std::getline(infile, line))
        {
            std::stringstream ss(line);
            std::string       value;
            int               j = 0;

            while (std::getline(ss, value, ','))
            {
                try
                {
                    double numeric_value = std::stod(value);
                    field(i, j)          = numeric_value;
                    j++;
                }
                catch (const std::invalid_argument&)
                {
                    std::cerr << "Invalid number at i " << i << ", j " << j << ": " << value << std::endl;
                }
            }
            i++;
        }

        infile.close();
        return true;
    }

    bool write_csv(field2& field, double* buffer, const std::string& filename, VariablePositionType pos_type)
    {
        fs::path path(filename);
        fs::path dir = path.parent_path();
        IO::create_directory(dir);

        std::ofstream outfile(filename + ".csv");

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }

        int nx = field.get_nx();
        int ny = field.get_ny();

        if (pos_type == VariablePositionType::XFace)
        {
            for (int i = 0; i < nx + 1; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    double val = 0;
                    if (i == nx)
                    {
                        val = buffer[j];
                    }
                    else
                    {
                        val = field(i, j);
                    }

                    if (j == (ny - 1))
                    {
                        outfile << val << std::endl;
                    }
                    else
                    {
                        outfile << val << ",";
                    }
                }
            }
        }
        else if (pos_type == VariablePositionType::YFace)
        {
            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny + 1; j++)
                {
                    double val = 0;
                    if (j == ny)
                    {
                        val = buffer[i];
                    }
                    else
                    {
                        val = field(i, j);
                    }

                    if (j == ny)
                    {
                        outfile << val << std::endl;
                    }
                    else
                    {
                        outfile << val << ",";
                    }
                }
            }
        }
        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }

    bool write_csv(const Variable2D& var, const std::string& filename)
    {
        auto& domains        = var.geometry->domains;
        auto& field_map      = var.field_map;
        auto& buffer_map     = var.buffer_map;
        auto& boundary_types = var.boundary_type_map;

        for (auto& domain : domains)
        {
            try
            {
                auto& field         = field_map.at(domain);
                auto& buffers       = buffer_map.at(domain);
                auto& boundary_type = boundary_types.at(domain);

                int nx = field->get_nx();
                int ny = field->get_ny();
                if (var.position_type == VariablePositionType::XFace)
                {
                    if (boundary_type.at(LocationType::Right) == PDEBoundaryType::Adjacented)
                        write_csv(*field, filename + "_" + domain->name);
                    else
                        write_csv(*field,
                                  buffers.at(LocationType::Right),
                                  filename + "_" + domain->name,
                                  VariablePositionType::XFace);
                }
                else if (var.position_type == VariablePositionType::YFace)
                {
                    if (boundary_type.at(LocationType::Up) == PDEBoundaryType::Adjacented)
                        write_csv(*field, filename + "_" + domain->name);
                    else
                        write_csv(*field,
                                  buffers.at(LocationType::Up),
                                  filename + "_" + domain->name,
                                  VariablePositionType::YFace);
                }
                else
                {
                    write_csv(*field, filename + "_" + domain->name);
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "[write_csv] Error: " << e.what() << std::endl;
                return false;
            }
        }
        return true;
    }

    bool matlab_read_var(const Variable2D& var, const std::string& filename)
    {
        if (!var.geometry)
        {
            std::cerr << "[matlab_read_var] Error: variable has no geometry" << std::endl;
            return false;
        }

        fs::path    path(filename);
        fs::path    dir         = path.parent_path();
        std::string stem        = path.stem().string();
        std::string base        = stem;
        std::string ext         = path.extension().string();
        std::string read_suffix = "_read";

        if (ext == ".m")
        {
            if (base.size() > read_suffix.size() &&
                base.compare(base.size() - read_suffix.size(), read_suffix.size(), read_suffix) == 0)
            {
                base = base.substr(0, base.size() - read_suffix.size());
            }
        }

        fs::path csv_base_path = dir / base;
        fs::path script_path   = dir / (base + "_read.m");

        if (!dir.empty())
            IO::create_directory(dir.string());

        std::ofstream outfile(script_path.string());

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << script_path.string() << std::endl;
            return false;
        }

        double shift_x = 0.0;
        double shift_y = 0.0;
        position_shift(var.position_type, shift_x, shift_y);

        outfile << "% Auto-generated by IO::matlab_read_var\n";
        outfile << "base = '" << csv_base_path.generic_string() << "';\n";
        outfile << "pos_type = '" << position_type_to_string(var.position_type) << "';\n";
        outfile << "shift_x = " << shift_x << ";\n";
        outfile << "shift_y = " << shift_y << ";\n";
        outfile << "domains = struct([]);\n\n";

        int idx = 1;
        for (auto& domain : var.geometry->domains)
        {
            outfile << "domains(" << idx << ").name = '" << domain->name << "';\n";
            outfile << "domains(" << idx << ").offset = [" << domain->get_offset_x() << ", " << domain->get_offset_y()
                    << "];\n";
            outfile << "domains(" << idx << ").hx = " << domain->get_hx() << ";\n";
            outfile << "domains(" << idx << ").hy = " << domain->get_hy() << ";\n";
            outfile << "domains(" << idx << ").nx = " << domain->get_nx() << ";\n";
            outfile << "domains(" << idx << ").ny = " << domain->get_ny() << ";\n";
            outfile << "domains(" << idx << ").lx = " << domain->get_lx() << ";\n";
            outfile << "domains(" << idx << ").ly = " << domain->get_ly() << ";\n";
            outfile << "domains(" << idx << ").field = readmatrix([base '_" << domain->name << ".csv']);\n";
            outfile << "domains(" << idx << ").size = size(domains(" << idx << ").field);\n";
            outfile << "domains(" << idx << ").x = domains(" << idx << ").offset(1) + (shift_x + (0:domains(" << idx
                    << ").size(1)-1)) * domains(" << idx << ").hx;\n";
            outfile << "domains(" << idx << ").y = domains(" << idx << ").offset(2) + (shift_y + (0:domains(" << idx
                    << ").size(2)-1)) * domains(" << idx << ").hy;\n\n";
            idx++;
        }

        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }
} // namespace IO
