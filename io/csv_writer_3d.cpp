#include "csv_writer_3d.h"
#include "common.h"

#include <fstream>

namespace IO
{
    bool write_csv(field3& field, const std::string& filename)
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
                for (int k = 0; k < field.get_nz(); k++)
                {
                    if (k == (field.get_nz() - 1))
                    {
                        outfile << field(i, j, k) << std::endl;
                    }
                    else
                    {
                        outfile << field(i, j, k) << ",";
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

    bool read_csv(field3& field, const std::string& filename)
    {
        std::ifstream infile(filename + ".csv");

        if (!infile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }

        std::string line;
        int         i = 0, j = 0;

        while (std::getline(infile, line))
        {
            std::stringstream ss(line);
            std::string       value;
            int               k = 0;

            while (std::getline(ss, value, ','))
            {
                try
                {
                    double numeric_value = std::stod(value);
                    field(i, j, k)       = numeric_value;
                    k++;
                }
                catch (const std::invalid_argument&)
                {
                    std::cerr << "Invalid number at i = " << i << ", j = " << j << ", k = " << k << ": " << value
                              << std::endl;
                }
            }
            j++;
            if (j >= field.get_ny())
            {
                j = 0;
                i++;
            }
        }

        infile.close();
        return true;
    }

    bool write_csv(field3& field, field2& buffer, const std::string& filename, VariablePositionType pos_type)
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
        int nz = field.get_nz();

        if (pos_type == VariablePositionType::XFace)
        {
            for (int i = 0; i < nx + 1; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        double val = 0;
                        if (i == nx)
                        {
                            val = buffer(j, k);
                        }
                        else
                        {
                            val = field(i, j, k);
                        }

                        if (k == nz - 1)
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
        }
        else if (pos_type == VariablePositionType::YFace)
        {
            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny + 1; j++)
                {
                    for (int k = 0; k < nz; k++)
                    {
                        double val = 0;
                        if (j == ny)
                        {
                            val = buffer(i, k);
                        }
                        else
                        {
                            val = field(i, j, k);
                        }

                        if (k == nz - 1)
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
        }
        else if (pos_type == VariablePositionType::ZFace)
        {
            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz + 1; k++)
                    {
                        double val = 0;
                        if (k == nz)
                        {
                            val = buffer(i, j);
                        }
                        else
                        {
                            val = field(i, j, k);
                        }

                        if (k == nz)
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
        }
        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }

    bool write_csv(const Variable3D& var, const std::string& filename)
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
                int nz = field->get_nz();
                if (var.position_type == VariablePositionType::XFace)
                {
                    if (boundary_type.at(LocationType::Right) == PDEBoundaryType::Adjacented)
                        write_csv(*field, filename + "_" + domain->name);
                    else
                        write_csv(*field,
                                  *buffers.at(LocationType::Right),
                                  filename + "_" + domain->name,
                                  VariablePositionType::XFace);
                }
                else if (var.position_type == VariablePositionType::YFace)
                {
                    if (boundary_type.at(LocationType::Back) == PDEBoundaryType::Adjacented)
                        write_csv(*field, filename + "_" + domain->name);
                    else
                        write_csv(*field,
                                  *buffers.at(LocationType::Back),
                                  filename + "_" + domain->name,
                                  VariablePositionType::YFace);
                }
                else if (var.position_type == VariablePositionType::ZFace)
                {
                    if (boundary_type.at(LocationType::Up) == PDEBoundaryType::Adjacented)
                        write_csv(*field, filename + "_" + domain->name);
                    else
                        write_csv(*field,
                                  *buffers.at(LocationType::Up),
                                  filename + "_" + domain->name,
                                  VariablePositionType::ZFace);
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
} // namespace IO