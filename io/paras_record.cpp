#include "paras_record.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace IO
{
    ParasRecord::ParasRecord(const std::string& root_dir, SimInfoOutputFormat output_format)
    {
        std::stringstream ss;
        ss << root_dir;

        create_directory(ss.str());

        std::string extend;
        switch (output_format)
        {
            case SimInfoOutputFormat::Matlab:
                extend = ".m";
                break;
            case SimInfoOutputFormat::Python:
                extend = ".py";
                break;
            default:
                break;
        }

        ss << "/" << "paras_record" + extend;

        output_filename = ss.str();

        outfile.open(output_filename, std::ios::app);

        switch (output_format)
        {
            case SimInfoOutputFormat::Matlab:
                delimiter  = " = ";
                terminator = ";";
                break;
            case SimInfoOutputFormat::Python:
                delimiter  = " = ";
                terminator = "";
                break;
            default:
                break;
        }
    }

    template<>
    ParasRecord& ParasRecord::record(std::string key, std::string value)
    {
        outfile << key << delimiter << std::quoted(value) << terminator << std::endl;
        return *this;
    }

    void swap(ParasRecord& lhs, ParasRecord& rhs)
    {
        using std::swap;

        swap(lhs.output_filename, rhs.output_filename);
        swap(lhs.outfile, rhs.outfile);
        swap(lhs.delimiter, rhs.delimiter);
        swap(lhs.terminator, rhs.terminator);
    }

} // namespace IO