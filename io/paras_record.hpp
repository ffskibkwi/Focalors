#pragma once

#include "common.h"

#include "non_copyable.h"
#include <fstream>
#include <unordered_map>

namespace IO
{
    enum SimInfoOutputFormat : uint8_t
    {
        Matlab,
        Python,
        Default
    };

    class ParasRecord : public NonCopyable
    {
    public:
        ParasRecord() {}

        ParasRecord(const std::string& root_dir, SimInfoOutputFormat output_format);

        ~ParasRecord() { outfile.close(); }

        ParasRecord(ParasRecord&& rhs) noexcept { swap(*this, rhs); }

        ParasRecord& operator=(ParasRecord&& rhs) noexcept
        {
            if (this != &rhs)
            {
                swap(*this, rhs);
            }

            return *this;
        }

        template<typename ValueType>
        ParasRecord& record(std::string key, ValueType value)
        {
            outfile << key << delimiter << value << terminator << std::endl;
            return *this;
        }

        friend void swap(ParasRecord& lhs, ParasRecord& rhs);

    private:
        std::string   output_filename;
        std::ofstream outfile;

        std::string delimiter  = ": ";
        std::string terminator = "";
    };

    template<>
    ParasRecord& ParasRecord::record(std::string key, std::string value);
} // namespace IO