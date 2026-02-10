#pragma once

#include "base/pch.h"

namespace IO
{
    bool write_csv(double** value, int nx, int ny, const std::string& filename);

    bool write_csv(field2& field, const std::string& filename);

    bool read_csv(field2& field, const std::string& filename);

    bool write_csv(const Variable2D& var, const std::string& filename);

    bool matlab_read_var(const Variable2D& var, const std::string& filename);
} // namespace IO
