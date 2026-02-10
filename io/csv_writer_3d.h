#pragma once

#include "base/pch.h"

namespace IO
{
    bool write_csv(field3& field, const std::string& filename);

    bool read_csv(field3& field, const std::string& filename);

    bool write_csv(const Variable3D& var, const std::string& filename);
} // namespace IO