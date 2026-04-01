#pragma once

#include "base/pch.h"

void write_savepoint(const Variable3D& var, const std::string& filename);
void read_savepoint(const Variable3D& var, const std::string& filename);