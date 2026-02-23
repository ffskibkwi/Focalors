#include "stat.h"

#include <cmath>

double calc_rms(Variable3D& var)
{
    double sum = 0.0;
    for (auto kv : var.field_map)
        sum += kv.second->squared_sum();

    unsigned long long mesh_num = 0;
    for (auto kv : var.field_map)
        mesh_num += kv.second->get_size_n();

    return std::sqrt(sum / mesh_num);
}