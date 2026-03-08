#pragma once

#include "base/pch.h"

#include "particles_coordinate_2d.h"

class PCoordMap2D
{
    ~PCoordMap2D();

    void add_cylinder(int n, double r, double cx, double cy);

    void generate_map(Geometry2D* geo);

    std::unordered_map<Domain2DUniform*, PCoord2D*> get_map() { return coord_map; }

    double get_h() { return h; }

private:
    std::vector<PCoord2D*> collections;

    std::unordered_map<Domain2DUniform*, PCoord2D*> coord_map;

    double h;
}