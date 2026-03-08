#pragma once

#include "base/pch.h"

#include "particles_coordinate_3d.h"

class PCoordMap3D
{
    ~PCoordMap3D();

    void add_sphere(int n, double r, double cx, double cy, double cz);

    void generate_map(Geometry3D* geo);

    std::unordered_map<Domain3DUniform*, PCoord3D*> get_map() { return coord_map; }

    double get_h() { return h; }

private:
    std::vector<PCoord3D*> collections;

    std::unordered_map<Domain3DUniform*, PCoord3D*> coord_map;

    double h;
}