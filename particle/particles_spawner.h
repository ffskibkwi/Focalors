#pragma once

// spawn cylinder in uniform grid, return uniform particles space
void spawn_cylinder(double* X, double* Y, int n, double r, double cx, double cy, double& h);
// spawn sphere in uniform grid, return uniform particles space
void spawn_sphere(double* X, double* Y, double* Z, int n, double r, double cx, double cy, double cz, double& h);