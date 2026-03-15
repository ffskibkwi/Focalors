#pragma once

#include "base/domain/domain3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field2.h"
#include "base/field/field3.h"

#include <unordered_map>
#include <vector>

void assign_x(field3& f, int dest, field2* val_ptr, double val_default);
void assign_y(field3& f, int dest, field2* val_ptr, double val_default);
void assign_z(field3& f, int dest, field2* val_ptr, double val_default);

void copy_x(field3& f, int src, int dest);
void copy_y(field3& f, int src, int dest);
void copy_z(field3& f, int src, int dest);

void mirror_x(field3& f, int src, int dest, field2* val_ptr, double val_default);
void mirror_y(field3& f, int src, int dest, field2* val_ptr, double val_default);
void mirror_z(field3& f, int src, int dest, field2* val_ptr, double val_default);

void assign_val_to_buffer(field2& buffer, field2* val_ptr, double val_default);

void copy_x_to_buffer(field2& buffer, field3& f, int src);
void copy_y_to_buffer(field2& buffer, field3& f, int src);
void copy_z_to_buffer(field2& buffer, field3& f, int src);

void copy_x_to_buffer(double* buffer, field3& f, int src_y, int src_z);
void copy_y_to_buffer(double* buffer, field3& f, int src_x, int src_z);
void copy_z_to_buffer(double* buffer, field3& f, int src_x, int src_y);

void mirror_x_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default);
void mirror_y_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default);
void mirror_z_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default);

void swap_field_data(field3& a, field3& b);

bool isAllNeumannBoundary(const Variable3D& var);

double normalizeRhsForNeumannBc(const Variable3D&                                    var,
                                const std::vector<Domain3DUniform*>&                 domains,
                                const std::unordered_map<Domain3DUniform*, field3*>& fieldMap);
