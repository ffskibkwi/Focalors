#pragma once

#include "base/field/field2.h"
#include "base/field/field3.h"

void assign_x(field2& f, int dest, double* val_ptr, double val_default);
void assign_y(field2& f, int dest, double* val_ptr, double val_default);

void copy_x(field2& f, int src, int dest);
void copy_y(field2& f, int src, int dest);

void mirror_x(field2& f, int src, int dest, double* val_ptr, double val_default);
void mirror_y(field2& f, int src, int dest, double* val_ptr, double val_default);

void assign_val_to_buffer(double* buffer, int length, double* val_ptr, double val_default);

void copy_x_to_buffer(double* buffer, field2& f, int src);
void copy_y_to_buffer(double* buffer, field2& f, int src);

void mirror_x_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default);
void mirror_y_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default);
// Swap underlying data pointers of two same-typed fields (no shape/metadata change)
void swap_field_data(field2& a, field2& b);
void swap_field_data(field3& a, field3& b);