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

// buffer(dest, j) = f(src, j)
void copy_src_x_to_buffer_x(field2& buffer, field2& f, int src, int dest);

// buffer(i, dest) = f(i, src);
void copy_src_y_to_buffer_y(field2& buffer, field2& f, int src, int dest);

// buffer(j, dest) = f(src, j);
void copy_src_x_to_buffer_y(field2& buffer, field2& f, int src, int dest);

// buffer(dest, i) = f(i, src);
void copy_src_y_to_buffer_x(field2& buffer, field2& f, int src, int dest);

void mirror_x_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default);
void mirror_y_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default);
// Swap underlying data pointers of two same-typed fields (no shape/metadata change)
void swap_field_data(field2& a, field2& b);

double get_u_with_boundary(int           i,
                           int           j,
                           int           nx,
                           int           ny,
                           const field2& u,
                           double*       u_left_buffer,
                           double*       u_right_buffer,
                           double*       u_down_buffer,
                           double*       u_up_buffer,
                           double        right_down_corner_value);

double get_v_with_boundary(int           i,
                           int           j,
                           int           nx,
                           int           ny,
                           const field2& v,
                           double*       v_left_buffer,
                           double*       v_right_buffer,
                           double*       v_down_buffer,
                           double*       v_up_buffer,
                           double        left_up_corner_value);