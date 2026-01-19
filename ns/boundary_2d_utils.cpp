#include "boundary_2d_utils.h"
#include <stdexcept>
#include <utility>

void assign_x(field2& f, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = val_ptr[j];
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = val_default;
    }
}

void assign_y(field2& f, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = val_ptr[i];
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = val_default;
    }
}

void copy_x(field2& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
        f(dest, j) = f(src, j);
}

void copy_y(field2& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
        f(i, dest) = f(i, src);
}

void mirror_x(field2& f, int src, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = 2.0 * val_ptr[j] - f(src, j);
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = 2.0 * val_default - f(src, j);
    }
}

void mirror_y(field2& f, int src, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = 2.0 * val_ptr[i] - f(i, src);
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = 2.0 * val_default - f(i, src);
    }
}

void assign_val_to_buffer(double* buffer, int length, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < length; i++)
            buffer[i] = val_ptr[i];
    }
    else
    {
        for (int i = 0; i < length; i++)
            buffer[i] = val_default;
    }
}

void copy_x_to_buffer(double* buffer, field2& f, int src)
{
    for (int j = 0; j < f.get_ny(); j++)
        buffer[j] = f(src, j);
}

void copy_y_to_buffer(double* buffer, field2& f, int src)
{
    for (int i = 0; i < f.get_nx(); i++)
        buffer[i] = f(i, src);
}

void copy_src_x_to_buffer_x(field2& buffer, field2& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
        buffer(dest, j) = f(src, j);
}

void copy_src_y_to_buffer_y(field2& buffer, field2& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
        buffer(i, dest) = f(i, src);
}

void copy_src_x_to_buffer_y(field2& buffer, field2& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
        buffer(j, dest) = f(src, j);
}

void copy_src_y_to_buffer_x(field2& buffer, field2& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
        buffer(dest, i) = f(i, src);
}

void mirror_x_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int j = 0; j < f.get_ny(); j++)
            buffer[j] = 2.0 * val_ptr[j] - f(src, j);
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
            buffer[j] = 2.0 * val_default - f(src, j);
    }
}

void mirror_y_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < f.get_nx(); i++)
            buffer[i] = 2.0 * val_ptr[i] - f(i, src);
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
            buffer[i] = 2.0 * val_default - f(i, src);
    }
}

// Swap underlying data pointers for two same-typed fields.
// Requires identical storage size; only swaps the data pointer without touching shape/metadata.
void swap_field_data(field2& a, field2& b)
{
    if (a.get_size_n() != b.get_size_n())
        throw std::runtime_error("swap_field_data(field2): size mismatch");
    std::swap(a.value, b.value);
}

void swap_field_data(field3& a, field3& b)
{
    if (a.get_size_n() != b.get_size_n())
        throw std::runtime_error("swap_field_data(field3): size mismatch");
    std::swap(a.value, b.value);
}

double get_u_with_boundary(int           i,
                           int           j,
                           int           nx,
                           int           ny,
                           const field2& u,
                           double*       u_left_buffer,
                           double*       u_right_buffer,
                           double*       u_down_buffer,
                           double*       u_up_buffer,
                           double        right_down_corner_value)
{
    if (i >= 0 && i < nx && j >= 0 && j < ny)
        return u(i, j);
    if (j < 0)
        return (i >= nx) ? right_down_corner_value : u_down_buffer[i];
    if (j >= ny)
        return u_up_buffer[i];
    if (i >= nx)
        return u_right_buffer[j];
    // i < 0
    return u_left_buffer[j];
}

double get_v_with_boundary(int           i,
                           int           j,
                           int           nx,
                           int           ny,
                           const field2& v,
                           double*       v_left_buffer,
                           double*       v_right_buffer,
                           double*       v_down_buffer,
                           double*       v_up_buffer,
                           double        left_up_corner_value)
{
    if (i >= 0 && i < nx && j >= 0 && j < ny)
        return v(i, j);
    if (i < 0)
        return (j >= ny) ? left_up_corner_value : v_left_buffer[j];
    if (i >= nx)
        return v_right_buffer[j];
    if (j >= ny)
        return v_up_buffer[i];
    // j < 0
    return v_down_buffer[i];
}