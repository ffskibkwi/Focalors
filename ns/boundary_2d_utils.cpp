#include "boundary_2d_utils.h"

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