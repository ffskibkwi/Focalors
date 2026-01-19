#include "boundary_3d_utils.h"
#include <stdexcept>
#include <utility>

void assign_x(field3& f, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = val(j, k);
            }
        }
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = val_default;
            }
        }
    }
}

void assign_y(field3& f, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = val(i, k);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = val_default;
            }
        }
    }
}

void assign_z(field3& f, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = val(i, j);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = val_default;
            }
        }
    }
}

void copy_x(field3& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            f(dest, j, k) = f(src, j, k);
        }
    }
}

void copy_y(field3& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            f(i, dest, k) = f(i, src, k);
        }
    }
}

void copy_z(field3& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            f(i, j, dest) = f(i, j, src);
        }
    }
}

void mirror_x(field3& f, int src, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = 2.0 * val(j, k) - f(src, j, k);
            }
        }
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = 2.0 * val_default - f(src, j, k);
            }
        }
    }
}

void mirror_y(field3& f, int src, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = 2.0 * val(i, k) - f(i, src, k);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = 2.0 * val_default - f(i, src, k);
            }
        }
    }
}

void mirror_z(field3& f, int src, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = 2.0 * val(i, j) - f(i, j, src);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = 2.0 * val_default - f(i, j, src);
            }
        }
    }
}

void assign_val_to_buffer(field2& buffer, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < val.get_nx(); i++)
        {
            for (int j = 0; j < val.get_ny(); j++)
            {
                buffer(i, j) = val(i, j);
            }
        }
    }
    else
    {
        for (int i = 0; i < buffer.get_nx(); i++)
        {
            for (int j = 0; j < buffer.get_ny(); j++)
            {
                buffer(i, j) = val_default;
            }
        }
    }
}

void copy_x_to_buffer(field2& buffer, field3& f, int src)
{
    for (int j = 0; j < f.get_ny(); j++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            buffer(j, k) = f(src, j, k);
        }
    }
}

void copy_y_to_buffer(field2& buffer, field3& f, int src)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            buffer(i, k) = f(i, src, k);
        }
    }
}

void copy_z_to_buffer(field2& buffer, field3& f, int src)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            buffer(i, j) = f(i, j, src);
        }
    }
}

void copy_x_to_buffer(double* buffer, field3& f, int src_y, int src_z)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        buffer[i] = f(i, src_y, src_z);
    }
}

void copy_y_to_buffer(double* buffer, field3& f, int src_x, int src_z)
{
    for (int j = 0; j < f.get_ny(); j++)
    {
        buffer[j] = f(src_x, j, src_z);
    }
}

void copy_z_to_buffer(double* buffer, field3& f, int src_x, int src_y)
{
    for (int k = 0; k < f.get_nz(); k++)
    {
        buffer[k] = f(src_x, src_y, k);
    }
}

void mirror_x_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(j, k) = 2.0 * val(j, k) - f(src, j, k);
            }
        }
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(j, k) = 2.0 * val_default - f(src, j, k);
            }
        }
    }
}

void mirror_y_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(i, k) = 2.0 * val(i, k) - f(i, src, k);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(i, k) = 2.0 * val_default - f(i, src, k);
            }
        }
    }
}

void mirror_z_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                buffer(i, j) = 2.0 * val(i, j) - f(i, j, src);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                buffer(i, j) = 2.0 * val_default - f(i, j, src);
            }
        }
    }
}

void swap_field_data(field3& a, field3& b)
{
    if (a.get_size_n() != b.get_size_n())
        throw std::runtime_error("swap_field_data(field3): size mismatch");
    std::swap(a.value, b.value);
}
