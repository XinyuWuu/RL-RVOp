#include <cmath>
#include <iostream>
#include "typesdef.hpp"
#include "funcsdef.hpp"

// inline double DOT(double px1, double py1, double px2, double py2)
// {
//     return px1 * px2 + py1 * py2;
// }
// inline double CROSS(double px1, double py1, double px2, double py2)
// {
//     return px1 * py2 - px2 * py1;
// }
// inline double NORM(double px, double py)
// {
//     return sqrt(px * px + py * py);
// }
// inline double NORM2(double px, double py)
// {
//     return px * px + py * py;
// }

std::ostream &operator<<(std::ostream &output, point_t const &values)
{
    output << "[";
    for (double const &value : values)
    {
        output << value << " ";
    }
    output << "]";
    return output;
}

std::ostream &operator<<(std::ostream &output, line_t const &values)
{
    output << "[";
    for (double const &value : values)
    {
        output << value << " ";
    }
    output << "]";
    return output;
}

std::ostream &operator<<(std::ostream &output, arc_t const &values)
{
    output << "[";
    for (double const &value : values)
    {
        output << value << " ";
    }
    output << "]";
    return output;
}

std::ostream &operator<<(std::ostream &output, const std::vector<point_t> &values)
{
    for (size_t i = 0; i < values.size(); i++)
    {
        output << values[i];
    }
    return output;
}

std::ostream &operator<<(std::ostream &output, const std::vector<line_t> &values)
{
    for (size_t i = 0; i < values.size(); i++)
    {
        output << values[i];
    }
    return output;
}

std::ostream &operator<<(std::ostream &output, const std::vector<arc_t> &values)
{
    for (size_t i = 0; i < values.size(); i++)
    {
        output << values[i];
    }
    return output;
}

template <typename T>
std::ostream &operator<<(std::ostream &output, const std::vector<T> &values)
{
    for (size_t i = 0; i < values.size(); i++)
    {
        output << values[i];
    }
    return output;
}
