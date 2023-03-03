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

std::ostream &operator<<(std::ostream &output, line_t const &values)
{
    output << "[";
    for (auto const &value : values)
    {
        output << value << " ";
    }
    output << "]\n";
    return output;
}

std::ostream &operator<<(std::ostream &output, arc_t const &values)
{
    output << "[";
    for (auto const &value : values)
    {
        output << value << " ";
    }
    output << "]\n";
    return output;
}

template <typename T>
std::ostream &operator<<(std::ostream &output, std::vector<T> const &values)
{
    for (auto const &value : values)
    {
        output << value;
    }
    return output;
}
