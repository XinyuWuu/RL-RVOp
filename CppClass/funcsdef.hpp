#ifndef __myfuncsdef__
#define __myfuncsdef__
#include "typesdef.hpp"

#define DOT(pointx1, pointy1, pointx2, pointy2) ((pointx1) * (pointx2) + (pointy1) * (pointy2))
#define CROSS(pointx1, pointy1, pointx2, pointy2) ((pointx1) * (pointy2) - (pointx2) * (pointy1))
#define NORM(pointx, pointy) sqrt((pointx) * (pointx) + (pointy) * (pointy))
#define NORM2(pointx, pointy) ((pointx) * (pointx) + (pointy) * (pointy))

std::ostream &operator<<(std::ostream &output, line_t const &values);
std::ostream &operator<<(std::ostream &output, arc_t const &values);
template <typename T>
std::ostream &operator<<(std::ostream &output, std::vector<T> const &values);
#endif
