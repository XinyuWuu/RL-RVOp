#ifndef __mytypesdef__
#define __mytypesdef__

#include "pybind11_common_header.hpp"
#include <array>
#include <vector>
namespace py = pybind11;

// typedef py::array_t<double> point_t;
// typedef py::array_t<double> line_t;
// typedef py::array_t<double> arc_t;
// typedef py::array_t<double> rvop_t;

typedef std::array<double, 2> point_t;
typedef std::array<double, 6> line_t;
typedef std::array<double, 9> arc_t;
typedef std::array<double, 8> rvop_t;
typedef std::array<double, 3> vector3d;
typedef std::array<double, 6> posvel_t;
// typedef double point_t[2];
// typedef double line_t[6];
// typedef double arc_t[9];
// typedef double rvop_t[8];

typedef std::vector<point_t> points_t;
typedef std::vector<line_t> lines_t;
typedef std::vector<arc_t> arcs_t;
typedef std::vector<rvop_t> rvops_t;
typedef std::vector<posvel_t> posvels_t;
// typedef std::list<point_t> points_t;
// typedef std::list<line_t> lines_t;
// typedef std::list<arc_t> arcs_t;
// typedef std::list<rvop_t> rvops_t;

// struct obs_t
// {
//     rvop_t rvop;
//     vector3d pos;
//     vector3d vel;
//     point_t target;
// };
typedef std::array<double, 16> obs_t;
typedef std::vector<obs_t> observation_t;
typedef std::vector<observation_t> observations_t;
// struct contour_t
// {
//     lines_t lines;
//     arcs_t arcs;
// };

typedef std::array<double, 10> obs_sur_t;
typedef std::array<double, 4> observation_self_t;
typedef std::vector<obs_sur_t> observation_sur_t;

typedef std::vector<observation_self_t> observations_self_t;
typedef std::vector<observation_sur_t> observations_sur_t;

typedef std::pair<lines_t, arcs_t> contour_t;
typedef std::vector<contour_t> contours_t;

typedef std::pair<observations_self_t, observations_sur_t> NNinput_t;
typedef std::tuple<observations_t, std::vector<double>, NNinput_t> OBSreturn;

#endif
