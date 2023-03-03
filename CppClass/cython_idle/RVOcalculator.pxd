# distutils: include_dirs = ./
# distutils: language = c++
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.list cimport list

cdef extern from "typesdef.hpp":
    ctypedef double point_t[2]
    ctypedef double line_t[6]
    ctypedef double arc_t[9]
    ctypedef double rvop_t[8]

    # ctypedef vector[point_t] points_t
    # ctypedef vector[line_t] lines_t
    # ctypedef vector[arc_t] arcs_t
    ctypedef list[point_t] points_t
    ctypedef list[line_t] lines_t
    ctypedef list[arc_t] arcs_t

cdef extern from "RVOcalculator.hpp" namespace "RVO":
    cdef cppclass RVOcalculator:
            RVOcalculator() except +
            RVOcalculator(double dmax) except +
            double dmax
            bool inLine(line_t line, point_t x0)
            bool inArc(arc_t arc, point_t x0)
            points_t LineP(line_t line, point_t xr)
            points_t ArcP(arc_t arc, point_t xr)
