# distutils: language = c++
# distutils: sources = RVOcalculator.cpp
from RVOcalculator cimport RVOcalculator
from RVOcalculator cimport line_t, point_t

cdef class PyRVOcalculator:
    cdef RVOcalculator cRVOcalculator

    def __init__(self, double dmax):
        self.cRVOcalculator = RVOcalculator(dmax)
    def inLine(self, line_t line, point_t x0):
        return self.cRVOcalculator.inLine(line,x0)
    # def inArc(self,arc_t arc, point_t x0):
    #     return self.cRVOcalculator.inArc(arc,x0)
    # def LineP(self,line_t line, point_t xr):
    #     return self.cRVOcalculator.LineP(line,xr)
