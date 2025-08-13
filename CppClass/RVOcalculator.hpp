#ifndef __RVO__
#define __RVO__
#include "pybind11_common_header.hpp"
#include "typesdef.hpp"
namespace RVO
{
    class RVOcalculator
    {
    private:
        /* data */
    public:
        double dmax;
        double robot_r;
        contours_t contours;
        points_t target;
        RVOcalculator();
        RVOcalculator(double dmax, double robot_r);
        RVOcalculator(double dmax, double robot_r, contours_t contours, points_t target);
        bool inLine(const line_t &line, const point_t &x0);
        bool inArc(const arc_t &arc, const point_t &x0);
        void LineP(const line_t &line, const point_t &xr, points_t &points);
        void ArcP(const arc_t &arc, const point_t &xr, points_t &points);
        rvop_t RVOplus(lines_t lines, arcs_t arcs, point_t xr, point_t vr, point_t vo);
        void RVOplus(const lines_t &lines, const arcs_t &arcs, const point_t &xr, const point_t &vr, const point_t &vo, double *obs, bool avevel);
        observations_t get_obs(posvels_t posvel, bool avevel);
        ~RVOcalculator();
    };
} // namespace RVO

#endif
