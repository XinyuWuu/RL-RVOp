#ifndef __ctrlconverter__
#define __ctrlconverter__
#include "pybind11_common_header.hpp"
#include "typesdef.hpp"

namespace CTRL
{
    class CtrlConverter
    {
    private:
        // gain = 7 convert omega to real ctrl, determined by mujoco module
        double vmax, rmax, tau, wr, wd, gain;

    public:
        CtrlConverter(/* args */);
        CtrlConverter(double vmax, double tau, double wheel_r, double wheel_d, double gain);
        point_t vw2ctrl(const double vl, const double w);
        void vw2ctrl(const double &vl, const double &w, double *ctrl);
        point_t v2ctrl(const double ori, const point_t v);
        void v2ctrl(const double &ori, const point_t &v, double *ctrl);
        std::vector<double> v2ctrlbatchL(const posvels_t posvels, const points_t vs);
        std::vector<double> v2ctrlbatchG(const posvels_t posvels, const points_t vs);
        double get_rmax();
        ~CtrlConverter();
    };
} // namespace CTRL
#endif
