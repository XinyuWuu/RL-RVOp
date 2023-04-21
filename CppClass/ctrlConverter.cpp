#include "ctrlConverter.hpp"
#include "funcsdef.hpp"
#include <iostream>
namespace CTRL
{
    point_t CtrlConverter::vw2ctrl(const double vl, const double w)
    {
        double p[2];
        this->vw2ctrl(vl, w, p);
        return point_t{p[0], p[1]};
    }
    void CtrlConverter::vw2ctrl(const double &vl, const double &w, double *ctrl)
    {
        double omega_ctrl = fmax(-this->rmax, fmin(w / 2 / this->wd / this->wr, this->rmax));
        double speed_ctrl = fmax(-this->rmax + abs(omega_ctrl), fmin(vl / this->wr, this->rmax - abs(omega_ctrl)));
        // ctrl[0] = fmax(-this->rmax, fmin((vl - w / 2 / this->wd) / this->wr, this->rmax)) * this->gain;
        // ctrl[1] = fmax(-this->rmax, fmin((vl + w / 2 / this->wd) / this->wr, this->rmax)) * this->gain;
        ctrl[0] = (speed_ctrl - omega_ctrl) * this->gain;
        ctrl[1] = (speed_ctrl + omega_ctrl) * this->gain;
    }
    point_t CtrlConverter::v2ctrl(const double ori, const point_t v)
    {
        double p[2];
        this->vw2ctrl(DOT(v[0], v[1], cos(ori), sin(ori)),
                      atan2(CROSS(cos(ori), sin(ori),
                                  v[0], v[1]),
                            DOT(v[0], v[1], cos(ori), sin(ori))) /
                          this->tau,
                      p);
        return point_t{p[0], p[1]};
    }
    void CtrlConverter::v2ctrl(const double &ori, const point_t &v, double *ctrl)
    {
        this->vw2ctrl(DOT(v[0], v[1], cos(ori), sin(ori)),
                      atan2(CROSS(cos(ori), sin(ori),
                                  v[0], v[1]),
                            DOT(v[0], v[1], cos(ori), sin(ori))) /
                          this->tau,
                      ctrl);
    }
    std::vector<double> CtrlConverter::v2ctrlbatchL(const posvels_t posvels, const points_t vs)
    {
        std::vector<double> ctrl(2 * posvels.size());
        points_t vsG(vs.size());
        for (size_t Nth = 0; Nth < posvels.size(); Nth++)
        {
            vsG[Nth][0] = cos(posvels[Nth][2]) * vs[Nth][0] - sin(posvels[Nth][2]) * vs[Nth][1];
            vsG[Nth][1] = sin(posvels[Nth][2]) * vs[Nth][0] + cos(posvels[Nth][2]) * vs[Nth][1];
        }
        for (size_t i = 0; i < posvels.size(); i++)
        {
            this->v2ctrl(posvels[i][2], vsG[i], ctrl.data() + i * 2);
        }
        return ctrl;
    }
    std::vector<double> CtrlConverter::v2ctrlbatchG(const posvels_t posvels, const points_t vs)
    {
        std::vector<double> ctrl(2 * posvels.size());
        for (size_t i = 0; i < posvels.size(); i++)
        {
            this->v2ctrl(posvels[i][2], vs[i], ctrl.data() + i * 2);
        }
        return ctrl;
    }
    CtrlConverter::CtrlConverter() {}
    CtrlConverter::CtrlConverter(double vmax, double tau, double wheel_r, double wheel_d, double gain)
    {
        this->vmax = vmax;
        this->tau = tau;
        this->wr = wheel_r;
        this->wd = wheel_d;
        this->gain = gain;
        this->rmax = this->vmax / this->wr;
        // std::cout << this->vmax << "," << this->tau << "," << this->wr << "," << this->wd << "," << this->rmax << "," << this->gain << std::endl;
    }
    double CtrlConverter::get_rmax()
    {
        return this->rmax;
    }
    CtrlConverter::~CtrlConverter() {}

} // namespace CTRL
