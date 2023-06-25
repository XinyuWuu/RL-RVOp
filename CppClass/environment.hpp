#ifndef __ENV__
#define __ENV__
#include "pybind11_common_header.hpp"
#include "simulator.hpp"
#include "ctrlConverter.hpp"
#include "myreward.hpp"
#include "RVOcalculator.hpp"
#include <memory>
#include <vector>
namespace ENV
{

    class Environment
    {
    private:
        std::shared_ptr<SIM::Simulator> simP;
        std::shared_ptr<CTRL::CtrlConverter> ctrlP;
        std::shared_ptr<RWD::Reward> rwdP;
        std::shared_ptr<RVO::RVOcalculator> rvopP;
        double ctrl[200];
        double reward[200], reward_mix[200];
        int death[200];
        points_t target;
        contours_t contour;
        observations_t observations;
        int Nrobot;
        bool isRender;
        int W, H;
        double vmax, tau, wheel_r, wheel_d, gain;
        double robot_r, rmax, tolerance, a, b, c, d, e, f, g, eta, h, mu, rreach, dreach, dmax, w, tb;
        bool remix;
        int rm_middle;

    public:
        Environment();
        ~Environment();
        bool setSim(const char *modelfile, int Nrobot, points_t target, contours_t contour, bool isRender, int W, int H);
        double setCtrl(double vmax, double tau, double wheel_r, double wheel_d, double gain);
        bool setRwd(double robot_r, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu, double rreach, double dreach, bool remix, int rm_middle, double dmax, double w, double tb);
        bool setRvop(double dmax, double robot_r);
        bool stepVL(const points_t vs, int N, int n);
        bool stepVG(const points_t vs, int N, int n);
        bool cal_obs(double *posvels, bool avevel);
        py::memoryview get_rgb();
        py::memoryview get_posvels();
        py::memoryview get_r();
        py::memoryview get_rm();
        py::memoryview get_d();
        bool CloseGLFW();
        bool render();
    };
} // namespace ENV
#endif //__ENV__
