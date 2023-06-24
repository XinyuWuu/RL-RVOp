#ifndef __ENV__
#define __ENV__
#include "pybind11_common_header.hpp"
#include "simulator.hpp"
#include "ctrlConverter.hpp"
#include <memory>
#include <vector>
namespace ENV
{

    class Environment
    {
    private:
        std::shared_ptr<SIM::Simulator> simP;
        std::shared_ptr<CTRL::CtrlConverter> ctrlP;
        double ctrl[200];
        double *posvels;
        int Nrobot;
        bool isRender;
        int W, H;
        double vmax, tau, wheel_r, wheel_d, gain;

    public:
        Environment();
        ~Environment();
        bool setSim(const char *modelfile, int Nrobot, bool isRender, int W, int H);
        bool setCtrl(double vmax, double tau, double wheel_r, double wheel_d, double gain);
        bool stepVL(const points_t vs, int N, int n);
        bool stepVG(const points_t vs, int N, int n);
        py::memoryview get_rgb();
        py::memoryview get_posvels();
        bool CloseGLFW();
        bool render();
    };
} // namespace ENV
#endif //__ENV__
