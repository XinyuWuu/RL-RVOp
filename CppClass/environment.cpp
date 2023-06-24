#include "environment.hpp"
#include <iostream>
namespace ENV
{
    Environment::Environment()
    {
    }
    bool Environment::stepVL(const points_t vs, int N, int n)
    {
        ctrlP->v2ctrlbatchL(this->simP->posvels, vs, this->ctrl);
        for (int i = 0; i < N; i += n)
        {
            ctrlP->v2ctrlbatchG(this->simP->posvels, vs.size(), this->ctrl);
            this->simP->step(ctrl, n);
        }
        return true;
    }
    bool Environment::stepVG(const points_t vs, int N, int n)
    {
        for (int i = 0; i < N; i += n)
        {
            ctrlP->v2ctrlbatchG(this->simP->posvels, vs, this->ctrl);
            this->simP->step(ctrl, n);
        }
        return true;
    }
    bool Environment::setCtrl(double vmax, double tau, double wheel_r, double wheel_d, double gain)
    {
        this->vmax = vmax;
        this->tau = tau;
        this->wheel_r = wheel_r;
        this->wheel_d = wheel_d;
        this->gain = gain;
        this->ctrlP = std::make_shared<CTRL::CtrlConverter>(vmax, tau, wheel_r, wheel_d, gain);
        return true;
    }
    bool Environment::setSim(const char *modelfile, int Nrobot, bool isRender, int W, int H)
    {
        this->isRender = isRender;
        this->Nrobot = Nrobot;
        this->H = H;
        this->W = W;
        this->simP = std::make_shared<SIM::Simulator>(modelfile, Nrobot, isRender, W, H);
        return true;
    }
    py::memoryview Environment::get_rgb()
    {
        return this->simP->get_rgb();
    }
    py::memoryview Environment::get_posvels()
    {
        return this->simP->get_posvels();
    }
    bool Environment::CloseGLFW()
    {
        return this->simP->CloseGLFW();
    }
    bool Environment::render()
    {
        return this->simP->render();
    }
    Environment::~Environment()
    {
    }
} // namespace ENV
