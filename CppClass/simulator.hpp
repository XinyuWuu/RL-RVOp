#ifndef __SIM__
#define __SIM__
#include "pybind11_common_header.hpp"
#include "typesdef.hpp"
#include "funcsdef.hpp"
#include "mujoco/include/mujoco/mujoco.h"
#include "glfw/include/GLFW/glfw3.h"

#include <vector>
#include <string>

namespace SIM
{
    class Simulator
    {
    private:
        /* data */
        int Nrobot;
        int *qveladr, *qposadr;
        mjModel *m;
        mjData *d;
        bool isRender;
        mjvScene *scn;
        mjvCamera *cam;
        mjvOption *opt;
        mjrContext *con;
        int W, H;
        mjrRect viewport;
        GLFWwindow *window;

    public:
        double *posvels;
        uint8_t *rgb;
        float *depth;
        Simulator();
        Simulator(const char *modelfile, int Nrobot, bool isRender, int W, int H);
        bool InitGLFW(int W, int H);
        bool CloseGLFW();
        bool InitMujoco(const char *modelfile, int Nrobot);
        bool InitRender(int W, int H);
        void cal_posvels();
        void step(std::vector<double> ctrl, int N);
        void step(const double *ctrl, int N);
        bool render();
        py::memoryview get_rgb();
        py::memoryview get_posvels();
        ~Simulator();
    };

} // namespace SIM
#endif //__SIM__
