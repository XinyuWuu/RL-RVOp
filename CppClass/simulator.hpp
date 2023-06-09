#include "pybind11_common_header.hpp"
#include "typesdef.hpp"
#include "funcsdef.hpp"
#include "mujoco/include/mujoco/mujoco.h"
#include "glfw/include/GLFW/glfw3.h"

namespace SIM
{
    class Simulator
    {
    private:
        /* data */
        mjModel *m;
        mjData *d;
        bool isRender;
        mjvScene *scn;
        mjvCamera *cam;
        mjvOption *opt;
        mjrContext *con;
        uint8_t *rgb;
        float *depth;
        int W, H;
        mjrRect viewport;
        GLFWwindow *window;

    public:
        Simulator(bool isRender, int W, int H, const char *modelfile);
        bool InitGLFW(int W, int H);
        bool CloseGLFW();
        bool InitMujoco(const char *modelfile);
        bool InitRender(int W, int H);
        void step();
        bool render();
        py::memoryview get_rgb();
        ~Simulator();
    };

} // namespace SIM
