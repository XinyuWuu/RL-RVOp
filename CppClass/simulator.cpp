#include "simulator.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

namespace SIM
{
    bool Simulator::InitGLFW(int W, int H)
    {
        if (!glfwInit())
        {
            return false;
        };
        glfwWindowHint(GLFW_VISIBLE, 0);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
        this->W = W;
        this->H = H;
        this->viewport = mjrRect{0, 0, this->W, this->H};
        window = glfwCreateWindow(this->W, this->H, "Invisible window", NULL, NULL);
        if (!window)
        {
            return false;
        }
        glfwMakeContextCurrent(window);
        return true;
    }
    bool Simulator::InitMujoco(const char *modelfile)
    {
        char error[1000] = "Could not load xml model";
        mjVFS vfs;
        mj_defaultVFS(&vfs);
        mj_makeEmptyFileVFS(&vfs, "model.xml", strlen(modelfile));
        std::cout << strlen(modelfile) << " point make file" << std::endl;
        std::memcpy(vfs.filedata[vfs.nfile - 1],
                    modelfile, strlen(modelfile));
        std::cout << vfs.nfile << " point copy file" << std::endl;
        this->m = mj_loadXML("model.xml", &vfs, error, 1000);
        this->d = mj_makeData(this->m);
        mj_forward(this->m, this->d);
        return true;
    }
    bool Simulator::InitRender(int W, int H)
    {
        this->cam = new mjvCamera;
        this->opt = new mjvOption;
        this->scn = new mjvScene;
        this->con = new mjrContext;
        mjv_defaultCamera(this->cam);
        mjv_defaultOption(this->opt);
        mjv_defaultScene(this->scn);
        mjr_defaultContext(this->con);

        mjv_makeScene(this->m, this->scn, 10000);
        mjr_makeContext(this->m, this->con, mjFONTSCALE_150);
        mjr_setBuffer(mjFB_OFFSCREEN, this->con);

        this->cam->type = mjCAMERA_FIXED;
        this->cam->fixedcamid = 0;

        // mjrRect viewport = mjr_maxViewport(&con);
        this->W = W;
        this->H = H;
        this->viewport = mjrRect{0, 0, this->W, this->H};

        // allocate rgb and depth buffers
        this->rgb = (uint8_t *)std::malloc(3 * this->W * this->H);
        this->depth = (float *)std::malloc(sizeof(float) * this->W * this->H);
        if (!rgb || !depth)
        {
            return false;
        }
        return true;
    }
    Simulator::Simulator(bool isRender, int W, int H, const char *modelfile)
    {
        this->isRender = isRender;
        this->W = W;
        this->H = H;
        if (this->isRender)
        {
            InitGLFW(this->W, this->H);
        }
        std::cout << "point 1" << std::endl;
        InitMujoco(modelfile);
        std::cout << "point 2" << std::endl;
        if (this->isRender)
        {
            InitRender(this->W, this->H);
            std::cout << "point 3" << std::endl;
        }
    }

    Simulator::~Simulator()
    {
        std::cout << "point destruction" << std::endl;
        mj_deleteModel(this->m);
        mj_deleteData(this->d);
        if (this->isRender)
        {
            // mjr_freeContext(this->con); // moved to CloseGLFW
            mjv_freeScene(this->scn);
            delete this->scn;
            delete this->cam;
            delete this->opt;
            delete this->con;
            std::free(this->rgb);
            std::free(this->depth);
            // glfwDestroyWindow(this->window); // moved to CloseGLFW
            // glfwTerminate(); // moved to CloseGLFW
        }
        std::cout << "point end" << std::endl;
    }
    bool Simulator::CloseGLFW()
    {
        mjr_freeContext(this->con);
        glfwDestroyWindow(this->window);
        glfwTerminate();
        return true;
    }
    void Simulator::step()
    {
        this->d->ctrl[0] = 100;
        this->d->ctrl[1] = 100;
        mj_step(this->m, this->d);
    }

    bool Simulator::render()
    {
        if (this->isRender)
        {
            mjv_updateScene(this->m, this->d, this->opt, NULL, this->cam, mjCAT_ALL, this->scn);
            mjr_render(this->viewport, this->scn, this->con);
            mjr_readPixels(this->rgb, this->depth, this->viewport, this->con);
            return true;
        }
        return false;
    }

    py::memoryview Simulator::get_rgb()
    {

        return py::memoryview::from_buffer(
            this->rgb,
            {this->H,
             this->W,
             3},
            {sizeof(uint8_t) * this->W * 3,
             sizeof(uint8_t) * 3,
             sizeof(uint8_t)});

        // return py::memoryview::from_memory(
        //     this->rgb,                              // buffer pointer
        //     sizeof(uint8_t) * this->H * this->W * 3 // buffer size
        // );
    }
} // namespace SIM
