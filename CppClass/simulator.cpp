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
    bool Simulator::InitMujoco(const char *modelfile, int Nrobot)
    {
        this->Nrobot = Nrobot;
        posvels = (double *)std::malloc(sizeof(double) * this->Nrobot * 6);
        qposadr = (int *)std::malloc(sizeof(int) * this->Nrobot);
        qveladr = (int *)std::malloc(sizeof(int) * this->Nrobot);
        char error[1000] = "Could not load xml model";
        mjVFS vfs;
        mj_defaultVFS(&vfs);
        mj_makeEmptyFileVFS(&vfs, "model.xml", strlen(modelfile));
        std::memcpy(vfs.filedata[vfs.nfile - 1],
                    modelfile, strlen(modelfile));
        this->m = mj_loadXML("model.xml", &vfs, error, 1000);
        this->d = mj_makeData(this->m);
        mj_forward(this->m, this->d);

        std::string tem_str = std::string("robot_");
        for (int i = 0; i < Nrobot; i++)
        {
            qposadr[i] = this->m->jnt_qposadr[mj_name2id(this->m, mjOBJ_JOINT, (tem_str + std::to_string(i)).c_str())];
            qveladr[i] = this->m->jnt_dofadr[mj_name2id(this->m, mjOBJ_JOINT, (tem_str + std::to_string(i)).c_str())];
        }

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
    Simulator::Simulator() {}
    Simulator::Simulator(const char *modelfile, int Nrobot, bool isRender, int W, int H)
    {
        this->isRender = isRender;
        this->W = W;
        this->H = H;
        if (this->isRender)
        {
            InitGLFW(this->W, this->H);
        }
        InitMujoco(modelfile, Nrobot);
        cal_posvels();
        if (this->isRender)
        {
            InitRender(this->W, this->H);
        }
    }

    Simulator::~Simulator()
    {
        mj_deleteModel(this->m);
        mj_deleteData(this->d);
        std::free(this->qposadr);
        std::free(this->qveladr);
        std::free(this->posvels);
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
    }
    bool Simulator::CloseGLFW()
    {
        mjr_freeContext(this->con);
        glfwDestroyWindow(this->window);
        glfwTerminate();
        return true;
    }
    void Simulator::cal_posvels()
    {
        for (int i = 0; i < this->Nrobot; i++)
        {
            this->posvels[i * 6] = this->d->qpos[qposadr[i]];
            this->posvels[i * 6 + 1] = this->d->qpos[qposadr[i] + 1];
            this->posvels[i * 6 + 2] = atan2(2 * this->d->qpos[qposadr[i] + 3] * this->d->qpos[qposadr[i] + 6],
                                             1 - 2 * this->d->qpos[qposadr[i] + 6] * this->d->qpos[qposadr[i] + 6]);
            this->posvels[i * 6 + 3] = this->d->qvel[qveladr[i]];
            this->posvels[i * 6 + 4] = this->d->qvel[qveladr[i] + 1];
            this->posvels[i * 6 + 5] = this->d->qvel[qveladr[i] + 5];
        }
    }
    void Simulator::step(std::vector<double> ctrl, int N)
    {
        // for (int i = 0; i < 2 * this->Nrobot; i++)
        // {
        //     this->d->ctrl[i] = ctrl[i];
        // }
        memcpy(this->d->ctrl, ctrl.data(), ctrl.size() * sizeof(double));
        for (int i = 0; i < N; i++)
        {
            mj_step(this->m, this->d);
        }
        cal_posvels();
    }
    void Simulator::step(const double *ctrl, int N)
    {
        memcpy(this->d->ctrl, ctrl, 2 * Nrobot * sizeof(double));
        for (int i = 0; i < N; i++)
        {
            mj_step(this->m, this->d);
        }
        cal_posvels();
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
    double Simulator::get_time()
    {
        return this->d->time;
    }
    py::memoryview Simulator::get_rgb()
    {

        return py::memoryview::from_buffer(
            this->rgb,
            {this->H,
             this->W,
             3},
            {sizeof(rgb[0]) * this->W * 3,
             sizeof(rgb[0]) * 3,
             sizeof(rgb[0])});

        // return py::memoryview::from_memory(
        //     this->rgb,                              // buffer pointer
        //     sizeof(uint8_t) * this->H * this->W * 3 // buffer size
        // );
    }
    py::memoryview Simulator::get_posvels()
    {

        return py::memoryview::from_buffer(
            this->posvels,
            {this->Nrobot, 6},
            {sizeof(posvels[0]) * 6,
             sizeof(posvels[0])});

        // return py::memoryview::from_memory(
        //     this->posvels,                    // buffer pointer
        //     sizeof(double) * this->Nrobot * 6 // buffer size
        // );
    }
} // namespace SIM
