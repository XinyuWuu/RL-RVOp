#include "environment.hpp"
#include "funcsdef.hpp"
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
    bool Environment::setRwd(double robot_r, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu, double rreach, double dreach, bool remix, int rm_middle, double dmax, double w, double tb)
    {
        this->robot_r = robot_r;
        this->tolerance = tolerance;
        this->a = a;
        this->b = b;
        this->c = c;
        this->d = d;
        this->e = e;
        this->f = f;
        this->g = g;
        this->eta = eta;
        this->h = h;
        this->mu = mu;
        this->rreach = rreach;
        this->dreach = dreach;
        this->remix = remix;
        this->rm_middle = rm_middle;
        this->dmax = dmax;
        this->w = w;
        this->tb = tb;
        this->rwdP = std::make_shared<RWD::Reward>(robot_r, this->vmax, this->rmax, tolerance,
                                                   a, b, c, d, e, f, g, eta, h, mu, rreach, dreach, remix, rm_middle, dmax, w, tb);
        return true;
    }
    double Environment::setCtrl(double vmax, double tau, double wheel_r, double wheel_d, double gain)
    {
        this->vmax = vmax;
        this->tau = tau;
        this->wheel_r = wheel_r;
        this->wheel_d = wheel_d;
        this->gain = gain;
        this->ctrlP = std::make_shared<CTRL::CtrlConverter>(vmax, tau, wheel_r, wheel_d, gain);
        this->rmax = this->ctrlP->get_rmax();
        return this->rmax;
    }
    bool Environment::setSim(const char *modelfile, int Nrobot, points_t target, contours_t contour, bool isRender, int W, int H)
    {
        this->isRender = isRender;
        this->Nrobot = Nrobot;
        this->target = target;
        this->contour = contour;
        this->H = H;
        this->W = W;
        this->simP = std::make_shared<SIM::Simulator>(modelfile, Nrobot, isRender, W, H);
        this->posvels = simP->posvels;
        memset(this->death, 0, Nrobot * sizeof(int));
        return true;
    }
    bool Environment::setRvop(double dmax, double robot_r)
    {
        this->rvopP = std::make_shared<RVO::RVOcalculator>(dmax, robot_r);
        return true;
    }
    bool Environment::cal_obs(bool avevel)
    {
        observations.clear();
        obs_t obs;
        for (int j = 0; j < this->Nrobot; j++)
        {
            observations.push_back(observation_t{});
            for (size_t i = 0; i < this->contour.size(); i++)
            {
                this->rvopP->RVOplus(this->contour[i].first, this->contour[i].second,
                                     point_t{posvels[j * 6], posvels[j * 6 + 1]}, point_t{posvels[j * 6 + 3], posvels[j * 6 + 4]},
                                     point_t{0, 0}, obs.data(), avevel);
                if (obs[0] > this->rvopP->dmax)
                {
                    continue;
                }
                obs[16] = -1;
                observations[j].push_back(obs);
            }
        }
        for (int j = 0; j < this->Nrobot; j++)
        {
            for (int i = 0; i < this->Nrobot; i++)
            {
                if (i == j)
                {
                    continue;
                }

                this->rvopP->RVOplus(lines_t{}, arcs_t{arc_t{posvels[i * 6 + 0], posvels[i * 6 + 1], 2 * this->rvopP->robot_r, posvels[i * 6 + 0] - 2 * this->rvopP->robot_r, posvels[i * 6 + 1], posvels[i * 6] + 2 * this->rvopP->robot_r, posvels[i * 6 + 1], posvels[i * 6], posvels[i * 6 + 1] + 2 * this->rvopP->robot_r}, arc_t{posvels[i * 6], posvels[i * 6 + 1], 2 * this->rvopP->robot_r, posvels[i * 6] - 2 * this->rvopP->robot_r, posvels[i * 6 + 1], posvels[i * 6] + 2 * this->rvopP->robot_r, posvels[i * 6 + 1], posvels[i * 6], posvels[i * 6 + 1] - 2 * this->rvopP->robot_r}},
                                     point_t{posvels[j * 6], posvels[j * 6 + 1]}, point_t{posvels[j * 6 + 3], posvels[j * 6 + 4]},
                                     point_t{posvels[i * 6 + 3], posvels[i * 6 + 4]}, obs.data(), avevel);
                if (obs[0] > this->rvopP->dmax)
                {
                    continue;
                }
                obs[8] = posvels[i * 6];
                obs[9] = posvels[i * 6 + 1];
                obs[10] = posvels[i * 6 + 2];
                obs[11] = posvels[i * 6 + 3];
                obs[12] = posvels[i * 6 + 4];
                obs[13] = posvels[i * 6 + 5];
                obs[14] = target[i][0];
                obs[15] = target[i][1];
                obs[16] = i;
                observations[j].push_back(obs);
            }
        }
        for (int j = 0; j < this->Nrobot; j++)
        {
            std::sort(observations[j].begin(), observations[j].end(), [](obs_t const &a, obs_t const &b)
                      { return a[6] * a[6] + a[7] * a[7] < b[6] * b[6] + b[7] * b[7]; });
        }
        return true;
    }
    bool Environment::cal_NNinput1(double Nullfill)
    {
        std::fill(
            &this->NNinput1[0][0],
            &this->NNinput1[0][0] + 180 * Nrobot,
            Nullfill);
        for (int Nth = 0; Nth < this->Nrobot; Nth++)
        {
            tranM[0][0] = cos(posvels[Nth * 6 + 2]);
            tranM[0][1] = sin(posvels[Nth * 6 + 2]);
            tranM[1][0] = -tranM[0][1];
            tranM[1][1] = tranM[0][0];
            // self
            NNinput1[Nth][0] = tranM[0][0] * (target[Nth][0] - posvels[Nth * 6]) + tranM[0][1] * (target[Nth][1] - posvels[Nth * 6 + 1]);
            NNinput1[Nth][1] = tranM[1][0] * (target[Nth][0] - posvels[Nth * 6]) + tranM[1][1] * (target[Nth][1] - posvels[Nth * 6 + 1]);
            NNinput1[Nth][2] = NORM(posvels[Nth * 6 + 3], posvels[Nth * 6 + 4]);
            NNinput1[Nth][3] = posvels[Nth * 6 + 5];
            // surrounding
            for (size_t i = 0; i < std::min(observations[Nth].size(), std::size_t(16)); i++)
            {
                NNinput1[Nth][i * 10 + 4] = tranM[0][0] * observations[Nth][i][0] + tranM[0][1] * observations[Nth][i][1];
                NNinput1[Nth][i * 10 + 5] = tranM[1][0] * observations[Nth][i][0] + tranM[1][1] * observations[Nth][i][1];

                NNinput1[Nth][i * 10 + 6] = tranM[0][0] * observations[Nth][i][2] + tranM[0][1] * observations[Nth][i][3];
                NNinput1[Nth][i * 10 + 7] = tranM[1][0] * observations[Nth][i][2] + tranM[1][1] * observations[Nth][i][3];

                NNinput1[Nth][i * 10 + 8] = tranM[0][0] * observations[Nth][i][4] + tranM[0][1] * observations[Nth][i][5];
                NNinput1[Nth][i * 10 + 9] = tranM[1][0] * observations[Nth][i][4] + tranM[1][1] * observations[Nth][i][5];

                NNinput1[Nth][i * 10 + 10] = tranM[0][0] * observations[Nth][i][6] + tranM[0][1] * observations[Nth][i][7];
                NNinput1[Nth][i * 10 + 11] = tranM[1][0] * observations[Nth][i][6] + tranM[1][1] * observations[Nth][i][7];

                if (observations[Nth][i][16] < 0)
                {
                    // static obstacle
                    NNinput1[Nth][i * 10 + 12] = 0;
                    NNinput1[Nth][i * 10 + 13] = 0;
                }
                else
                {
                    // other robot
                    NNinput1[Nth][i * 10 + 12] = tranM[0][0] * (observations[Nth][i][14] - observations[Nth][i][8]) + tranM[0][1] * (observations[Nth][i][15] - observations[Nth][i][9]);
                    NNinput1[Nth][i * 10 + 13] = tranM[1][0] * (observations[Nth][i][14] - observations[Nth][i][8]) + tranM[1][1] * (observations[Nth][i][15] - observations[Nth][i][9]);
                }
            }
        }
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
    py::memoryview Environment::get_r()
    {
        return py::memoryview::from_buffer(
            this->reward,
            {this->Nrobot},
            {sizeof(reward[0])});
    }
    py::memoryview Environment::get_rm()
    {
        return py::memoryview::from_buffer(
            this->reward_mix,
            {this->Nrobot},
            {sizeof(reward_mix[0])});
    }
    py::memoryview Environment::get_d()
    {
        return py::memoryview::from_buffer(
            this->death,
            {this->Nrobot},
            {sizeof(death[0])});
    }
    py::memoryview Environment::get_NNinput1()
    {
        return py::memoryview::from_buffer(
            &this->NNinput1[0][0],
            {Nrobot, 180},
            {sizeof(double) * 180,
             sizeof(double)});
    }
    Environment::~Environment()
    {
    }
} // namespace ENV
