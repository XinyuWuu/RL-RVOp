#ifndef __observator__
#define __observator__
#include "pybind11_common_header.hpp"
#include "typesdef.hpp"
#include "RVOcalculator.hpp"
#include "myreward.hpp"
#include "observator.hpp"
namespace OBS
{
    class Observator
    {
    private:
        RWD::Reward Rwd;

    public:
        contours_t contours;
        points_t target;
        RVO::RVOcalculator RVOp;
        OBSreturn get_obs(posvels_t posvels);
        void get_NNinput(const posvels_t &posvels, const observations_t &observations, const points_t &target, NNinput_t &NNinput);
        void set_model(contours_t contours, points_t target);
        void set_reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu, bool remix, int rm_middle);
        void change_robot(double dmax, double robot_r);
        Observator(/* args */);
        Observator(double dmax, double robot_r);
        ~Observator();
    };
} // namespace OBS

#endif
