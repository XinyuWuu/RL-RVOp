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
        std::pair<observations_t, std::vector<double>> get_obs(posvels_t posvel);
        void set_model(contours_t contours, points_t target);
        void set_reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu);
        void change_robot(double dmax, double robot_r);
        Observator(/* args */);
        Observator(double dmax, double robot_r);
        ~Observator();
    };
} // namespace OBS

#endif
