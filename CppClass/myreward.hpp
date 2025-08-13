#ifndef __reward__
#define __reward__
#include "pybind11_common_header.hpp"
#include "typesdef.hpp"

namespace RWD
{
    class Reward
    {
    private:
        double robot_r, vmax, rmax, tolerance, a, b, c, d, e, f, g, h, eta, mu, rreach, dreach2;
        bool remix;
        int rm_middle;
        double selfw;
        double dmax, w;
        double tb;
        int remix_count;
        double remix_sum;
        double weight_sum;
        double weight;

        std::size_t Nrobot;
        double lenp2t, lenvnear, vdotvnear, lenp2p, rvdotp2p;

    public:
        Reward();
        Reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu, bool remix, int rm_middle, double dmax, double w, double tb);
        Reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu, double rreach, double dreach, bool remix, int rm_middle, double dmax, double w, double tb);
        ~Reward();
        // def reward(self, pos_vel
        //            : np.ndarray, observation
        //            : list, target
        //            : np.ndarray) :
        std::vector<double> calreward(const posvels_t &posvels,
                                      const observations_t &observations,
                                      const points_t &target);
        void calreward(const double *posvels,
                       const observations_t &observations,
                       const points_t &target, double *reward, double *reward_mix, int *death);
    };
} // namespace RWD

#endif
