#include "myreward.hpp"
#include "funcsdef.hpp"
namespace RWD
{
    std::vector<double> Reward::calreward(const posvels_t &posvels,
                                          const observations_t &observations,
                                          const points_t &target)
    {
        // Nrobot = pos_vel.shape[0]
        const std::size_t Nrobot = posvels.size();
        // r = zeros((Nrobot,))
        // memset(r, 0, Nrobot * sizeof(double));
        std::vector<double> r(Nrobot, 0.0);
        double lenp2t = 0;
        double lenvnear = 0;
        double vdotvnear = 0;
        double lenp2p = 0;
        double rvdotp2p = 0;
        for (size_t Nth = 0; Nth < Nrobot; Nth++)
        {
            // target
            lenp2t = NORM(posvels[Nth][0] - target[Nth][0], posvels[Nth][1] - target[Nth][1]);
            r[Nth] += this->a * DOT(posvels[Nth][3] / this->vmax,
                                    posvels[Nth][4] / this->vmax,
                                    (target[Nth][0] - posvels[Nth][0]) / lenp2t,
                                    (target[Nth][1] - posvels[Nth][1]) / lenp2t);
            // return self.a * dot(vel / self.vmax, (target - pos) / norm(target - pos))
            for (const obs_t &o : observations[Nth])
            {
                lenvnear = NORM(o[6], o[7]);
                vdotvnear = DOT(posvels[Nth][3] / this->vmax,
                                posvels[Nth][4] / this->vmax,
                                o[6] / lenvnear,
                                o[7] / lenvnear);
                if (o[15] > 1000)
                {
                    // static obstacle
                    // obs
                    if (lenvnear < this->tolerance)
                    {
                        r[Nth] += -this->b * vdotvnear - this->c;
                    }
                    // if norm(vnear) > self.tolerance:
                    //     return 0
                    // return -self.b * dot(vel / self.vmax, vnear / norm(vnear)) - self.c
                    r[Nth] += -this->g * (vdotvnear + 1) * exp(-lenvnear / this->eta / this->robot_r);
                    // return -self.g * (dot(vel / self.vmax, vnear / norm(vnear)) + 1) * exp(-norm(vnear) / self.eta / self.robot_r)
                }
                else
                {
                    // other robot
                    lenp2p = NORM(posvels[Nth][0] - o[8], posvels[Nth][1] - o[9]);
                    rvdotp2p = DOT((posvels[Nth][3] - o[11]) / this->vmax,
                                   (posvels[Nth][4] - o[12]) / this->vmax,
                                   (o[8] - posvels[Nth][0]) / lenp2p,
                                   (o[9] - posvels[Nth][1]) / lenp2p);
                    if (lenvnear < this->tolerance)
                    {

                        r[Nth] += -this->d * rvdotp2p - this->f;
                        // return -self.d * dot((vel - ovel) / self.vmax, (opos - pos) / norm(opos - pos)) - self.f
                    }
                    r[Nth] += -this->h * (rvdotp2p + 2) * exp(-lenvnear / this->mu / this->robot_r);
                    // return -self.h * (dot((vel - ovel) / self.vmax, (opos - pos) / norm(opos - pos)) + 2) *
                    //        exp(-norm(vnear) / self.mu / self.robot_r)
                }
            }
        }
        return r;
    }
    Reward::Reward() {}
    Reward::Reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu)
    {
        this->robot_r = robot_r;
        this->vmax = vmax;
        this->rmax = rmax;
        this->tolerance = tolerance;
        this->a = a;
        this->b = b;
        this->c = c;
        this->d = d;
        this->e = e;
        this->f = f;
        this->g = g;
        this->h = h;
        this->eta = eta;
        this->mu = mu;
    }

    Reward::~Reward()
    {
    }

} // namespace RWD
