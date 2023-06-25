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
            // target + time bias
            lenp2t = NORM(posvels[Nth][0] - target[Nth][0], posvels[Nth][1] - target[Nth][1]);
            r[Nth] += this->a * DOT(posvels[Nth][3] / this->vmax,
                                    posvels[Nth][4] / this->vmax,
                                    (target[Nth][0] - posvels[Nth][0]) / lenp2t,
                                    (target[Nth][1] - posvels[Nth][1]) / lenp2t) +
                      this->tb;
            // return self.a * dot(vel / self.vmax, (target - pos) / norm(target - pos))
            for (const obs_t &o : observations[Nth])
            {
                lenvnear = NORM(o[6], o[7]);
                vdotvnear = DOT(posvels[Nth][3] / this->vmax,
                                posvels[Nth][4] / this->vmax,
                                o[6] / lenvnear,
                                o[7] / lenvnear);
                if (o[16] < 0)
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
        if (!this->remix)
        {
            return r;
        }
        else
        {
            std::vector<double> r_m(Nrobot);
            int remix_count = 0;
            double remix_sum = 0.0;
            double weight_sum = 0.0;
            double weight = 0.0;
            for (size_t Nth = 0; Nth < Nrobot; Nth++)
            {
                remix_count = 0;
                remix_sum = 0.0;
                weight_sum = 0.0;
                for (const obs_t &o : observations[Nth])
                {
                    if (o[16] > -0.5)
                    {
                        remix_count += 1;
                        weight = w * (this->dmax - NORM(o[6], o[7])) / this->dmax + 1;
                        remix_sum += r[int(o[16])] * weight;
                        weight_sum += weight;
                    }
                }
                remix_sum /= weight_sum ? remix_count != 0 : 1;
                remix_count = this->rm_middle ? this->rm_middle < remix_count : remix_count;
                r_m[Nth] = r[Nth] * this->selfw / (this->selfw + remix_count) +
                           remix_sum * remix_count / (this->selfw + remix_count);
            }
            return r_m;
        }
    }
    void Reward::calreward(const double *posvels,
                           const observations_t &observations,
                           const points_t &target, double *reward, double *reward_mix, int *death)
    {

        memset(reward, 0, Nrobot * sizeof(double));
        memset(reward_mix, 0, Nrobot * sizeof(double));
        for (std::size_t Nth = 0; Nth < Nrobot; Nth++)
        {
            if (death[Nth])
            {
                continue;
            }
            // target + time bias
            lenp2t = NORM(posvels[Nth * 6] - target[Nth][0], posvels[Nth * 6 + 1] - target[Nth][1]);
            reward[Nth] += this->a * DOT(posvels[Nth * 6 + 3] / this->vmax,
                                         posvels[Nth * 6 + 4] / this->vmax,
                                         (target[Nth][0] - posvels[Nth * 6 + 0]) / lenp2t,
                                         (target[Nth][1] - posvels[Nth * 6 + 1]) / lenp2t) +
                           this->tb;
            for (const obs_t &o : observations[Nth])
            {
                lenvnear = NORM(o[6], o[7]);
                vdotvnear = DOT(posvels[Nth * 6 + 3] / this->vmax,
                                posvels[Nth * 6 + 4] / this->vmax,
                                o[6] / lenvnear,
                                o[7] / lenvnear);
                if (o[16] < 0)
                {
                    // static obstacle
                    // obs
                    if (lenvnear < this->tolerance)
                    {
                        reward[Nth] += -this->b * vdotvnear - this->c;
                    }
                    reward[Nth] += -this->g * (vdotvnear + 1) * exp(-lenvnear / this->eta / this->robot_r);
                }
                else
                {
                    // other robot
                    lenp2p = NORM(posvels[Nth * 6] - o[8], posvels[Nth * 6 + 1] - o[9]);
                    rvdotp2p = DOT((posvels[Nth * 6 + 3] - o[11]) / this->vmax,
                                   (posvels[Nth * 6 + 4] - o[12]) / this->vmax,
                                   (o[8] - posvels[Nth * 6 + 0]) / lenp2p,
                                   (o[9] - posvels[Nth * 6 + 1]) / lenp2p);
                    if (lenvnear < this->tolerance)
                    {

                        reward[Nth] += -this->d * rvdotp2p - this->f;
                    }
                    reward[Nth] += -this->h * (rvdotp2p + 2) * exp(-lenvnear / this->mu / this->robot_r);
                }
            }

            if (dreach2 > NORM2(posvels[Nth * 6] - target[Nth][0], posvels[Nth * 6 + 1] - target[Nth][1]))
            {
                reward[Nth] += rreach;
            }
        }
        if (!this->remix)
        {
            return;
        }
        for (std::size_t Nth = 0; Nth < Nrobot; Nth++)
        {
            remix_count = 0;
            remix_sum = 0.0;
            weight_sum = 0.0;
            for (const obs_t &o : observations[Nth])
            {
                if (o[16] > -0.5)
                {
                    remix_count += 1;
                    weight = w * (this->dmax - NORM(o[6], o[7])) / this->dmax + 1;
                    remix_sum += reward[int(o[16])] * weight;
                    weight_sum += weight;
                }
            }
            remix_sum /= weight_sum ? remix_count != 0 : 1;
            remix_count = this->rm_middle ? this->rm_middle < remix_count : remix_count;
            reward_mix[Nth] = reward[Nth] * this->selfw / (this->selfw + remix_count) +
                              remix_sum * remix_count / (this->selfw + remix_count);

            return;
        }
    }
    Reward::Reward() {}
    Reward::Reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu, bool remix, int rm_middle, double dmax, double w, double tb)
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
        this->remix = remix;
        this->rm_middle = rm_middle;
        this->selfw = this->rm_middle * 3;
        this->dmax = dmax;
        this->w = w;
        this->tb = tb;
    }
    Reward::Reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu, double rreach, double dreach, bool remix, int rm_middle, double dmax, double w, double tb)
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
        this->rreach = rreach;
        this->dreach2 = dreach * dreach;
        this->remix = remix;
        this->rm_middle = rm_middle;
        this->selfw = this->rm_middle * 3;
        this->dmax = dmax;
        this->w = w;
        this->tb = tb;
    }
    Reward::~Reward()
    {
    }

} // namespace RWD
