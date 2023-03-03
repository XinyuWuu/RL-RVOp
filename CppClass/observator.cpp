#include "observator.hpp"
#include "funcsdef.hpp"

namespace OBS
{

    Observator::Observator(/* args */)
    {
    }
    Observator::Observator(double dmax, double robot_r)
    {
        this->RVOp = RVO::RVOcalculator(dmax, robot_r);
    }
    void Observator::set_model(contours_t contours, points_t target)
    {
        this->contours = contours;
        this->target = target;
    }
    void Observator::set_reward(double robot_r, double vmax, double rmax, double tolerance, double a, double b, double c, double d, double e, double f, double g, double eta, double h, double mu)
    {
        this->Rwd = RWD::Reward(robot_r = robot_r, vmax = vmax, rmax = rmax, tolerance = tolerance,
                                a = a, b = b, c = c, d = d, e = e, f = f, g = g, eta = eta, h = h, mu = mu);
    }

    void Observator::change_robot(double dmax, double robot_r)
    {
        this->RVOp = RVO::RVOcalculator(dmax, robot_r);
    }

    std::pair<observations_t, std::vector<double>> Observator::get_obs(posvels_t posvels)
    {

        observations_t observations;
        // for j in range(self.Nrobot):  # for jth robot
        //     for i in range(len(self.contours)):  # for obstacles
        //         observation[j].append({
        //             'rvop': self.RVO.RVOplus(
        //                 self.contours[i][0], self.contours[i][1], pos_vel[j][0:2], pos_vel[j][3:5], np.zeros((2,))),
        //             "pos": self.obs[i][0:2],
        //             "vel": array([-1, -1]),
        //             "target": array([-1, -1])})
        obs_t obs;
        for (size_t j = 0; j < posvels.size(); j++)
        {
            observations.push_back(observation_t{});
            for (size_t i = 0; i < this->contours.size(); i++)
            {
                this->RVOp.RVOplus(this->contours[i].first, this->contours[i].second,
                                   point_t{posvels[j][0], posvels[j][1]}, point_t{posvels[j][3], posvels[j][4]},
                                   point_t{0, 0}, obs.data());
                if (obs[0] > this->RVOp.dmax)
                {
                    continue;
                }
                obs[15] = 10000000;
                observations[j].push_back(obs);
                // observations[j]
                //     .push_back(
                //         obs_t{
                //             this->RVOplus(this->contours[i].first, this->contours[i].second,
                //                           point_t{posvel[j][0], posvel[j][1]}, point_t{posvel[j][3], posvel[j][4]},
                //                           point_t{0, 0}),
                //             vector3d{0, 0, 0},
                //             vector3d{0, 0, 0},
                //             point_t{-1, -1}});
            }
        }
        // for j in range(self.Nrobot):  # for jth robot
        //     for i in range(self.Nrobot):  # for other robot
        //         if i == j:
        //             continue
        //         con = self.CG.cylinder_contour(
        //             np.array([pos_vel[i][0], pos_vel[i][1], self.robot_r]))
        //         observation[j].append({
        //             'rvop': self.RVO.RVOplus(
        //                 con[0], con[1], pos_vel[j][0:2], pos_vel[j][3:5], pos_vel[i][3:5]),
        //             'pos': pos_vel[i, 0:3],
        //             'vel': pos_vel[i, 3:6],
        //             'target': self.target[i]
        //         })
        for (size_t j = 0; j < posvels.size(); j++)
        {
            for (size_t i = 0; i < posvels.size(); i++)
            {
                if (i == j)
                {
                    continue;
                }

                this->RVOp.RVOplus(lines_t{}, arcs_t{arc_t{posvels[i][0], posvels[i][1], 2 * this->RVOp.robot_r, posvels[i][0] - 2 * this->RVOp.robot_r, posvels[i][1], posvels[i][0] + 2 * this->RVOp.robot_r, posvels[i][1], posvels[i][0], posvels[i][1] + 2 * this->RVOp.robot_r}, arc_t{posvels[i][0], posvels[i][1], 2 * this->RVOp.robot_r, posvels[i][0] - 2 * this->RVOp.robot_r, posvels[i][1], posvels[i][0] + 2 * this->RVOp.robot_r, posvels[i][1], posvels[i][0], posvels[i][1] - 2 * this->RVOp.robot_r}},
                                   point_t{posvels[j][0], posvels[j][1]}, point_t{posvels[j][3], posvels[j][4]},
                                   point_t{posvels[i][3], posvels[i][4]}, obs.data());
                if (obs[0] > this->RVOp.dmax)
                {
                    continue;
                }
                obs[8] = posvels[i][0];
                obs[9] = posvels[i][1];
                obs[10] = posvels[i][2];
                obs[11] = posvels[i][3];
                obs[12] = posvels[i][4];
                obs[13] = posvels[i][5];
                obs[14] = target[i][0];
                obs[15] = target[i][1];
                // observations[j].push_back(obs_t{this->RVOplus(lines_t{}, arcs_t{arc_t{posvel[i][0], posvel[i][0], 2 * robot_r, posvel[i][0] - 2 * robot_r, posvel[i][0], posvel[i][0] + 2 * robot_r, posvel[i][0], posvel[i][0], posvel[i][0] + 2 * robot_r}, arc_t{posvel[i][0], posvel[i][0], 2 * robot_r, posvel[i][0] - 2 * robot_r, posvel[i][0], posvel[i][0] + 2 * robot_r, posvel[i][0], posvel[i][0], posvel[i][0] - 2 * robot_r}}, point_t{posvel[j][0], posvel[j][1]}, point_t{posvel[j][3], posvel[j][4]}, point_t{posvel[i][3], posvel[i][4]}), vector3d{posvel[i][0], posvel[i][1], posvel[i][2]}, vector3d{posvel[i][3], posvel[i][4], posvel[i][5]}, this->target[i]});
                observations[j].push_back(obs);
            }
        }
        return std::pair<observations_t, std::vector<double>>{observations, this->Rwd.calreward(posvels, observations, this->target)};
    }
    Observator::~Observator()
    {
    }
} // namespace OBS
