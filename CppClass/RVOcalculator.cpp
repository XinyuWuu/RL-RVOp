#include "RVOcalculator.hpp"
#include "funcsdef.hpp"

namespace RVO
{

    RVOcalculator::RVOcalculator(/* args */)
    {
        this->dmax = 3;
        this->robot_r = 0.17;
    }
    RVOcalculator::RVOcalculator(double dmax, double robot_r)
    {
        this->dmax = dmax;
        this->robot_r = robot_r;
    }
    RVOcalculator::RVOcalculator(double dmax, double robot_r, contours_t contours, points_t target)
    {
        this->dmax = dmax;
        this->robot_r = robot_r;
        this->contours = contours;
        this->target = target;
    }
    bool RVOcalculator::inLine(const line_t &line, const point_t &x0)
    {
        // return dot(line[2:4] - x0, line[4:6] - x0) <= 0
        return DOT(line[2] - x0[0], line[3] - x0[1],
                   line[4] - x0[0], line[5] - x0[1]) <= 0;
    }
    bool RVOcalculator::inArc(const arc_t &arc, const point_t &x0)
    {
        // return cross(arc[3:5] - arc[0:2], x0 - arc[0:2]) * \
            // cross(arc[3:5] - arc[0:2], arc[7:9] - arc[0:2]) >= 0 and\
            // cross(arc[5:7] - arc[0:2], x0 - arc[0:2]) * \
            // cross(arc[5:7] - arc[0:2], arc[7:9] - arc[0:2]) >= 0
        return CROSS(arc[3] - arc[0], arc[4] - arc[1], x0[0] - arc[0], x0[1] - arc[1]) *
                       CROSS(arc[3] - arc[0], arc[4] - arc[1], arc[7] - arc[0], arc[8] - arc[1]) >=
                   0 &&
               CROSS(arc[5] - arc[0], arc[6] - arc[1], x0[0] - arc[0], x0[1] - arc[1]) *
                       CROSS(arc[5] - arc[0], arc[6] - arc[1], arc[7] - arc[0], arc[8] - arc[1]) >=
                   0;
    }
    void RVOcalculator::LineP(const line_t &line, const point_t &xr, points_t &points)
    { // return posible point in line, the last one is the nearest x0

        // points = []
        const size_t ps_pre = points.size();
        if (line[0] > 1000)
        {
            return;
        }

        // x0 = dot(xr - line[2:4], line[4:6] - line[2:4]) / \
        //     norm(line[4:6] - line[2:4])**2 * \
        //     (line[4:6] - line[2:4]) + line[2:4]
        double tem = DOT(xr[0] - line[2], xr[1] - line[3],
                         line[4] - line[2], line[5] - line[3]) /
                     NORM2(line[4] - line[2], line[5] - line[3]);
        point_t x0{(line[4] - line[2]) * tem + line[2],
                   (line[5] - line[3]) * tem + line[3]};

        // if norm(x0 - xr) > self.dmax:
        //     return points
        // if dot(x0 - xr, line[0:2]) < 0:
        //     return points
        if (NORM2(x0[0] - xr[0], x0[1] - xr[1]) > this->dmax * this->dmax ||
            DOT(x0[0] - xr[0], x0[1] - xr[1], line[0], line[1]) < 0)
        {
            return;
        }
        // if norm(line[2:4] - xr) < self.dmax:
        //     points.append(line[2:4])
        // if norm(line[4:6] - xr) < self.dmax:
        //     points.append(line[4:6])
        if (NORM2(line[2] - xr[0], line[3] - xr[1]) < this->dmax * this->dmax)
        {
            points.push_back(point_t{line[2], line[3]});
        }
        if (NORM2(line[4] - xr[0], line[5] - xr[1]) < this->dmax * this->dmax)
        {
            points.push_back(point_t{line[4], line[5]});
        }
        // if len(points) < 2:
        //     l = sqrt(abs(self.dmax**2 - norm(xr - x0)**2)) * \
        //         (line[4:6] - line[2:4]) / norm(line[4:6] - line[2:4])
        //     xc1 = x0 + l
        //     xc2 = x0 - l
        //     if self.inLine(line, xc1):
        //         points.append(xc1)
        //     if self.inLine(line, xc2):
        //         points.append(xc2)
        if (points.size() - ps_pre < 2)
        {
            tem = sqrt(abs(this->dmax * this->dmax - NORM2(x0[0] - xr[0], x0[1] - xr[1]))) /
                  NORM(line[4] - line[2], line[5] - line[3]);
            point_t xc1{x0[0] + tem * (line[4] - line[2]),
                        x0[1] + tem * (line[5] - line[3])};
            point_t xc2{x0[0] - tem * (line[4] - line[2]),
                        x0[1] - tem * (line[5] - line[3])};
            if (this->inLine(line, xc1))
            {
                points.push_back(xc1);
            }
            if (this->inLine(line, xc2))
            {
                points.push_back(xc2);
            }
        }
        // if self.inLine(line, x0):
        //     points.append(x0)
        if (this->inLine(line, x0))
        {
            points.push_back(x0);
        }
        // return points
        return;
    }
    void RVOcalculator::ArcP(const arc_t &arc, const point_t &xr, points_t &points)
    { // return posible point in arc, the last one is the nearest x0

        //  points = []
        //  x0 = arc[0:2] + arc[2] * (xr - arc[0:2]) / norm(xr - arc[0:2])
        const size_t ps_pre = points.size();
        double tem = arc[2] / NORM(xr[0] - arc[0], xr[1] - arc[1]);
        point_t x0{arc[0] + tem * (xr[0] - arc[0]),
                   arc[1] + tem * (xr[1] - arc[1])};

        // if norm(x0 - xr) > self.dmax:
        //     return points
        if (NORM2(x0[0] - xr[0], x0[1] - xr[1]) > this->dmax * this->dmax)
        {
            return;
        }

        // tangent points

        // cose1 = arc[2] / norm(arc[0:2] - xr)
        // cose1 = min(cose1, 1.0)
        // sine1 = float(sqrt(1 - cose1**2))
        double cose1 = fmin(arc[2] / NORM(arc[0] - xr[0], arc[1] - xr[1]), 1);
        double sine1 = sqrt(1 - cose1 * cose1);

        // xc1 = np.matmul(array([[cose1, -sine1],
        //                        [sine1, cose1]]), (x0 - arc[0:2])) + arc[0:2]
        point_t xc1{
            arc[0] + cose1 * (x0[0] - arc[0]) - sine1 * (x0[1] - arc[1]),
            arc[1] + sine1 * (x0[0] - arc[0]) + cose1 * (x0[1] - arc[1])};
        // xc2 = np.matmul(array([[cose1, sine1],
        //                        [-sine1, cose1]]), (x0 - arc[0:2])) + arc[0:2]
        point_t xc2{
            arc[0] + cose1 * (x0[0] - arc[0]) + sine1 * (x0[1] - arc[1]),
            arc[1] - sine1 * (x0[0] - arc[0]) + cose1 * (x0[1] - arc[1])};

        // if norm(xr - xc1) <= self.dmax and self.inArc(arc, xc1):
        //     points.append(xc1)
        // if norm(xr - xc2) <= self.dmax and self.inArc(arc, xc2):
        //     points.append(xc2)
        if (NORM2(xc1[0] - xr[0], xc1[1] - xr[1]) < this->dmax * this->dmax &&
            this->inArc(arc, xc1))
        {
            points.push_back(xc1);
        }
        if (NORM2(xc2[0] - xr[0], xc2[1] - xr[1]) < this->dmax * this->dmax &&
            this->inArc(arc, xc2))
        {
            points.push_back(xc2);
        }

        if (points.size() - ps_pre > 0)
        {
            if (this->inArc(arc, x0))
            {
                points.push_back(x0);
            }
            return;
        }

        // dmax points
        // cose2 = -(self.dmax**2 - norm(xr - arc[0:2])**2 -
        //           arc[2]**2) / 2 / norm(xr - arc[0:2]) / arc[2]
        double cose2 = -(this->dmax * this->dmax -
                         NORM2(xr[0] - arc[0], xr[1] - arc[1]) -
                         arc[2] * arc[2]) /
                       2 / NORM(xr[0] - arc[0], xr[1] - arc[1]) / arc[2];
        // if abs(cose2) < 1.000001:
        //     cose2 = min(cose2, 1.0)
        //     cose2 = max(cose2, -1.0)
        //     sine2 = float(sqrt(1 - cose2**2))
        if (abs(cose2) < 1.000001)
        {
            cose2 = fmin(1, fmax(-1, cose2));
            double sine2 = sqrt(1 - cose2 * cose2);
            //     xc1 = np.matmul(array([[cose2, -sine2],
            //                            [sine2, cose2]]), (x0 - arc[0:2])) + arc[0:2]
            //     xc2 = np.matmul(array([[cose2, sine2],
            //                            [-sine2, cose2]]), (x0 - arc[0:2])) + arc[0:2]

            xc1[0] = arc[0] + cose2 * (x0[0] - arc[0]) - sine2 * (x0[1] - arc[1]);
            xc1[1] = arc[1] + sine2 * (x0[0] - arc[0]) + cose2 * (x0[1] - arc[1]);
            xc2[0] = arc[0] + cose2 * (x0[0] - arc[0]) + sine2 * (x0[1] - arc[1]);
            xc2[1] = arc[1] - sine2 * (x0[0] - arc[0]) + cose2 * (x0[1] - arc[1]);
            //     if self.inArc(arc, xc1):
            //         points.append(xc1)
            //     if self.inArc(arc, xc2):
            //         points.append(xc2)
            if (this->inArc(arc, xc1))
            {
                points.push_back(xc1);
            }
            if (this->inArc(arc, xc2))
            {
                points.push_back(xc2);
            }
        }

        // if norm(xr - arc[3:5]) <= self.dmax:
        //     points.append(arc[3:5])
        // if norm(xr - arc[5:7]) <= self.dmax:
        //     points.append(arc[5:7])
        if (NORM2(arc[3] - xr[0], arc[4] - xr[1]) < this->dmax * this->dmax)
        {
            points.push_back(point_t{arc[3], arc[4]});
        }
        if (NORM2(arc[5] - xr[0], arc[6] - xr[1]) < this->dmax * this->dmax)
        {
            points.push_back(point_t{arc[5], arc[6]});
        }

        // if self.inArc(arc, x0):
        //     points.append(x0)
        // return points
        if (this->inArc(arc, x0))
        {
            points.push_back(x0);
        }
        return;
    }
    rvop_t RVOcalculator::RVOplus(lines_t lines, arcs_t arcs, point_t xr, point_t vr, point_t vo)
    {
        // points = []
        // vnear = array([10000, 10000])
        // dmin = 100000
        points_t points;
        long unsigned int p_size = points.size();
        point_t vnear{10000, 10000};
        double d2min = 100000;
        double d2 = 0;
        // for line in lines:
        //     if len(line) == 0:
        //         continue
        //     ps = self.LineP(line, xr)
        //     if len(ps) == 0:
        //         continue
        //     else:
        //         points += ps
        //     if norm(points[-1] - xr) < dmin:
        //         dmin = norm(points[-1] - xr)
        //         vnear = points[-1] - xr
        for (const line_t &line : lines)
        {
            LineP(line, xr, points);
            if (points.size() == p_size)
            {
                continue;
            }
            else
            {
                p_size = points.size();
            }

            d2 = NORM2(points.back()[0] - xr[0], points.back()[1] - xr[1]);
            if (d2 < d2min)
            {
                d2min = d2;
                vnear = points.back();
            }
        }
        // for arc in arcs:
        //     ps = self.ArcP(arc, xr)
        //     if len(ps) == 0:
        //         continue
        //     else:
        //         points += ps
        //     if norm(points[-1] - xr) < dmin:
        //         dmin = norm(points[-1] - xr)
        //         vnear = points[-1] - xr
        for (const arc_t &arc : arcs)
        {
            ArcP(arc, xr, points);
            if (points.size() == p_size)
            {
                continue;
            }
            else
            {
                p_size = points.size();
            }
            d2 = NORM2(points.back()[0] - xr[0], points.back()[1] - xr[1]);
            if (d2 < d2min)
            {
                d2min = d2;
                vnear = points.back();
            }
        }
        vnear[0] = vnear[0] - xr[0];
        vnear[1] = vnear[1] - xr[1];
        // lvnear = norm(vnear)
        // if lvnear > self.dmax:
        //     return vnear
        if (d2min > this->dmax * this->dmax)
        {
            return rvop_t{10000, 10000, 10000, 10000, 10000, 100000, 10000, 100000};
        }
        // emax = 0
        // v1 = array([100000, 100000])
        // emin = 0
        // v2 = array([100000, 100000])
        // vs = array(points) - xr
        double emax = 0, emin = 0, e = 0;
        point_t v1{100000, 100000}, v2{100000, 100000};
        for (point_t &point : points)
        {
            point[0] = point[0] - xr[0];
            point[1] = point[1] - xr[1];
        }
        // for v in vs:
        //     lv = norm(v)
        //     # e = cross(vnear, v) / lvnear / lv
        //     e = arctan2(cross(vnear, v), dot(vnear, v))
        //     if e >= emax:
        //         v1 = v
        //         emax = e
        //     if e <= emin:
        //         v2 = v
        //         emin = e
        for (const point_t &v : points)
        {
            e = atan2(CROSS(vnear[0], vnear[1], v[0], v[1]),
                      DOT(vnear[0], vnear[1], v[0], v[1]));
            if (e >= emax)
            {
                v1 = v;
                emax = e;
            }
            if (e <= emin)
            {
                v2 = v;
                emin = e;
            }
        }
        // return np.hstack([ (vr + vo) / 2, v1, v2, vnear ])
        return rvop_t{(vr[0] + vo[0]) / 2, (vr[1] + vo[1]) / 2,
                      v1[0], v1[1],
                      v2[0], v2[1],
                      vnear[0], vnear[1]};
    }
    void RVOcalculator::RVOplus(const lines_t &lines, const arcs_t &arcs, const point_t &xr, const point_t &vr, const point_t &vo, double *obs, bool avevel)
    {

        // std::cout << "C++ view:" << std::endl
        //           << lines << std::endl
        //           << arcs << std::endl;
        // points = []
        // vnear = array([10000, 10000])
        // dmin = 100000
        points_t points;
        long unsigned int p_size = points.size();
        point_t vnear{10000, 10000};
        double d2min = 100000;
        double d2 = 0;
        // for line in lines:
        //     if len(line) == 0:
        //         continue
        //     ps = self.LineP(line, xr)
        //     if len(ps) == 0:
        //         continue
        //     else:
        //         points += ps
        //     if norm(points[-1] - xr) < dmin:
        //         dmin = norm(points[-1] - xr)
        //         vnear = points[-1] - xr
        for (const line_t &line : lines)
        {
            LineP(line, xr, points);
            if (points.size() == p_size)
            {
                continue;
            }
            else
            {
                p_size = points.size();
            }

            d2 = NORM2(points.back()[0] - xr[0], points.back()[1] - xr[1]);
            if (d2 < d2min)
            {
                d2min = d2;
                vnear = points.back();
            }
        }
        // for arc in arcs:
        //     ps = self.ArcP(arc, xr)
        //     if len(ps) == 0:
        //         continue
        //     else:
        //         points += ps
        //     if norm(points[-1] - xr) < dmin:
        //         dmin = norm(points[-1] - xr)
        //         vnear = points[-1] - xr
        for (const arc_t &arc : arcs)
        {
            ArcP(arc, xr, points);
            if (points.size() == p_size)
            {
                continue;
            }
            else
            {
                p_size = points.size();
            }
            d2 = NORM2(points.back()[0] - xr[0], points.back()[1] - xr[1]);
            if (d2 < d2min)
            {
                d2min = d2;
                vnear = points.back();
            }
        }
        vnear[0] = vnear[0] - xr[0];
        vnear[1] = vnear[1] - xr[1];
        // lvnear = norm(vnear)
        // if lvnear > self.dmax:
        //     return vnear
        if (d2min > this->dmax * this->dmax)
        {
            // return rvop_t{10000, 10000, 10000, 10000, 10000, 100000, 10000, 100000};
            *obs = 1000000;
            return;
        }
        // emax = 0
        // v1 = array([100000, 100000])
        // emin = 0
        // v2 = array([100000, 100000])
        // vs = array(points) - xr
        double emax = 0, emin = 0, e = 0;
        point_t v1{100000, 100000}, v2{100000, 100000};
        for (point_t &point : points)
        {
            point[0] = point[0] - xr[0];
            point[1] = point[1] - xr[1];
        }
        // for v in vs:
        //     lv = norm(v)
        //     # e = cross(vnear, v) / lvnear / lv
        //     e = arctan2(cross(vnear, v), dot(vnear, v))
        //     if e >= emax:
        //         v1 = v
        //         emax = e
        //     if e <= emin:
        //         v2 = v
        //         emin = e
        for (const point_t &v : points)
        {
            e = atan2(CROSS(vnear[0], vnear[1], v[0], v[1]),
                      DOT(vnear[0], vnear[1], v[0], v[1]));
            if (e >= emax)
            {
                v1 = v;
                emax = e;
            }
            if (e <= emin)
            {
                v2 = v;
                emin = e;
            }
        }
        // return np.hstack([ (vr + vo) / 2, v1, v2, vnear ])
        // return rvop_t{(vr[0] + vo[0]) / 2, (vr[1] + vo[1]) / 2,
        //               v1[0], v1[1],
        //               v2[0], v2[1],
        //               vnear[0], vnear[1]};
        if (avevel)
        {
            *obs = (vr[0] + vo[0]) / 2;
            *(obs + 1) = (vr[1] + vo[1]) / 2;
        }
        else
        {
            *obs = vo[0];
            *(obs + 1) = vo[1];
        }

        *(obs + 2) = v1[0];
        *(obs + 3) = v1[1];
        *(obs + 4) = v2[0];
        *(obs + 5) = v2[1];
        *(obs + 6) = vnear[0];
        *(obs + 7) = vnear[1];
        return;
    }

    observations_t RVOcalculator::get_obs(posvels_t posvels, bool avevel)
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
                this->RVOplus(this->contours[i].first, this->contours[i].second,
                              point_t{posvels[j][0], posvels[j][1]}, point_t{posvels[j][3], posvels[j][4]},
                              point_t{0, 0}, obs.data(), avevel);
                obs[15] = 1000000;
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
                this->RVOplus(lines_t{}, arcs_t{arc_t{posvels[i][0], posvels[i][0], 2 * this->robot_r, posvels[i][0] - 2 * this->robot_r, posvels[i][0], posvels[i][0] + 2 * this->robot_r, posvels[i][0], posvels[i][0], posvels[i][0] + 2 * this->robot_r}, arc_t{posvels[i][0], posvels[i][0], 2 * this->robot_r, posvels[i][0] - 2 * this->robot_r, posvels[i][0], posvels[i][0] + 2 * this->robot_r, posvels[i][0], posvels[i][0], posvels[i][0] - 2 * this->robot_r}},
                              point_t{posvels[j][0], posvels[j][1]}, point_t{posvels[j][3], posvels[j][4]},
                              point_t{posvels[i][3], posvels[i][4]}, obs.data(), avevel);
                obs[8] = posvels[i][0];
                obs[9] = posvels[i][1];
                obs[10] = posvels[i][2];
                obs[11] = posvels[i][3];
                obs[12] = posvels[i][4];
                obs[13] = posvels[i][5];
                obs[14] = target[i][0];
                obs[15] = target[i][1];
                // observations[j].push_back(obs_t{this->RVOplus(lines_t{}, arcs_t{arc_t{posvel[i][0], posvel[i][0], 2 * this->robot_r, posvel[i][0] - 2 * this->robot_r, posvel[i][0], posvel[i][0] + 2 * this->robot_r, posvel[i][0], posvel[i][0], posvel[i][0] + 2 * this->robot_r}, arc_t{posvel[i][0], posvel[i][0], 2 * this->robot_r, posvel[i][0] - 2 * this->robot_r, posvel[i][0], posvel[i][0] + 2 * this->robot_r, posvel[i][0], posvel[i][0], posvel[i][0] - 2 * this->robot_r}}, point_t{posvel[j][0], posvel[j][1]}, point_t{posvel[j][3], posvel[j][4]}, point_t{posvel[i][3], posvel[i][4]}), vector3d{posvel[i][0], posvel[i][1], posvel[i][2]}, vector3d{posvel[i][3], posvel[i][4], posvel[i][5]}, this->target[i]});
                observations[j].push_back(obs);
            }
        }
        return observations;
    }

    RVOcalculator::~RVOcalculator()
    {
    }

} // namespace RVO
