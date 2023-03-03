import numpy as np
from numpy.linalg import norm
from numpy import sqrt, dot, cross, array, arctan2


class RVO():
    def __init__(self, dmax=3.0) -> None:
        self.dmax = dmax

    def inLine(self, line: np.ndarray, x0: np.ndarray):
        return dot(line[2:4] - x0, line[4:6] - x0) <= 0

    def inArc(self, arc: np.ndarray, x0: np.ndarray):
        return cross(arc[3:5] - arc[0:2], x0 - arc[0:2]) * \
            cross(arc[3:5] - arc[0:2], arc[7:9] - arc[0:2]) >= 0 and\
            cross(arc[5:7] - arc[0:2], x0 - arc[0:2]) * \
            cross(arc[5:7] - arc[0:2], arc[7:9] - arc[0:2]) >= 0

    def LineP(self, line: np.ndarray, xr: np.ndarray):
        # return posible point in line, the last one is the nearest x0
        points = []
        x0 = dot(xr - line[2:4], line[4:6] - line[2:4]) / \
            norm(line[4:6] - line[2:4])**2 * \
            (line[4:6] - line[2:4]) + line[2:4]
        if norm(x0 - xr) > self.dmax:
            return points
        if dot(x0 - xr, line[0:2]) < 0:
            return points

        if norm(line[2:4] - xr) < self.dmax:
            points.append(line[2:4])
        if norm(line[4:6] - xr) < self.dmax:
            points.append(line[4:6])

        if len(points) < 2:
            l = sqrt(abs(self.dmax**2 - norm(xr - x0)**2)) * \
                (line[4:6] - line[2:4]) / norm(line[4:6] - line[2:4])
            xc1 = x0 + l
            xc2 = x0 - l
            if self.inLine(line, xc1):
                points.append(xc1)
            if self.inLine(line, xc2):
                points.append(xc2)

        if self.inLine(line, x0):
            points.append(x0)

        return points

    def ArcP(self, arc, xr):
        # return posible point in arc, the last one is the nearest x0
        points = []
        x0 = arc[0:2] + arc[2] * (xr - arc[0:2]) / norm(xr - arc[0:2])
        if norm(x0 - xr) > self.dmax:
            return points

        # tangent points
        cose1 = arc[2] / norm(arc[0:2] - xr)
        cose1 = min(cose1, 1.0)
        sine1 = float(sqrt(1 - cose1**2))
        xc1 = np.matmul(array([[cose1, -sine1],
                               [sine1, cose1]]), (x0 - arc[0:2])) + arc[0:2]
        xc2 = np.matmul(array([[cose1, sine1],
                               [-sine1, cose1]]), (x0 - arc[0:2])) + arc[0:2]
        if norm(xr - xc1) <= self.dmax and self.inArc(arc, xc1):
            points.append(xc1)
        if norm(xr - xc2) <= self.dmax and self.inArc(arc, xc2):
            points.append(xc2)

        # dmax points
        cose2 = -(self.dmax**2 - norm(xr - arc[0:2])**2 -
                  arc[2]**2) / 2 / norm(xr - arc[0:2]) / arc[2]
        if abs(cose2) < 1.000001:
            cose2 = min(cose2, 1.0)
            cose2 = max(cose2, -1.0)
            sine2 = float(sqrt(1 - cose2**2))
            xc1 = np.matmul(array([[cose2, -sine2],
                                   [sine2, cose2]]), (x0 - arc[0:2])) + arc[0:2]
            xc2 = np.matmul(array([[cose2, sine2],
                                   [-sine2, cose2]]), (x0 - arc[0:2])) + arc[0:2]
            if self.inArc(arc, xc1):
                points.append(xc1)
            if self.inArc(arc, xc2):
                points.append(xc2)
        if norm(xr - arc[3:5]) <= self.dmax:
            points.append(arc[3:5])
        if norm(xr - arc[5:7]) <= self.dmax:
            points.append(arc[5:7])
        if self.inArc(arc, x0):
            points.append(x0)
        return points

    def RVOplus(self, lines, arcs, xr, vr, vo):
        points = []
        vnear = array([10000, 10000])
        dmin = 100000
        for line in lines:
            if len(line) == 0:
                continue
            ps = self.LineP(line, xr)
            if len(ps) == 0:
                continue
            else:
                points += ps
            if norm(points[-1] - xr) < dmin:
                dmin = norm(points[-1] - xr)
                vnear = points[-1] - xr
        for arc in arcs:
            ps = self.ArcP(arc, xr)
            if len(ps) == 0:
                continue
            else:
                points += ps
            if norm(points[-1] - xr) < dmin:
                dmin = norm(points[-1] - xr)
                vnear = points[-1] - xr
        lvnear = norm(vnear)
        if lvnear > self.dmax:
            return vnear
        emax = 0
        v1 = array([100000, 100000])
        emin = 0
        v2 = array([100000, 100000])
        vs = array(points) - xr
        for v in vs:
            e = arctan2(cross(vnear, v), dot(vnear, v))
            if e >= emax:
                v1 = v
                emax = e
            if e <= emin:
                v2 = v
                emin = e
        return np.hstack([(vr + vo) / 2, v1, v2, vnear])
