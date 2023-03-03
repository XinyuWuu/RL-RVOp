import numpy as np
from numpy import array, dot
from numpy.linalg import norm
import random

# contour
# type: line, n: (2,), vertex (4,); (6,0)
# type: arc, c: (2,), r: (1,), vertex (4,), m:(2,); (9,)


class ContourGenrator():
    def __init__(self, robot_r: float) -> None:
        self.robot_r = robot_r

    def line_contour(self, obs: np.ndarray):
        # 2 line
        # 2 arc
        lines = np.zeros((2, 6))
        arcs = np.zeros((2, 9))
        l = obs[2:4] - obs[0:2]
        l = l / norm(l)
        n = array([l[1], -l[0]])
        lines[0] = np.hstack(
            [-n, obs[0:2] + n * self.robot_r, obs[2:4] + n * self.robot_r])
        lines[1] = np.hstack(
            [n, obs[0:2] - n * self.robot_r, obs[2:4] - n * self.robot_r])
        arcs[0] = np.hstack([obs[0:2], self.robot_r,
                            obs[0:2] + n * self.robot_r,
                            obs[0:2] - n * self.robot_r,
                            obs[0:2] - l * self.robot_r,])
        arcs[1] = np.hstack([obs[2:4], self.robot_r,
                            obs[2:4] + n * self.robot_r,
                            obs[2:4] - n * self.robot_r,
                            obs[2:4] + l * self.robot_r])
        return lines, arcs

    def cylinder_contour(self, obs: np.ndarray):
        # 2 arc
        arcs = np.zeros((2, 9))
        arcs[0] = np.hstack([obs[0:2], (obs[2] + self.robot_r),
                            obs[0:2] + array([1, 0]) * (obs[2] + self.robot_r),
                            obs[0:2] + array([-1, 0]) *
                             (obs[2] + self.robot_r),
                            obs[0:2] + array([0, 1]) * (obs[2] + self.robot_r)])
        arcs[1] = np.hstack([obs[0:2], (obs[2] + self.robot_r),
                            obs[0:2] + array([1, 0]) * (obs[2] + self.robot_r),
                            obs[0:2] + array([-1, 0]) *
                             (obs[2] + self.robot_r),
                            obs[0:2] + array([0, -1]) * (obs[2] + self.robot_r)])
        return np.array([[1, 1, 1, 1, 1, 1]]) * 1000000, arcs

    def polygon_contour(self, obs: np.ndarray):
        # 3/4 line
        # 3/4 arc
        vex_num = int(obs.shape[0] / 2 - 1)
        lines = np.zeros((vex_num, 6))
        arcs = np.zeros((vex_num, 9))
        for i in range(vex_num):
            x0 = obs[i * 2:i * 2 +
                     2] if i != 0 else obs[vex_num * 2: vex_num * 2 + 2]
            x1 = obs[i * 2 + 2:i * 2 + 4]
            x2 = obs[i * 2 + 4:i * 2 + 6] if i != vex_num - 1 else obs[2:4]
            n1 = array([(x1 - x0)[1], -(x1 - x0)[0]])
            n1 = n1 / norm(n1)
            lines[i] = np.hstack(
                [-n1, x0 + n1 * self.robot_r, x1 + n1 * self.robot_r])
            n2 = array([(x2 - x1)[1], -(x2 - x1)[0]])
            n2 = n2 / norm(n2)
            arcs[i] = np.hstack([x1, self.robot_r, x1 + n1 * self.robot_r,
                                x1 + n2 * self.robot_r, x1 + (n1 + n2) / norm(n1 + n2) * self.robot_r])
        return lines, arcs

    def generate_contour(self, obs: list):
        contours = []
        for o in obs:
            if o.shape[0] == 4:
                contours.append(self.line_contour(o))
            elif o.shape[0] == 3:
                contours.append(self.cylinder_contour(o))
            else:
                contours.append(self.polygon_contour(o))
        return contours
