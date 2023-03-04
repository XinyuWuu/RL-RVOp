from numpy import array, clip, dot, cross, sin, cos, arctan2
from numpy.linalg import norm
import numpy as np


class CtrlConverter():
    def __init__(self, vmax: float, tau: float, wheel_r: float = 0.04, wheel_d: float = 0.28) -> None:
        self.vmax = vmax
        self.tau = tau
        self.wr = wheel_r
        self.wd = wheel_d
        self.wmax = self.vmax / self.wr
        self.gain = 7  # convert omega to real ctrl, determined by mujoco

    def vw2ctrl(self, vl: float, w: float):
        return clip(array([vl - w / 2 / self.wd, vl + w / 2 / self.wd]) / self.wr, -self.wmax, self.wmax) * self.gain

    def v2ctrl(self, ori: float, v: np.ndarray):
        vl = dot(v, array([cos(ori), sin(ori)]))
        w = arctan2(cross(array([cos(ori), sin(ori)]),
                    v) / norm(v), vl / norm(v)) / self.tau
        return self.vw2ctrl(vl=vl, w=w)
