from numpy import zeros, array, dot, cross, sqrt, exp, sin, cos, arctan2
from numpy.linalg import norm
import numpy as np


class Reward():
    def __init__(self, robot_r=0.17, vmax=1, rmax=3, tolerance=0.005,
                 a=2, b=1, c=10,
                 d=1, e=1, f=20,
                 g=1, eta=0.5,
                 h=0.5, mu=0.75) -> None:
        self.robot_r = robot_r
        self.vmax = vmax
        self.rmax = rmax
        self.tolerance = tolerance
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.eta = eta
        self.h = h
        self.mu = mu

    def reward(self, pos_vel: np.ndarray, observation: list, target: np.ndarray):
        Nrobot = pos_vel.shape[0]
        r = zeros((Nrobot,))
        for Nth in range(Nrobot):
            r[Nth] = r[Nth] + \
                self.target(pos=pos_vel[Nth][0:2],
                            vel=pos_vel[Nth][3:5],
                            target=target[Nth])
            for o in observation[Nth]:
                if o[15] > 1000:
                    # static obstacle
                    # collision
                    r[Nth] = r[Nth] + \
                        self.collision_obs(
                            vel=pos_vel[Nth][3:5], vnear=o[6:8])
                    # near
                    r[Nth] = r[Nth] + \
                        self.near_obs(
                            vel=pos_vel[Nth][3:5], vnear=o[6:8])
                    pass
                else:
                    # other robot
                    # collision
                    r[Nth] = r[Nth] + self.collision_r(
                        pos=pos_vel[Nth][0:2], opos=o[8:10], vel=pos_vel[Nth][3:5], ovel=o[11:13], vnear=o[6:8])
                    # near
                    r[Nth] = r[Nth] + self.near_r(
                        pos=pos_vel[Nth][0:2], opos=o[8:10], vel=pos_vel[Nth][3:5], ovel=o[11:13], vnear=o[6:8])

                    pass
        return r

    def target(self, pos, vel, target):
        return self.a * dot(vel / self.vmax, (target - pos) / norm(target - pos))

    def collision_obs(self, vel, vnear):
        if norm(vnear) > self.tolerance:
            return 0

        return -self.b * dot(vel / self.vmax, vnear / norm(vnear)) - self.c

    def collision_r(self, pos, opos, vel, ovel, vnear):
        if norm(vnear) > self.tolerance:
            return 0
        return -self.d * dot((vel - ovel) / self.vmax, (opos - pos) / norm(opos - pos)) - self.f

    def near_obs(self, vel, vnear):
        return -self.g * (dot(vel / self.vmax, vnear / norm(vnear)) + 1) * exp(-norm(vnear) / self.eta / self.robot_r)

    def near_r(self, pos, opos, vel, ovel, vnear):
        return -self.h * (dot((vel - ovel) / self.vmax, (opos - pos) / norm(opos - pos)) + 2) * \
            exp(-norm(vnear) / self.mu / self.robot_r)


RW = Reward()
RW.target(pos=array([0, 0]), vel=array([-0.1, -0.1]), target=array([1, 1]))
RW.collision_obs(vel=array([0.1, -0.1]), vnear=array([0.001, 0.001]))
RW.collision_r(pos=array([0, 0]), opos=array([1, 0]), vel=array(
    [1, 1]), ovel=array([-1, 0]), vnear=array([0.0001, 0]))
RW.near_obs(vel=array([1, 0]), vnear=array([0.17, 0]))
RW.near_r(pos=array([0, 0]), opos=array([1, 0]), vel=array(
    [1, 0]), ovel=array([-1, 0]), vnear=array([0.17, 0]))
