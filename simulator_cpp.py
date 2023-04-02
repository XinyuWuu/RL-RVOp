import importlib
import videoIO
import matplotlib.pyplot as plt
import canvas
from CppClass.RVOcalculator import RVOcalculator
from CppClass.Observator import Observator
import contourGenerator
import envCreator
import render
import ctrlConverter
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from numpy import array, arctan2, flipud, zeros
from numpy.linalg import norm
import mujoco as mj
import reward

importlib.reload(envCreator)
importlib.reload(contourGenerator)
# importlib.reload(RVOcalculator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(reward)
importlib.reload(ctrlConverter)


class Simulator():

    def __init__(self, robot_r=0.17, dmax=4.0, framerate=50, dreach=0.02):
        self.robot_r = robot_r
        self.dmax = dmax
        self.dreach = dreach
        self.EC = envCreator.EnvCreator()
        self.CG = contourGenerator.ContourGenrator(self.robot_r)
        self.OBS = Observator(self.dmax, self.robot_r)
        self.framerate = framerate
        self.step_time = 1 / framerate
        self.step_num = int(round(self.step_time / 0.002))
        self.step_time = self.step_num * 0.002
        self.set_reward()

    def set_reward(self, vmax=1.0, rmax=3.0, tolerance=0.005,
                   a=2.0, b=1.0, c=10.0,
                   d=1.0, e=1.0, f=20.0,
                   g=1.0, eta=0.5,
                   h=0.5, mu=0.75, rreach=30, remix=True, rm_middle=5):
        self.OBS.set_reward(robot_r=self.robot_r, vmax=vmax, rmax=rmax, tolerance=tolerance,
                            a=a, b=b, c=c, d=d, e=e, f=f, g=g, eta=eta, h=h, mu=mu, remix=remix, rm_middle=rm_middle)
        self.rreach = rreach

    def set_model(self, Nrobot=0, robot_text="", obs_text="", obs=[], target_type='circle'):
        self.Nrobot = Nrobot
        actuator_text = self.EC.actuator(self.Nrobot)
        text = self.EC.env_text(robot_text, obs_text, actuator_text)
        self.obs = obs
        self.contours = self.CG.generate_contour(obs)

        self.mjMODEL = mj.MjModel.from_xml_string(text)
        self.mjDATA = mj.MjData(self.mjMODEL)

        self.ctrl = self.mjDATA.ctrl
        self.qpos = [self.mjDATA.joint(
            f"robot_{id}").qpos for id in range(self.Nrobot)]
        self.qvel = [self.mjDATA.joint(
            f"robot_{id}").qvel for id in range(self.Nrobot)]

        if target_type == "random":
            pass
        elif target_type == "line":
            self.target = array(self.qpos)[:, 0:2] * array([-1.0, 1.0])
        else:  # "circle"
            self.target = -array(self.qpos)[:, 0:2]

        self.OBS.set_model(self.contours, self.target)
        self.d = np.zeros((self.Nrobot))
        self.dpre = self.d.copy()
        return self.step(np.zeros((2 * self.Nrobot,)))

    def step(self, ctrl: np.ndarray, observe=True):
        for Nth in range(self.Nrobot):
            if self.d[Nth] == 1:
                ctrl[Nth * 2: Nth * 2 + 2] = [0, 0]
        self.mjDATA.ctrl = ctrl
        mj.mj_step(self.mjMODEL, self.mjDATA, self.step_num)

        pos_vel = zeros((self.Nrobot, 6))
        observation = []
        for j in range(self.Nrobot):
            pos_vel[j] = array([self.qpos[j][0], self.qpos[j][1],
                                arctan2(
                                    2 * self.qpos[j][3] * self.qpos[j][6], 1 - 2 * self.qpos[j][6]**2),
                                self.qvel[j][0], self.qvel[j][1], self.qvel[j][5]], dtype=np.float32)
        if not observe:
            return array([]), [], array([]), [], array([]), array([])

        self.dpre = self.d.copy()
        self.d = array([1 if norm(self.target[Nth] - pos_vel[Nth][0:2]) <
                        self.dreach else 0
                        for Nth in range(self.Nrobot)])
        self.d = array([1 if self.dpre[i] == 1 or self.d[i] ==
                        1 else 0 for i in range(self.Nrobot)])

        observation, r, NNinput = self.OBS.get_obs(pos_vel)

        r = array([r[rNth] + self.rreach if self.d[rNth] == 1 and self.dpre[rNth] ==
                   0 else r[rNth] for rNth in range(self.Nrobot)], dtype=np.float32)
        r = array([0 if self.dpre[rNth] == 1 else r[rNth]
                  for rNth in range(self.Nrobot)], dtype=np.float32)

        return pos_vel, observation, r, NNinput, self.d, self.dpre
