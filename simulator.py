import importlib
import videoIO
import matplotlib.pyplot as plt
import canvas
import RVOcalculator
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
importlib.reload(RVOcalculator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(reward)
importlib.reload(ctrlConverter)


class Simulator():

    def __init__(self, robot_r=0.17, dmax=4.0, framerate=50):
        self.robot_r = robot_r
        self.dmax = dmax
        self.EC = envCreator.EnvCreator()
        self.CG = contourGenerator.ContourGenrator(self.robot_r)
        self.RVO = RVOcalculator.RVO(dmax)
        self.framerate = framerate
        self.step_time = 1 / framerate
        self.step_num = int(round(self.step_time / 0.002))
        self.step_time = self.step_num * 0.002

    def set_model(self, Nrobot=0, robot_text="", obs_text="", obs=[], target_type='circle'):
        self.Nrobot = Nrobot
        # robot_text = self.EC.gate_robot(self.Nrobot)
        # obs_text, obs = self.EC.line_obstacle(10)
        # obs_text, obs = self.EC.circle_obstacle(10)
        # obs_text, obs = self.EC.gate_obstacle(True)
        # obs_text, obs = self.EC.bigObs_obstacle(False, 1)
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
            self.target = array(self.qpos)[:, 0:2] * array([-1, 1])
        else:  # "circle"
            self.target = -array(self.qpos)[:, 0:2]
        return self.step(np.zeros((2 * self.Nrobot,)))

    def step(self, ctrl: np.ndarray, observe=True):
        self.mjDATA.ctrl = ctrl
        mj.mj_step(self.mjMODEL, self.mjDATA, self.step_num)

        pos_vel = zeros((self.Nrobot, 6))
        observation = []
        for j in range(self.Nrobot):
            pos_vel[j] = array([self.qpos[j][0], self.qpos[j][1],
                                arctan2(
                                    2 * self.qpos[j][3] * self.qpos[j][6], 1 - 2 * self.qpos[j][6]**2),
                                self.qvel[j][0], self.qvel[j][1], self.qvel[j][5]])
            observation.append([])
        if not observe:
            return pos_vel, observation
        for j in range(self.Nrobot):  # for jth robot
            for i in range(len(self.contours)):  # for obstacles
                observation[j].append({
                    'rvop': self.RVO.RVOplus(
                        self.contours[i][0], self.contours[i][1], pos_vel[j][0:2], pos_vel[j][3:5], np.zeros((2,))),
                    "pos": self.obs[i][0:2],
                    "vel": array([-1, -1]),
                    "target": array([-1, -1])})

        for j in range(self.Nrobot):  # for jth robot
            for i in range(self.Nrobot):  # for other robot
                if i == j:
                    continue
                con = self.CG.cylinder_contour(
                    np.array([pos_vel[i][0], pos_vel[i][1], self.robot_r]))
                observation[j].append({
                    'rvop': self.RVO.RVOplus(
                        con[0], con[1], pos_vel[j][0:2], pos_vel[j][3:5], pos_vel[i][3:5]),
                    'pos': pos_vel[i, 0:3],
                    'vel': pos_vel[i, 3:6],
                    'target': self.target[i]
                })

        return pos_vel, observation
