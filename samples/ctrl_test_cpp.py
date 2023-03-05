import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
import mujoco as mj
from numpy.linalg import norm
from numpy import array, arctan2, flipud, zeros
from math import pi, remainder
import numpy as np
from time import sleep, time
# import simulator
import simulator_cpp
import ctrlConverter
from CppClass.CtrlConverter import CtrlConverter
import reward_cpp
import render
import envCreator
import contourGenerator
from CppClass.RVOcalculator import RVOcalculator
import canvas
import matplotlib.pyplot as plt
import videoIO
import importlib
import os

importlib.reload(envCreator)
importlib.reload(contourGenerator)
# importlib.reload(RVOcalculator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(reward_cpp)
importlib.reload(ctrlConverter)
importlib.reload(simulator_cpp)

isdraw = False
isrender = False
codec = 'h264'
framerate = 500

SMLT = simulator_cpp.Simulator(dmax=3.0, framerate=framerate)
Nrobot = 11
robot_text = SMLT.EC.circle_robot(Nrobot)
obs_text1, obs1 = SMLT.EC.circle_obstacle(9, 'l')
obs_text2, obs2 = SMLT.EC.circle_obstacle(5, 's')
pos_vel, observation, r, NNinput = SMLT.set_model(Nrobot, robot_text, obs_text1 +
                                                  obs_text2, obs1 + obs2, "circle")
RW = reward_cpp.Reward()
# SMLT.set_reward(tolerance=10)
# RW = reward_cpp.Reward(tolerance=10)
if isrender:
    RD = render.Render()
    RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
    RD.switchCam()
    videofp = videoIO.VideoIO("assets/video.mp4", SMLT.framerate, codec=codec)

if isdraw:
    CV = canvas.Canvas(w=32, h=16)
    canvasfp = videoIO.VideoIO(
        "assets/video_canvas.mp4", SMLT.framerate, codec=codec, w=CV.w * CV.dpi, h=CV.h * CV.dpi)


CC = ctrlConverter.CtrlConverter(vmax=1, tau=0.5)
CC.wmax
CCcpp = CtrlConverter(vmax=1.0, tau=0.5)
CCcpp.get_rmax()
total_frames = 1000
ctrl = np.zeros((2 * SMLT.Nrobot,))
ctrlcpp = np.zeros((2 * SMLT.Nrobot,))
vs = np.zeros((SMLT.Nrobot, 2))
pos_accumulation = pos_vel[:, 0:3]
start_t = time()
for stepi in range(total_frames):
    vsbatch = SMLT.target - pos_vel[:, 0:2]
    vsbatch = vsbatch / norm(vsbatch, axis=1, keepdims=True)
    ctrlcpp = CCcpp.v2ctrlbatch(pos_vel, vsbatch)
    pos_accumulation += pos_vel[:, 3:6] / SMLT.framerate / 2
    pos_vel, observation, r, NNinput = SMLT.step(ctrlcpp, True)
    pos_accumulation += pos_vel[:, 3:6] / SMLT.framerate / 2
    # mj.mj_step(SMLT.mjMODEL,SMLT.mjDATA,SMLT.step_num)
    # rpy = RW.reward(pos_vel, observation, SMLT.target)
    # print("________________________________")
    # print(array(r))
    # print(rpy)
    # print(((array(r) - rpy)**2).sum())
    # print(r)
    # print(observation)
    if isrender:
        RD.render()
        videofp.write_frame(flipud(RD.rgb))
    if isdraw:
        CV.newCanvas()
        CV.draw_contour(SMLT.contours)
        for i in range(SMLT.Nrobot):
            for o in observation[i]:
                # if o['rvop'][-1] > 1000:
                #     continue
                CV.draw_rvop(array(o[0:8]), pos_vel[i][0:2])
        for pos in pos_vel:
            CV.draw_dmax(pos[0:2], SMLT.dmax)
            CV.draw_dmax(pos[0:2], 2 * SMLT.robot_r, 'black', 4)

        canvasfp.write_frame(array(CV.img))

if isdraw:
    canvasfp.close()
if isrender:
    videofp.close()

end_t = time()
print((end_t - start_t) / SMLT.mjDATA.time)
print((end_t - start_t) / ((total_frames + 1) / 50))
# pos_accumulation[:,2] =remainder(pos_accumulation[:,2], 2 * pi)
for i in range(Nrobot):
    pos_accumulation[i][2] = remainder(pos_accumulation[i][2], 2 * pi)
i = 0
j = 0
pos_vel[i]
pos_accumulation[i]
SMLT.target[i]
NNinput[0][i]
observation[i][j]
NNinput[1][i][j]
