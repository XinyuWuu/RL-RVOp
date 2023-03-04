import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
import mujoco as mj
from numpy.linalg import norm
from numpy import array, arctan2, flipud, zeros
import numpy as np
from time import sleep, time
import simulator
import ctrlConverter
import reward
import render
import envCreator
import contourGenerator
import RVOcalculator
import canvas
import matplotlib.pyplot as plt
import videoIO
import importlib
import os


importlib.reload(envCreator)
importlib.reload(contourGenerator)
importlib.reload(RVOcalculator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(reward)
importlib.reload(ctrlConverter)
importlib.reload(simulator)

isdraw = False
isrender = False
codec = 'h264'

SMLT = simulator.Simulator(dmax=3.0)
Nrobot = 11
robot_text = SMLT.EC.circle_robot(Nrobot)
obs_text1, obs1 = SMLT.EC.circle_obstacle(9, 'l')
obs_text2, obs2 = SMLT.EC.circle_obstacle(5, 's')
pos_vel, observation = SMLT.set_model(Nrobot, robot_text, obs_text1 +
                                      obs_text2, obs1 + obs2, "circle")
if isrender:
    RD = render.Render()
    RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
    RD.switchCam()
    videofp = videoIO.VideoIO("assets/video.mp4", SMLT.framerate, codec=codec)

if isdraw:
    CV = canvas.Canvas(w=32, h=16)
    canvasfp = videoIO.VideoIO(
        "assets/video_canvas.mp4", SMLT.framerate, codec=codec)

RW = reward.Reward()
CC = ctrlConverter.CtrlConverter(vmax=1, tau=0.5)
start_t = time()

for stepi in range(1500):
    ctrl = np.zeros((2 * SMLT.Nrobot,))
    for Nth in range(Nrobot):
        ctrl[Nth * 2:Nth * 2 + 2] = CC.v2ctrl(pos_vel[Nth][2], v=(
            SMLT.target[Nth] - pos_vel[Nth][0:2]) / 2 / norm((SMLT.target[Nth] - pos_vel[Nth][0:2])))
    # ctrl = np.ones((2 * SMLT.Nrobot,)) * 25
    pos_vel, observation = SMLT.step(ctrl, True)
    # r = RW.reward(pos_vel, observation, SMLT.target)
    # print(r)
    if isrender:
        RD.render()
        videofp.write_frame(flipud(RD.rgb))
    if isdraw:
        CV.newCanvas()
        CV.draw_contour(SMLT.contours)
        for i in range(SMLT.Nrobot):
            for o in observation[i]:
                CV.draw_rvop(o['rvop'], pos_vel[i][0:2])
        for pos in pos_vel:
            CV.draw_dmax(pos[0:2], SMLT.dmax)

        canvasfp.write_frame(array(CV.img))

if isdraw:
    canvasfp.close()
if isrender:
    videofp.close()

end_t = time()
print((end_t - start_t) / SMLT.mjDATA.time)
