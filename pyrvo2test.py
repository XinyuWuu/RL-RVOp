import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
import pyrvo2.rvo2 as rvo2
import os
import mujoco as mj
from numpy.linalg import norm
from numpy import array, arctan2, flipud, zeros
import numpy as np
from time import sleep, time
# import simulator
import simulator_cpp
from CppClass.CtrlConverter import CtrlConverter
import render
import envCreator
import contourGenerator
import canvas
from PIL import ImageFont
import matplotlib.pyplot as plt
import videoIO
import importlib

import time
from copy import deepcopy
import numpy as np
import torch
from torch import tensor
from torch.optim import Adam
import random
import nornnsac
import importlib
from base_config import PARAMs
importlib.reload(nornnsac)
importlib.reload(envCreator)
importlib.reload(contourGenerator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(simulator_cpp)
PARAMs["framerate"] = 25
PARAMs["max_ep_len"] = int(PARAMs["max_simu_second"] * PARAMs["framerate"])
vf_start = "module_saves/pyrvo2/"
horizon_r = 80 / PARAMs['framerate']
horizon_o = 80 / PARAMs['framerate']
num_test_episodes = 15  # no meaning to set it bigger than 15
PARAMs["isrender"] = True
PARAMs["isdraw"] = True
PARAMs["a"] = 4
# PARAMs["tolerance"] = 0.08
PARAMs["b"] = 4
PARAMs["d"] = 4
PARAMs["g"] = 4
PARAMs["h"] = 4
PARAMs["eta"] = 1
PARAMs["mu"] = 1.5
PARAMs["remix"] = False
PARAMs["gate_ratio"] = 1 / 4
torch.manual_seed(PARAMs["seed"])
np.random.seed(PARAMs["seed"])
random.seed(PARAMs["seed"])

torch.set_num_threads(torch.get_num_threads())

# config environment
CCcpp = CtrlConverter(vmax=PARAMs["vmax"], tau=PARAMs["tau"])
PARAMs["rmax"] = CCcpp.get_rmax()
SMLT = simulator_cpp.Simulator(
    dmax=PARAMs["dmax"], framerate=PARAMs["framerate"], dreach=PARAMs["dreach"])
SMLT.set_reward(vmax=PARAMs["vmax"], rmax=PARAMs["rmax"], tolerance=PARAMs["tolerance"],
                a=PARAMs["a"], b=PARAMs["b"], c=PARAMs["c"], d=PARAMs["d"], e=PARAMs["e"],
                f=PARAMs["f"], g=PARAMs["g"], eta=PARAMs["eta"],
                h=PARAMs["h"], mu=PARAMs["mu"], rreach=PARAMs["rreach"],
                remix=PARAMs["remix"], rm_middle=PARAMs["rm_middle"], dmax=PARAMs["dmax"], w=PARAMs["w"])

font = ImageFont.truetype(
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 25)


def preNNinput(NNinput: tuple, obs_sur_dim: int, max_obs: int, device):
    # NNinput[0] Oself
    # NNinput[1] Osur
    Osur = np.ones((NNinput[0].__len__(), max_obs,
                    obs_sur_dim + 1), dtype=np.float32) * 2 * SMLT.dmax
    for Nth in range(NNinput[0].__len__()):
        true_len = min(NNinput[1][Nth].__len__(), max_obs)
        if true_len == 0:
            continue
        Osur[Nth][0:true_len] = np.hstack(
            [np.zeros((true_len, 1)), NNinput[1][Nth][0:true_len]])
        # total_len = NNinput[1][Nth].__len__()
        # idxs = list(range(total_len))
        # idxs.sort(key=lambda i: norm(NNinput[1][Nth][i][6:8]))
        # for iobs in range(min(total_len, max_obs)):
        #     Osur[Nth][iobs] = [0] + NNinput[1][Nth][iobs]

    return np.array([np.hstack([NNinput[0][Nth], Osur[Nth].flatten()]) for Nth in range(NNinput[0].__len__())])


###########################################################
# init environment get initial observation
# init model
MODE, mode = 0, 0
SMLT.EC.gate_ratio = PARAMs["gate_ratio"]
Nrobot, robot_text, obs_text, obs, target_mode = SMLT.EC.env_create(
    MODE=MODE, mode=mode)

# actuator_text = SMLT.EC.actuator(Nrobot)
# text = SMLT.EC.env_text(robot_text, obs_text, actuator_text)
# with open("./assets/test.xml", 'w') as fp:
#     fp.write(text)

pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
    Nrobot, robot_text, obs_text, obs, target_mode)
o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
               PARAMs["max_obs"], PARAMs["device"])
sim = rvo2.PyRVOSimulator(
    1 / PARAMs['framerate'], PARAMs['dmax'], PARAMs['max_obs'],
    horizon_r, horizon_o, 0.2, PARAMs['vmax'])
for Nth in range(SMLT.Nrobot):
    sim.addAgent(tuple(pos_vel[Nth][0:2]))
# TODO add obs
for obsi in obs:
    if obsi.shape[0] == 3:  # cylinder
        obsi_t = []
        for i in range(30):
            obsi_t.append((obsi[0] + obsi[2] * np.cos(np.deg2rad(i * 12)),
                           obsi[1] + obsi[2] * np.sin(np.deg2rad(i * 12))))
        obsi_t.sort(key=lambda x: np.arctan2(x[1] - obsi[1], x[0] - obsi[0]))
        sim.addObstacle(obsi_t)
    elif obsi.shape[0] == 4:  # line
        obsi_t = []
        obsi_t.append(obsi[0:2])
        obsi_t.append(obsi[2:4])
        sim.addObstacle(obsi_t)
    else:  # polygon
        obsi_t = []
        for i in range(int(obsi.shape[0] / 2 - 1)):
            obsi_t.append(obsi[i * 2 + 2:i * 2 + 4])
        obsi_t.sort(key=lambda x: np.arctan2(x[1] - obsi[1], x[0] - obsi[0]))
        # for i in range(obsi_t.__len__()):
        #     obsi_t[i] = obsi_t[i] + obsi[0:2]
        sim.addObstacle(obsi_t)
sim.processObstacles()
ep_ret = 0
ep_len = 0

# config training process
if PARAMs["isrender"]:
    RD = render.Render()
    RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
    RD.switchCam()
    videofp = videoIO.VideoIO(
        vf_start + "assets/video.mp4", SMLT.framerate, codec=PARAMs["codec"], vf_start=vf_start)

if PARAMs["isdraw"]:
    CV = canvas.Canvas(w=16, h=16)
    canvasfp = videoIO.VideoIO(
        vf_start + "assets/video_canvas.mp4", SMLT.framerate, codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_start=vf_start)

eps_count = 14
# Main loop: collect experience in env and update/log each epoch
for t in range(PARAMs["max_ep_len"] * (num_test_episodes + 1)):

    for Nth in range(SMLT.Nrobot):
        sim.setAgentPosition(Nth, tuple(pos_vel[Nth][0:2]))
        sim.setAgentVelocity(Nth, tuple(pos_vel[Nth][3:5]))
        sim.setAgentPrefVelocity(Nth, tuple(np.matmul(
            np.array([[np.cos(pos_vel[Nth][2]), -np.sin(pos_vel[Nth][2])],
                      [np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
            o[Nth][0:2]
        ).reshape(-1)))
        # sim.setAgentPrefVelocity(Nth, tuple(-pos_vel[Nth][0:2]))
    sim.doStep()
    a = np.zeros((SMLT.Nrobot, 2))
    for Nth in range(SMLT.Nrobot):
        a[Nth] = sim.getAgentVelocity(Nth)
    for Nth in range(SMLT.Nrobot):
        a[Nth] = np.matmul(
            np.array([[np.cos(pos_vel[Nth][2]), np.sin(pos_vel[Nth][2])],
                      [-np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
            a[Nth]
        )

    # Step the env
    aglobal = a.copy()
    onumpy = o.copy()
    for Nth in range(SMLT.Nrobot):
        aglobal[Nth] = np.matmul(
            np.array([[np.cos(pos_vel[Nth][2]), -np.sin(pos_vel[Nth][2])],
                      [np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
            aglobal[Nth]
        )
    # # aglobal[Nth] = np.matmul(
    #     np.array([[np.cos(pos_vel[Nth][2]), -np.sin(pos_vel[Nth][2])],
    #               [np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
    #     onumpy[Nth][0:2] / norm(onumpy[Nth][0:2])
    # )
    # ctrl = CCcpp.v2ctrlbatchG(posvels=pos_vel, vs=aglobal)
    # ctrl = CCcpp.v2ctrlbatchL(posvels=pos_vel, vs=a)
    # pos_vel, observation, r, NNinput, d, dpre = SMLT.step(ctrl)
    if PARAMs["target_bias"]:
        for Nth in range(SMLT.Nrobot):
            a[Nth] = a[Nth] + onumpy[Nth][0:2] / norm(onumpy[Nth][0:2])
    pos_vel, observation, r, NNinput, d, dpre = SMLT.step(
        a, isvs=True, CCcpp=CCcpp)
    o2 = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                    PARAMs["max_obs"], PARAMs["device"])
    ep_ret += r.mean()
    ep_len += 1

    if PARAMs["isrender"]:
        RD.render()
        videofp.write_frame(flipud(RD.rgb))
    if PARAMs["isdraw"]:
        CV.newCanvas()
        CV.draw_contour(SMLT.contours)
        # print(o2)
        onumpy = o2.copy()
        for i in range(SMLT.Nrobot):
            oi = onumpy[i]
            tranM = np.array([[np.cos(pos_vel[i][2]), -np.sin(pos_vel[i][2])],
                              [np.sin(pos_vel[i][2]), np.cos(pos_vel[i][2])]])
            # draw reward
            CV.draw_text(pos_vel[i][0:2], f"{r[i]:.2f}", font=font)
            # draw velocity
            CV.draw_line(pos_vel[i][0:2], pos_vel[i]
                         [0:2] + pos_vel[i][3:5], "purple", 2)
            # draw action
            CV.draw_line(pos_vel[i][0:2], pos_vel[i]
                         [0:2] + aglobal[i], "green", 2)
            # draw target
            # CV.draw_line(pos_vel[i][0:2], np.matmul(
            #     tranM, oi[0:2]) + pos_vel[i][0:2])
            # for o in range(min(observation[i].__len__(), PARAMs["max_obs"])):
            #     o2draw = oi[5 + o * 11:5 + o * 11 + 8]
            #     o2draw[0:2] = np.matmul(tranM, o2draw[0:2])
            #     o2draw[2:4] = np.matmul(tranM, o2draw[2:4])
            #     o2draw[4:6] = np.matmul(tranM, o2draw[4:6])
            #     o2draw[6:8] = np.matmul(tranM, o2draw[6:8])
            #     CV.draw_rvop(o2draw, pos_vel[i][0:2])
        for pos in pos_vel:
            CV.draw_dmax(pos[0:2], SMLT.dmax)
            CV.draw_dmax(pos[0:2], 2 * SMLT.robot_r, 'black', 4)
            pass

        canvasfp.write_frame(array(CV.img))

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if (d == 1).all() or (ep_len == PARAMs["max_ep_len"]):
        print(
            f"eps: {eps_count+1}, {Nrobot} robots, mode: {MODE}_{mode}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, Nreach: {d.sum()}")
        # TODO change environment according to t
        if eps_count > num_test_episodes - 1 or eps_count < -1:
            break
        if eps_count > 14:
            break
        if eps_count == 0:
            MODE, mode = 0, 0
        if eps_count == 1:
            MODE, mode = 0, 1
        if eps_count == 2:
            MODE, mode = 0, 2
        if eps_count == 3:
            MODE, mode = 0, 3
        if eps_count == 4:
            MODE, mode = 0, 4
            ###############################
        if eps_count == 5:
            MODE, mode = 1, 0
        if eps_count == 6:
            MODE, mode = 1, 1
        if eps_count == 7:
            MODE, mode = 1, 2
        if eps_count == 8:
            MODE, mode = 1, 3
        if eps_count == 9:
            MODE, mode = 1, 4
            ####################################
        if eps_count == 10:
            MODE, mode = 2, 0
        if eps_count == 11:
            MODE, mode = 2, 1
        if eps_count == 12:
            MODE, mode = 2, 2
        if eps_count == 13:
            MODE, mode = 2, 3
        if eps_count == 14:
            MODE, mode = 2, 4
        eps_count -= 1

        Nrobot, robot_text, obs_text, obs, target_mode = SMLT.EC.env_create2(
            MODE=MODE, mode=mode)
        pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
            Nrobot, robot_text, obs_text, obs, target_mode)
        o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                       PARAMs["max_obs"], PARAMs["device"])
        sim = rvo2.PyRVOSimulator(
            1 / PARAMs['framerate'], PARAMs['dmax'], PARAMs['max_obs'],
            horizon_r, horizon_o, 0.2, PARAMs['vmax'])
        for Nth in range(SMLT.Nrobot):
            sim.addAgent(tuple(pos_vel[Nth][0:2]))
        for obsi in obs:
            if obsi.shape[0] == 3:  # cylinder
                obsi_t = []
                for i in range(30):
                    obsi_t.append((obsi[0] + obsi[2] * np.cos(np.deg2rad(i * 12)),
                                obsi[1] + obsi[2] * np.sin(np.deg2rad(i * 12))))
                obsi_t.sort(key=lambda x: np.arctan2(x[1] - obsi[1], x[0] - obsi[0]))
                sim.addObstacle(obsi_t)
            elif obsi.shape[0] == 4:  # line
                obsi_t = []
                obsi_t.append(obsi[0:2])
                obsi_t.append(obsi[2:4])
                sim.addObstacle(obsi_t)
            else:  # polygon
                obsi_t = []
                for i in range(int(obsi.shape[0] / 2 - 1)):
                    obsi_t.append(obsi[i * 2 + 2:i * 2 + 4])
                obsi_t.sort(key=lambda x: np.arctan2(x[1] - obsi[1], x[0] - obsi[0]))
                # for i in range(obsi_t.__len__()):
                #     obsi_t[i] = obsi_t[i] + obsi[0:2]
                sim.addObstacle(obsi_t)
        sim.processObstacles()
        # TODO add obs
        ep_ret = 0
        ep_len = 0

        if PARAMs["isdraw"]:
            canvasfp.close()
        if PARAMs["isrender"]:
            videofp.close()

        if PARAMs["isrender"]:
            RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
            RD.switchCam()
            videofp = videoIO.VideoIO(
                "", SMLT.framerate, codec=PARAMs["codec"], vf_start=vf_start)

        if PARAMs["isdraw"]:
            canvasfp = videoIO.VideoIO(
                "", SMLT.framerate, codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_end="draw", vf_start=vf_start)
