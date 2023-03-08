import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
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

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

importlib.reload(nornnsac)
importlib.reload(envCreator)
importlib.reload(contourGenerator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(simulator_cpp)

torch.set_num_threads(torch.get_num_threads())

# config environment
isdraw = True
isrender = True
codec = 'h264'
framerate = 10
dreach = 0.02
rreach = 30
dmax = 3.0
vmax = 1.0
tau = 0.5
CCcpp = CtrlConverter(vmax=vmax, tau=tau)
rmax = CCcpp.get_rmax()
SMLT = simulator_cpp.Simulator(dmax=dmax, framerate=framerate, dreach=dreach)
SMLT.set_reward(vmax=vmax, rmax=rmax, tolerance=0.015,
                a=4.0, b=0.5, c=2, d=0.5, e=0.5, f=4, g=0.1, eta=0.125, h=0.15, mu=0.375)
font = ImageFont.truetype(
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 25)
# cofig SAC
device = "cuda"
act_dim = 2
max_obs = 16
obs_self_dim = 4
obs_sur_dim = 10
obs_dim = obs_self_dim + max_obs * (obs_sur_dim + 1)
act_limit = np.array([vmax, vmax], dtype=np.float32)
hidden_sizes = [1024, 1024, 1024]

Pi = nornnsac.nornncore.Policy(obs_dim, act_dim, act_limit, hidden_sizes)
Pi.load_state_dict(torch.load(
    "module_saves/nornn_6/0h_6min_5999steps_policy.ptd"))
Pi.to(device)
Pi.act_limit = Pi.act_limit.to(device)


def preNNinput(NNinput: tuple, obs_sur_dim: int, max_obs: int, device):
    # NNinput[0] Oself
    # NNinput[1] Osur
    Osur = np.ones((NNinput[0].__len__(), max_obs,
                    obs_sur_dim + 1), dtype=np.float32) * SMLT.dmax
    for Nth in range(NNinput[0].__len__()):
        total_len = NNinput[1][Nth].__len__()
        idxs = list(range(total_len))
        idxs.sort(key=lambda i: norm(NNinput[1][Nth][i][6:8]))
        for iobs in range(min(total_len, max_obs)):
            Osur[Nth][iobs] = [0] + NNinput[1][Nth][idxs[iobs]]

    return torch.as_tensor(np.array([np.hstack([NNinput[0][Nth], Osur[Nth].flatten()]) for Nth in range(NNinput[0].__len__())]), dtype=torch.float32, device=device)


###########################################################
# init environment get initial observation
# init model
Nrobot = 1
robot_text = SMLT.EC.circle_robot(Nrobot)
obs_text1, obs1 = SMLT.EC.circle_obstacle(3, 'l')
obs_text2, obs2 = SMLT.EC.circle_obstacle(1, 's')
pos_vel, observation, r, NNinput, d = SMLT.set_model(Nrobot, robot_text, obs_text1 +
                                                     obs_text2, obs1 + obs2, "circle")
o = preNNinput(NNinput, obs_sur_dim, max_obs, device)
ep_ret = 0
ep_len = 0

# config training process
max_simu_second = 30
max_ep_len = int(max_simu_second * framerate)
num_test_episodes = 10
total_steps = max_ep_len * num_test_episodes

if isrender:
    RD = render.Render()
    RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
    RD.switchCam()
    videofp = videoIO.VideoIO("assets/video.mp4", SMLT.framerate, codec=codec)

if isdraw:
    CV = canvas.Canvas(w=16, h=16)
    canvasfp = videoIO.VideoIO(
        "assets/video_canvas.mp4", SMLT.framerate, codec=codec, w=CV.w * CV.dpi, h=CV.h * CV.dpi)


# Main loop: collect experience in env and update/log each epoch
for t in range(total_steps):

    a, logp = Pi(o, True, with_logprob=False)
    a = a.cpu().detach().numpy()

    # Step the env
    aglobal = a
    onumpy = o.cpu().detach().numpy()
    for Nth in range(SMLT.Nrobot):
        aglobal[Nth] = np.matmul(
            np.array([[np.cos(pos_vel[Nth][2]), -np.sin(pos_vel[Nth][2])],
                      [np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
            aglobal[Nth]
        )
        # aglobal[Nth] = np.matmul(
        #     np.array([[np.cos(pos_vel[Nth][2]), -np.sin(pos_vel[Nth][2])],
        #               [np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
        #     onumpy[Nth][0:2] / norm(onumpy[Nth][0:2])
        # )
    ctrl = CCcpp.v2ctrlbatch(posvels=pos_vel, vs=aglobal)
    for Nth in range(SMLT.Nrobot):
        if d[Nth] == 1:
            ctrl[Nth * 2: Nth * 2 + 2] = [0, 0]
    dpre = d
    pos_vel, observation, r, NNinput, d = SMLT.step(ctrl)
    d = array([1 if dpre[i] == 1 or d[i] ==
              1 else 0 for i in range(d.shape[0])])
    r = array([r[rNth] + rreach if d[rNth] == 1 and dpre[rNth] ==
              0 else r[rNth] for rNth in range(r.__len__())], dtype=np.float32)
    o2 = preNNinput(NNinput, obs_sur_dim, max_obs, device)
    ep_ret += r.mean()
    ep_len += 1

    if isrender:
        RD.render()
        videofp.write_frame(flipud(RD.rgb))
    if isdraw:
        CV.newCanvas()
        CV.draw_contour(SMLT.contours)
        # print(o2)
        onumpy = o2.cpu().detach().numpy()
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
            CV.draw_line(pos_vel[i][0:2], np.matmul(
                tranM, oi[0:2]) + pos_vel[i][0:2])
            for o in range(observation[i].__len__()):
                o2draw = oi[5 + o * 11:5 + o * 11 + 8]
                o2draw[0:2] = np.matmul(tranM, o2draw[0:2])
                o2draw[2:4] = np.matmul(tranM, o2draw[2:4])
                o2draw[4:6] = np.matmul(tranM, o2draw[4:6])
                o2draw[6:8] = np.matmul(tranM, o2draw[6:8])
                CV.draw_rvop(o2draw, pos_vel[i][0:2])
        for pos in pos_vel:
            CV.draw_dmax(pos[0:2], SMLT.dmax)
            CV.draw_dmax(pos[0:2], 2 * SMLT.robot_r, 'black', 4)

        canvasfp.write_frame(array(CV.img))

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if (d == 1).all() or (ep_len == max_ep_len):
        print(
            f"t: {t}, {Nrobot} robots, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, Nreach: {d.sum()}")
        # TODO change environment according to t
        if t < max_ep_len * 1:
            Nrobot = 1
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(3, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(1, 's')
        elif t < max_ep_len * 4:
            Nrobot = 4
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(6, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(2, 's')
        elif t < max_ep_len * 8:
            Nrobot = 8
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(8, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(3, 's')
        else:
            Nrobot = 12
            robot_text = SMLT.EC.circle_robot(16)
            robot_text += SMLT.EC.circle_robot(6, 's', 16)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(8, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(3, 's')

        pos_vel, observation, r, NNinput, d = SMLT.set_model(Nrobot, robot_text, obs_text1 +
                                                             obs_text2, obs1 + obs2, "circle")
        o = preNNinput(NNinput, obs_sur_dim, max_obs, device)
        ep_ret = 0
        ep_len = 0

        if isdraw:
            canvasfp.close()
        if isrender:
            videofp.close()

        if isrender:
            RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
            RD.switchCam()
            videofp = videoIO.VideoIO("", SMLT.framerate, codec=codec)

        if isdraw:
            canvasfp = videoIO.VideoIO(
                "", SMLT.framerate, codec=codec, w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_end="draw")
