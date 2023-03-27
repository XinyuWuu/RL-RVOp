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

PARAMs = {
    "seed": 0,
    "isdraw": True,
    "isrender": True,
    "codec": 'h264',
    "framerate": 10,
    "dreach": 0.05,
    "tolerance": 0.02,
    "rreach": 30.0,
    "dmax": 3.0,
    "vmax": 1.0,
    "tau": 0.5,
    "device": "cuda",
    "gamma": 0.99,
    "polyak": 0.995,
    "lr": 5e-4,
    "alpha": 0.005,
    "act_dim": 2,
    "max_obs": 16,
    "obs_self_dim": 4,
    "obs_sur_dim": 10,
    "hidden_sizes": [1024, 1024, 1024],
    "replay_size": int(1e6),
}
PARAMs["obs_dim"] = PARAMs["obs_self_dim"] + \
    PARAMs["max_obs"] * (PARAMs["obs_sur_dim"] + 1)
PARAMs["act_limit"] = np.array(
    [PARAMs["vmax"], PARAMs["vmax"]], dtype=np.float32)


torch.manual_seed(PARAMs["seed"])
np.random.seed(PARAMs["seed"])
random.seed(PARAMs["seed"])

importlib.reload(nornnsac)
importlib.reload(envCreator)
importlib.reload(contourGenerator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(simulator_cpp)

torch.set_num_threads(torch.get_num_threads())

# config environment
CCcpp = CtrlConverter(vmax=PARAMs["vmax"], tau=PARAMs["tau"])
PARAMs["rmax"] = CCcpp.get_rmax()
SMLT = simulator_cpp.Simulator(
    dmax=PARAMs["dmax"], framerate=PARAMs["framerate"], dreach=PARAMs["dreach"])
# SMLT.set_reward(vmax=PARAMs["vmax"], rmax=PARAMs["rmax"], tolerance=0.015,
#                 a=4.0, b=0.5, c=2, d=0.5, e=0.5, f=4, g=0.1, eta=0.125, h=0.15, mu=0.375, rreach=PARAMs["rreach"])
SMLT.set_reward(vmax=PARAMs["vmax"], rmax=PARAMs["rmax"], tolerance=0.015,
                a=4.0)

font = ImageFont.truetype(
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 25)
# cofig SAC
Pi = nornnsac.nornncore.Policy(obs_dim=PARAMs["obs_dim"], act_dim=PARAMs["act_dim"],
                               act_limit=PARAMs["act_limit"], hidden_sizes=PARAMs["hidden_sizes"])
Pi.load_state_dict(torch.load(
    "module_saves/nornn4/70h_48min_2207999steps_policy.ptd", map_location=torch.device(PARAMs["device"])))
Pi.to(device=PARAMs["device"])
Pi.act_limit = Pi.act_limit.to(device=PARAMs["device"])


def preNNinput(NNinput: tuple, obs_sur_dim: int, max_obs: int, device):
    # NNinput[0] Oself
    # NNinput[1] Osur
    Osur = np.ones((NNinput[0].__len__(), max_obs,
                    obs_sur_dim + 1), dtype=np.float32) * 2 * SMLT.dmax
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
Nrobot = 4
robot_text = SMLT.EC.circle_robot(Nrobot)
obs_text1, obs1 = SMLT.EC.circle_obstacle(3, 'l')
obs_text2, obs2 = SMLT.EC.circle_obstacle(1, 's')
pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
    Nrobot, robot_text, obs_text1 + obs_text2, obs1 + obs2, "circle")
o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
               PARAMs["max_obs"], PARAMs["device"])
ep_ret = 0
ep_len = 0

# config training process
max_simu_second = 30
max_ep_len = int(max_simu_second * PARAMs["framerate"])
num_test_episodes = 10
total_steps = max_ep_len * num_test_episodes

if PARAMs["isrender"]:
    RD = render.Render()
    RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
    RD.switchCam()
    videofp = videoIO.VideoIO(
        "assets/video.mp4", SMLT.framerate, codec=PARAMs["codec"])

if PARAMs["isdraw"]:
    CV = canvas.Canvas(w=16, h=16)
    canvasfp = videoIO.VideoIO(
        "assets/video_canvas.mp4", SMLT.framerate, codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi)


# Main loop: collect experience in env and update/log each epoch
for t in range(total_steps):

    a, logp = Pi(o, True, with_logprob=False)
    a = a.cpu().detach().numpy()

    # Step the env
    aglobal = a.copy()
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
    pos_vel, observation, r, NNinput, d, dpre = SMLT.step(ctrl)
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
            # CV.draw_line(pos_vel[i][0:2], np.matmul(
            #     tranM, oi[0:2]) + pos_vel[i][0:2])
            for o in range(min(observation[i].__len__(), PARAMs["max_obs"])):
                o2draw = oi[5 + o * 11:5 + o * 11 + 8]
                o2draw[0:2] = np.matmul(tranM, o2draw[0:2])
                o2draw[2:4] = np.matmul(tranM, o2draw[2:4])
                o2draw[4:6] = np.matmul(tranM, o2draw[4:6])
                o2draw[6:8] = np.matmul(tranM, o2draw[6:8])
                CV.draw_rvop(o2draw, pos_vel[i][0:2])
        for pos in pos_vel:
            CV.draw_dmax(pos[0:2], SMLT.dmax)
            CV.draw_dmax(pos[0:2], 2 * SMLT.robot_r, 'black', 4)
            pass

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
            Nrobot = 4
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(3, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(1, 's')
        elif t < max_ep_len * 3:
            Nrobot = 4
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(6, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(2, 's')
        elif t < max_ep_len * 6:
            Nrobot = 8
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(8, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(3, 's')
        else:
            Nrobot = 12
            robot_text = SMLT.EC.circle_robot(8)
            robot_text += SMLT.EC.circle_robot(4, 's', 8)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(8, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(3, 's')

        pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(Nrobot, robot_text, obs_text1 +
                                                                   obs_text2, obs1 + obs2, "circle")
        o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                       PARAMs["max_obs"], PARAMs["device"])
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
                "", SMLT.framerate, codec=PARAMs["codec"])

        if PARAMs["isdraw"]:
            canvasfp = videoIO.VideoIO(
                "", SMLT.framerate, codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_end="draw")
