import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
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
# PARAMs["codec"] = 'hevc'
PARAMs["framerate"] = 10
PARAMs["max_ep_len"] = int(PARAMs["max_simu_second"] * PARAMs["framerate"])
PARAMs["hidden_sizes"] = [1024] * 4
# PARAMs["hidden_sizes"] = [2048] * 3
# PARAMs["target_bias"] = True
# PARAMs["act_limit"] = PARAMs["act_limit"] * 3
# model_file = "module_saves/nornn29/112h_23min_5639999steps_16625150updates_policy.ptd"
model_file = "module_saves/nornn31/305h_30min_7299999steps_32295763updates_policy.ptd"
vf_start = "module_saves/nornn31/"
PARAMs["avevel"] = False
PARAMs["nullfill"] = 20 * PARAMs["dmax"]
num_test_episodes = 100
MODEs = [6] * 5 + [3] * 7
modes = list(range(5)) + list(range(7))
# MODEs = [3]
# modes = [2]
MODE, mode = MODEs[0], modes[0]
eps_count = 18
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
PARAMs["seed"] = 667
torch.manual_seed(PARAMs["seed"])
np.random.seed(PARAMs["seed"])
random.seed(PARAMs["seed"])

torch.set_num_threads(torch.get_num_threads())

# config environment
CCcpp = CtrlConverter(vmax=PARAMs["vmax"], tau=PARAMs["tau"])
PARAMs["rmax"] = CCcpp.get_rmax()
SMLT = simulator_cpp.Simulator(
    dmax=PARAMs["dmax"], framerate=PARAMs["framerate"], dreach=PARAMs["dreach"], avevel=PARAMs["avevel"])
SMLT.set_reward(vmax=PARAMs["vmax"], rmax=PARAMs["rmax"], tolerance=PARAMs["tolerance"],
                a=PARAMs["a"], b=PARAMs["b"], c=PARAMs["c"], d=PARAMs["d"], e=PARAMs["e"],
                f=PARAMs["f"], g=PARAMs["g"], eta=PARAMs["eta"],
                h=PARAMs["h"], mu=PARAMs["mu"], rreach=PARAMs["rreach"],
                remix=PARAMs["remix"], rm_middle=PARAMs["rm_middle"], dmax=PARAMs["dmax"], w=PARAMs["w"])

font = ImageFont.truetype(
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 25)
# cofig SAC
Pi = nornnsac.nornncore.Policy(obs_dim=PARAMs["obs_dim"], act_dim=PARAMs["act_dim"],
                               act_limit=PARAMs["act_limit"], hidden_sizes=PARAMs["hidden_sizes"])
Pi.load_state_dict(torch.load(
    model_file, map_location=torch.device(PARAMs["device"])))
Pi.to(device=PARAMs["device"])
Pi.act_limit = Pi.act_limit.to(device=PARAMs["device"])

for p in Pi.parameters():
    p.requires_grad = False


def preNNinput(NNinput: tuple, obs_sur_dim: int, max_obs: int, device):
    # NNinput[0] Oself
    # NNinput[1] Osur
    Osur = np.ones((NNinput[0].__len__(), max_obs,
                    obs_sur_dim + 1), dtype=np.float32) * PARAMs["nullfill"]
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

    return torch.as_tensor(np.array([np.hstack([NNinput[0][Nth], Osur[Nth].flatten()]) for Nth in range(NNinput[0].__len__())]), dtype=torch.float32, device=device)


###########################################################
# init environment get initial observation
# init model
prefix = f"{MODE}_{mode}_"
SMLT.EC.gate_ratio = PARAMs["gate_ratio"]
Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy, w, h = SMLT.EC.env_create3(
    MODE=MODE, mode=mode)

actuator_text = SMLT.EC.actuator(Nrobot)
text = SMLT.EC.env_text(robot_text, obs_text, actuator_text, ow, oh, ch, fovy)
with open("./assets/test.xml", 'w') as fp:
    fp.write(text)

pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
    Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy)
o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
               PARAMs["max_obs"], PARAMs["device"])
ep_ret = 0
ep_len = 0

# config training process
# if PARAMs["isrender"]:
#     RD = render.Render(ow, oh)
#     RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
#     RD.switchCam()
#     videofp = videoIO.VideoIO(
#         vf_start + "assets/video.mp4", SMLT.framerate, w=ow, h=oh, codec=PARAMs["codec"], vf_start=vf_start)

# if PARAMs["isdraw"]:
#     CV = canvas.Canvas(w=w, h=h, dpi=100)
#     canvasfp = videoIO.VideoIO(
#         vf_start + "assets/video_canvas.mp4", SMLT.framerate, codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_start=vf_start)

if PARAMs["isrender"]:
    RD = render.Render(ow, oh)
    RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
    RD.switchCam()
    videofp = videoIO.VideoIO(
        "", SMLT.framerate, w=ow, h=oh, codec=PARAMs["codec"], vf_start=vf_start, prefix=prefix)

if PARAMs["isdraw"]:
    CV = canvas.Canvas(w=w, h=h, dpi=100)
    canvasfp = videoIO.VideoIO(
        "", SMLT.framerate, codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_end="draw", vf_start=vf_start, prefix=prefix)


eps_count_r = 0
# Main loop: collect experience in env and update/log each epoch
for t in range(PARAMs["max_ep_len"] * (num_test_episodes + 1)):
    with torch.no_grad():
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
    if (d == 1).all() or (ep_len == PARAMs["max_ep_len"]):
        print(
            f"eps: {eps_count+1}, {Nrobot} robots, mode: {MODE}_{mode}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, Nreach: {d.sum()}")
        # TODO change environment according to t
        eps_count -= 1
        eps_count_r += 1
        if eps_count_r < MODEs.__len__():
            MODE, mode = MODEs[eps_count_r], modes[eps_count_r]
        else:
            break
        prefix = f"{MODE}_{mode}_"
        Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy, w, h = SMLT.EC.env_create3(
            MODE=MODE, mode=mode)
        pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
            Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy)
        o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                       PARAMs["max_obs"], PARAMs["device"])
        ep_ret = 0
        ep_len = 0

        if PARAMs["isdraw"]:
            canvasfp.close()
        if PARAMs["isrender"]:
            videofp.close()

        if PARAMs["isrender"]:
            RD = render.Render(ow, oh)
            RD.set_model(SMLT.mjMODEL, SMLT.mjDATA)
            RD.switchCam()
            videofp = videoIO.VideoIO(
                "", SMLT.framerate, w=ow, h=oh, codec=PARAMs["codec"], vf_start=vf_start, prefix=prefix)

        if PARAMs["isdraw"]:
            CV = canvas.Canvas(w=w, h=h, dpi=100)
            canvasfp = videoIO.VideoIO(
                "", SMLT.framerate, codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_end="draw", vf_start=vf_start, prefix=prefix)
