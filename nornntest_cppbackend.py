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
import envCreator
import canvas
from PIL import ImageFont
import matplotlib.pyplot as plt
import videoIO
import importlib
from CppClass.Environment import Environment

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
importlib.reload(canvas)
importlib.reload(videoIO)
# PARAMs["codec"] = 'hevc'
PARAMs["framerate"] = 10
PARAMs["max_ep_len"] = int(PARAMs["max_simu_second"] * PARAMs["framerate"])
PARAMs["hidden_sizes"] = [1024] * 4
# PARAMs["hidden_sizes"] = [2048] * 3
# PARAMs["target_bias"] = True
# PARAMs["act_limit"] = PARAMs["act_limit"] * 3
# model_file = "module_saves/nornn29/112h_23min_5639999steps_16625150updates_policy.ptd"
model_file = "module_saves/nornn31/232h_54min_5719999steps_24636760updates_policy.ptd"
vf_start = "module_saves/nornn31/"
# PARAMs["avevel"] = False
# PARAMs["nullfill"] = 20 * PARAMs["dmax"]
num_test_episodes = 100
MODEs = [0]
modes = [0]
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
EC = envCreator.EnvCreator(PARAMs["robot_r"])
env = Environment()
PARAMs["rmax"] = env.setCtrl(vmax=PARAMs["vmax"], tau=PARAMs["tau"],
                             wheel_d=PARAMs["wheel_d"], wheel_r=PARAMs["wheel_r"],
                             gain=PARAMs["gain"])
env.setRvop(dmax=PARAMs["dmax"], robot_r=PARAMs["robot_r"])
env.setRwd(robot_r=PARAMs["robot_r"], vmax=PARAMs["vmax"], rmax=PARAMs["rmax"], tolerance=PARAMs["tolerance"], dreach=PARAMs["dreach"], tb=PARAMs["tb"],
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

###########################################################
# init environment get initial observation
# init model
prefix = f"{MODE}_{mode}_"
EC.gate_ratio = PARAMs["gate_ratio"]


modelfile, Nrobot, target, contour, ow, oh, w, h = EC.env_create4(MODE, mode)
# with open("./assets/test.xml", 'w') as fp:
#     fp.write(modelfile)
env.setSim(modelfile, Nrobot, target, contour, True, ow, oh)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((ow, oh, 3))
NNinput1 = np.frombuffer(
    env.get_NNinput1(), dtype=np.float64
).reshape(Nrobot, 180)
if not PARAMs["remix"]:
    r = np.frombuffer(env.get_r(), dtype=np.float64)
else:
    r = np.frombuffer(env.get_rm(), dtype=np.float64)
d = np.frombuffer(env.get_d(), dtype=np.int32)
env.stepVL([[0.0, 0.0]] * Nrobot, 1, 1)
env.cal_obs(PARAMs['avevel'])
env.cal_NNinput1(PARAMs["nullfill"])
o = torch.as_tensor(NNinput1, dtype=torch.float32, device=PARAMs["device"])

if PARAMs["isrender"]:
    videofp = videoIO.VideoIO(
        "", PARAMs["framerate"], w=ow, h=oh, codec=PARAMs["codec"], vf_start=vf_start, prefix=prefix)

if PARAMs["isdraw"]:
    CV = canvas.Canvas(w=w, h=h, dpi=100)
    canvasfp = videoIO.VideoIO(
        "", PARAMs["framerate"], codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_end="draw", vf_start=vf_start, prefix=prefix)

ep_ret = 0
ep_len = 0
eps_count_r = 0
N = int(1 / PARAMs["framerate"] / 0.002)
n = 5
if N % n != 0:
    n = 3
if N % n != 0:
    exit(-1)
# Main loop: collect experience in env and update/log each epoch
for t in range(PARAMs["max_ep_len"] * (num_test_episodes + 1)):
    with torch.no_grad():
        a, logp = Pi(o, True, with_logprob=False)
    a = a.cpu().detach().numpy()
    for i in range(Nrobot):
        if d[i] == 1:
            a[i] = [0, 0]

    # Step the env
    aglobal = a.copy()
    onumpy = o.cpu().detach().numpy()
    for Nth in range(Nrobot):
        aglobal[Nth] = np.matmul(
            np.array([[np.cos(posvels[Nth][2]), -np.sin(posvels[Nth][2])],
                      [np.sin(posvels[Nth][2]), np.cos(posvels[Nth][2])]]),
            aglobal[Nth]
        )
    if PARAMs["target_bias"]:
        for Nth in range(Nrobot):
            a[Nth] = a[Nth] + onumpy[Nth][0:2] / norm(onumpy[Nth][0:2])
    env.stepVL(a, N, n)
    env.cal_obs(PARAMs['avevel'])
    env.cal_NNinput1(PARAMs["nullfill"])
    env.cal_reward()
    o2 = torch.as_tensor(
        NNinput1, dtype=torch.float32, device=PARAMs["device"])

    ep_ret += r.mean()
    ep_len += 1

    if PARAMs["isrender"]:
        env.render()
        videofp.write_frame(flipud(rgb))
    if PARAMs["isdraw"]:
        CV.newCanvas()
        CV.draw_contour(contour)
        # print(o2)
        onumpy = o2.cpu().detach().numpy()
        for i in range(Nrobot):
            oi = onumpy[i]
            tranM = np.array([[np.cos(posvels[i][2]), -np.sin(posvels[i][2])],
                              [np.sin(posvels[i][2]), np.cos(posvels[i][2])]])
            # draw reward
            CV.draw_text(posvels[i][0:2], f"{r[i]:.2f}", font=font)
            # draw velocity
            CV.draw_line(posvels[i][0:2], posvels[i]
                         [0:2] + posvels[i][3:5], "purple", 2)
            # draw action
            CV.draw_line(posvels[i][0:2], posvels[i]
                         [0:2] + aglobal[i], "green", 2)
            # draw target
            CV.draw_line(posvels[i][0:2], np.matmul(
                tranM, oi[0:2]) + posvels[i][0:2])
            for o in range(PARAMs["max_obs"]):
                if abs(oi[4 + o * 11] - 0) > 1e-3:
                    break
                o2draw = oi[5 + o * 11:5 + o * 11 + 8]
                o2draw[0:2] = np.matmul(tranM, o2draw[0:2])
                o2draw[2:4] = np.matmul(tranM, o2draw[2:4])
                o2draw[4:6] = np.matmul(tranM, o2draw[4:6])
                o2draw[6:8] = np.matmul(tranM, o2draw[6:8])
                CV.draw_rvop(o2draw, posvels[i][0:2])
        for pos in posvels:
            CV.draw_dmax(pos[0:2], PARAMs["dmax"])
            CV.draw_dmax(pos[0:2], 2 * PARAMs["robot_r"], 'black', 4)
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
        modelfile, Nrobot, target, contour, ow, oh, w, h = EC.env_create4(
            MODE, mode)
        env.setSim(modelfile, Nrobot, target, contour, True, ow, oh)
        rgb = env.get_rgb()
        posvels = np.frombuffer(
            env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
        img_arr = np.frombuffer(
            rgb, dtype=np.uint8).reshape((ow, oh, 3))
        NNinput1 = np.frombuffer(
            env.get_NNinput1(), dtype=np.float64
        ).reshape(Nrobot, 180)
        if not PARAMs["remix"]:
            r = np.frombuffer(env.get_r(), dtype=np.float64)
        else:
            r = np.frombuffer(env.get_rm(), dtype=np.float64)
        d = np.frombuffer(env.get_d(), dtype=np.int32)
        env.stepVL([[0.0, 0.0]] * Nrobot, 1, 1)
        env.cal_obs(PARAMs['avevel'])
        env.cal_NNinput1(PARAMs["nullfill"])
        o = torch.as_tensor(NNinput1, dtype=torch.float32,
                            device=PARAMs["device"])

        ep_ret = 0
        ep_len = 0

        if PARAMs["isdraw"]:
            canvasfp.close()
        if PARAMs["isrender"]:
            videofp.close()

        if PARAMs["isrender"]:
            videofp = videoIO.VideoIO(
                "", PARAMs["framerate"], w=ow, h=oh, codec=PARAMs["codec"], vf_start=vf_start, prefix=prefix)

        if PARAMs["isdraw"]:
            CV = canvas.Canvas(w=w, h=h, dpi=100)
            canvasfp = videoIO.VideoIO(
                "", PARAMs["framerate"], codec=PARAMs["codec"], w=CV.w * CV.dpi, h=CV.h * CV.dpi, vf_end="draw", vf_start=vf_start, prefix=prefix)
