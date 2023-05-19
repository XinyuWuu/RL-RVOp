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
PARAMs["framerate"] = 25
PARAMs["max_ep_len"] = int(PARAMs["max_simu_second"] * PARAMs["framerate"])
PARAMs["hidden_sizes"] = [1024] * 4
# model_file = "module_saves/nornn29/78h_3min_3999999steps_11547900updates_policy.ptd"
# model_file = "module_saves/nornn29/112h_23min_5639999steps_16625150updates_policy.ptd"
model_file = "module_saves/nornn31/232h_54min_5719999steps_24636760updates_policy.ptd"
vf_start = "module_saves/nornn31/"
num_test_episodes = 100

Nrobot_log = np.zeros(num_test_episodes)
death_log = np.zeros(num_test_episodes)
reach_log = np.zeros(num_test_episodes)
lave_log = np.zeros(num_test_episodes)
tave_log = np.zeros(num_test_episodes)
vave_log = np.zeros(num_test_episodes)
extra_log = np.zeros(num_test_episodes)

MODE, mode = 6, 2

PARAMs["tolerance"] = 0.031
PARAMs["dreach"] = 0.075
PARAMs["c"] = 100
PARAMs["f"] = 100
PARAMs["rreach"] = 100
PARAMs["gate_ratio"] = 1 / 4
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

# cofig Network
Pi = nornnsac.nornncore.Policy(obs_dim=PARAMs["obs_dim"], act_dim=PARAMs["act_dim"],
                               act_limit=PARAMs["act_limit"], hidden_sizes=PARAMs["hidden_sizes"])
Pi.load_state_dict(torch.load(
    model_file, map_location=torch.device(PARAMs["device"])))
Pi.to(device=PARAMs["device"])
Pi.act_limit = Pi.act_limit.to(device=PARAMs["device"])

for p in Pi.parameters():
    p.requires_grad = False

Pi.eval()


def preNNinput(NNinput: tuple, obs_sur_dim: int, max_obs: int, device):
    # NNinput[0] Oself1.5
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

    return torch.as_tensor(np.array([np.hstack([NNinput[0][Nth], Osur[Nth].flatten()]) for Nth in range(NNinput[0].__len__())]), dtype=torch.float32, device=device)


###########################################################
# init environment get initial observation
# init model
SMLT.EC.gate_ratio = PARAMs["gate_ratio"]
Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy, w, h = SMLT.EC.env_create3(
    MODE=MODE, mode=mode)
pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
    Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy)
pos0 = np.zeros((Nrobot, 2), dtype=np.float64)
for i in range(Nrobot):
    pos0[i] = pos_vel[i][0:2]
die_mask = np.zeros(Nrobot, dtype=np.uint8)
len_count = np.zeros(Nrobot, dtype=np.uint64)
time_count = np.zeros(Nrobot, dtype=np.float64)
speed_sum = np.zeros(Nrobot, dtype=np.float64)
for i in range(Nrobot):
    if r[i] < -50:
        die_mask[i] = 1
o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
               PARAMs["max_obs"], PARAMs["device"])
ep_ret = 0
ep_len = 0

eps_count = 0
# Main loop: collect experience in env and update/log each epoch
for t in range(PARAMs["max_ep_len"] * (num_test_episodes + 1)):
    with torch.no_grad():
        a, logp = Pi(o, True, with_logprob=False)
    a = a.cpu().detach().numpy()

    # Step the env
    onumpy = o.cpu().detach().numpy()
    if PARAMs["target_bias"]:
        for Nth in range(SMLT.Nrobot):
            a[Nth] = a[Nth] + onumpy[Nth][0:2] / norm(onumpy[Nth][0:2])
    pos_vel, observation, r, NNinput, d, dpre = SMLT.step(
        a, isvs=True, CCcpp=CCcpp)
    for i in range(Nrobot):
        if r[i] < -50:
            die_mask[i] = 1
    o2 = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                    PARAMs["max_obs"], PARAMs["device"])
    ep_ret += r.mean()
    ep_len += 1
    for i in range(Nrobot):
        if r[i] > 50 and die_mask[i] == 0:
            len_count[i] = ep_len
        if die_mask[i] == 0:
            speed_sum[i] += np.sqrt(pos_vel[i][3]**2 + pos_vel[i][4]**2)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if (d == 1).all() or (ep_len == PARAMs["max_ep_len"]):
        for Nth in range(SMLT.Nrobot):
            if len_count[Nth] == 0:
                len_count[Nth] = ep_len
                time_count[Nth] = SMLT.mjDATA.time
                die_mask[Nth] = 1
        print(
            f"\neps: {eps_count+1}, {Nrobot} robots, obs: {obs.__len__()}, mode: {MODE}_{mode}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, Nreach: {d.sum()}, Ndeath: {die_mask.sum()}")
        # print(die_mask)
        # print(len_count)
        if (1 - die_mask).sum() == 0:
            print("all died")
            Nrobot_log[eps_count] = SMLT.Nrobot
            death_log[eps_count] = die_mask.sum()
        else:
            l_ave = (len_count * (1 - die_mask)).sum() / (1 - die_mask).sum()
            # print(l_ave)
            v_ave = ((speed_sum / len_count) * (1 - die_mask)
                     ).sum() / (1 - die_mask).sum()
            # print(v_ave)
            pos1 = np.zeros((Nrobot, 2), dtype=np.float64)
            for i in range(Nrobot):
                pos1[i] = pos_vel[i][0:2]
            posc = pos1 - pos0
            posc = np.sqrt(posc[:, 0]**2 + posc[:, 1]**2).reshape((Nrobot))
            p_ave = (posc * (1 - die_mask)).sum() / (1 - die_mask).sum()
            # print(p_ave)
            # m_ave = l_ave / PARAMs["framerate"] * v_ave
            m_ave = (len_count * (1 - die_mask) * ((speed_sum / len_count))
                     ).sum() / (1 - die_mask).sum() / PARAMs["framerate"]
            # print(m_ave)
            # print(m_ave / p_ave * 100)
            print(f"die_mask:{die_mask};len_count:{len_count}")
            print(
                f"l_ave {l_ave}; v_ave {v_ave}; p_ave {p_ave}; m_ave {m_ave}; m/p {m_ave / p_ave * 100}%")
            Nrobot_log[eps_count] = SMLT.Nrobot
            death_log[eps_count] = die_mask.sum()
            reach_log[eps_count] = d.sum()
            lave_log[eps_count] = l_ave
            vave_log[eps_count] = v_ave
            extra_log[eps_count] = m_ave / p_ave

        eps_count += 1
        if eps_count == num_test_episodes:
            break
        Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy, w, h = SMLT.EC.env_create3(
            MODE=MODE, mode=mode)
        pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
            Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy)
        pos0 = np.zeros((Nrobot, 2), dtype=np.float64)
        for i in range(Nrobot):
            pos0[i] = pos_vel[i][0:2]
        die_mask = np.zeros(Nrobot, dtype=np.uint8)
        len_count = np.zeros(Nrobot, dtype=np.uint64)
        speed_sum = np.zeros(Nrobot, dtype=np.float64)
        for i in range(Nrobot):
            if r[i] < -50:
                die_mask[i] = 1
        o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                       PARAMs["max_obs"], PARAMs["device"])
        ep_ret = 0
        ep_len = 0

print("##########################Over All Performance#####################################")
print(f"sucess rate: {(Nrobot_log - death_log).sum() / Nrobot_log.sum()}")
print(f"reach rate: {reach_log.sum() / Nrobot_log.sum()}")
print(
    f"average velocity: {(vave_log * (Nrobot_log - death_log)).sum() / (Nrobot_log - death_log).sum()}")
print(
    f"average time: {(lave_log * (Nrobot_log - death_log)).sum() / (Nrobot_log - death_log).sum()*(1/PARAMs['framerate'])}")
print(
    f"extra length: {(extra_log * (Nrobot_log - death_log)).sum() / (Nrobot_log - death_log).sum()}")
