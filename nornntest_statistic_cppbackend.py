import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
import os
import mujoco as mj
from numpy.linalg import norm
from numpy import array, arctan2, flipud, zeros
import numpy as np
from time import sleep, time
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
PARAMs["framerate"] = 25
PARAMs["max_ep_len"] = int(PARAMs["max_simu_second"] * PARAMs["framerate"])
PARAMs["hidden_sizes"] = [1024] * 4
# PARAMs["avevel"] = False
# PARAMs["nullfill"] = 20 * PARAMs["dmax"]
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

MODE, mode = 0, 0

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

###########################################################
# init environment get initial observation
# init model
modelfile, Nrobot, target, contour, ow, oh, w, h = EC.env_create4(MODE, mode)
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
env.stepVL([[0.0, 0.0]]*Nrobot, 1, 1)
env.cal_obs(PARAMs['avevel'])
env.cal_NNinput1(PARAMs["nullfill"])
o = torch.as_tensor(NNinput1, dtype=torch.float32, device=PARAMs["device"])
pos0 = np.zeros((Nrobot, 2), dtype=np.float64)
for i in range(Nrobot):
    pos0[i] = posvels[i][0:2]
die_mask = np.zeros(Nrobot, dtype=np.uint8)
len_count = np.zeros(Nrobot, dtype=np.uint64)
time_count = np.zeros(Nrobot, dtype=np.float64)
speed_sum = np.zeros(Nrobot, dtype=np.float64)
for i in range(Nrobot):
    if r[i] < -50:
        die_mask[i] = 1

N = int(1/PARAMs["framerate"]/0.002)
n = 5
if N % n != 0:
    n = 3
if N % n != 0:
    exit(-1)

ep_ret = 0
ep_len = 0
eps_count = 0
# Main loop: collect experience in env and update/log each epoch
for t in range(PARAMs["max_ep_len"] * (num_test_episodes + 1)):
    with torch.no_grad():
        a, logp = Pi(o, True, with_logprob=False)
    a = a.cpu().detach().numpy()
    for i in range(Nrobot):
        if d[i] == 1:
            a[i] = [0, 0]
    # Step the env
    onumpy = o.cpu().detach().numpy()
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
    for i in range(Nrobot):
        if r[i] > 50 and die_mask[i] == 0:
            len_count[i] = ep_len
        if die_mask[i] == 0:
            speed_sum[i] += np.sqrt(posvels[i][3]**2 + posvels[i][4]**2)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if (d == 1).all() or (ep_len == PARAMs["max_ep_len"]):
        for Nth in range(Nrobot):
            if len_count[Nth] == 0:
                len_count[Nth] = ep_len
                time_count[Nth] = env.get_time()
                die_mask[Nth] = 1
        print(
            f"\neps: {eps_count+1}, {Nrobot} robots, obs: {contour.__len__()}, mode: {MODE}_{mode}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, Nreach: {d.sum()}, Ndeath: {die_mask.sum()}")
        # print(die_mask)
        # print(len_count)
        if (1 - die_mask).sum() == 0:
            print("all died")
            Nrobot_log[eps_count] = Nrobot
            death_log[eps_count] = die_mask.sum()
        else:
            l_ave = (len_count * (1 - die_mask)).sum() / (1 - die_mask).sum()
            # print(l_ave)
            v_ave = ((speed_sum / len_count) * (1 - die_mask)
                     ).sum() / (1 - die_mask).sum()
            # print(v_ave)
            pos1 = np.zeros((Nrobot, 2), dtype=np.float64)
            for i in range(Nrobot):
                pos1[i] = posvels[i][0:2]
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
            Nrobot_log[eps_count] = Nrobot
            death_log[eps_count] = die_mask.sum()
            reach_log[eps_count] = d.sum()
            lave_log[eps_count] = l_ave
            vave_log[eps_count] = v_ave
            extra_log[eps_count] = m_ave / p_ave

        eps_count += 1
        if eps_count == num_test_episodes:
            break
        modelfile, Nrobot, target, contour, ow, oh, w, h = EC.env_create4(MODE, mode)
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
        env.stepVL([[0.0, 0.0]]*Nrobot, 1, 1)
        env.cal_obs(PARAMs['avevel'])
        env.cal_NNinput1(PARAMs["nullfill"])
        o = torch.as_tensor(NNinput1, dtype=torch.float32, device=PARAMs["device"])
        pos0 = np.zeros((Nrobot, 2), dtype=np.float64)
        for i in range(Nrobot):
            pos0[i] = posvels[i][0:2]
        die_mask = np.zeros(Nrobot, dtype=np.uint8)
        len_count = np.zeros(Nrobot, dtype=np.uint64)
        time_count = np.zeros(Nrobot, dtype=np.float64)
        speed_sum = np.zeros(Nrobot, dtype=np.float64)
        for i in range(Nrobot):
            if r[i] < -50:
                die_mask[i] = 1
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
