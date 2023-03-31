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

# seed = 0
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
                h=PARAMs["h"], mu=PARAMs["mu"], rreach=PARAMs["rreach"])

# cofig SAC
SAC = nornnsac.SAC(obs_dim=PARAMs["obs_dim"], act_dim=PARAMs["act_dim"], act_limit=PARAMs["act_limit"],
                   hidden_sizes=PARAMs["hidden_sizes"], lr=PARAMs["lr"], gamma=PARAMs["gamma"], polyak=PARAMs["polyak"], alpha=PARAMs["alpha"], device=PARAMs["device"])

# config replay buffer
replay_buffer = nornnsac.nornncore.ReplayBufferLite(
    obs_dim=PARAMs["obs_dim"], act_dim=PARAMs["act_dim"], max_size=PARAMs["replay_size"])


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
MODE, mode = 0, 0
Nrobot, robot_text, obs_text, obs, target_mode = SMLT.EC.env_create(
    MODE=MODE, mode=mode)
pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
    Nrobot, robot_text, obs_text, obs, target_mode)
o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
               PARAMs["max_obs"], PARAMs["device"])
ep_ret = 0
ep_len = 0

# config training process
start_time = time.time()
time_for_NN_update = 0
time_for_step = 0
NN_update_count = 0
max_ret=0
# Main loop: collect experience in env and update/log each epoch
for t in range(PARAMs["total_steps"]):

    if t > PARAMs["random_steps"]:
        with torch.no_grad():
            a, logp = SAC.Pi(o, with_logprob=False)
            a = a.cpu().detach().numpy()
    else:
        a = (np.random.rand(
            Nrobot, PARAMs["act_dim"]) * 2 - 1) * PARAMs["act_limit"]

    # Step the env
    timebegin = time.time()
    aglobal = a.copy()
    for Nth in range(SMLT.Nrobot):
        aglobal[Nth] = np.matmul(
            np.array([[np.cos(pos_vel[Nth][2]), -np.sin(pos_vel[Nth][2])],
                      [np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
            aglobal[Nth]
        )
    ctrl = CCcpp.v2ctrlbatch(posvels=pos_vel, vs=aglobal)
    pos_vel, observation, r, NNinput, d, dpre = SMLT.step(ctrl)
    o2 = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                    PARAMs["max_obs"], PARAMs["device"])
    ep_ret += r.mean()
    ep_len += 1
    time_for_step += time.time() - timebegin

    # Store experience to replay buffer
    for Nth in np.arange(d.shape[0])[dpre == 0]:
        replay_buffer.store(o[Nth].cpu().detach().numpy(),
                            a[Nth], r[Nth], o2[Nth].cpu().detach().numpy(), d[Nth])

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if (d == 1).all() or (ep_len == PARAMs["max_ep_len"]):
        print(
            f"t: {t}, {Nrobot} robots, mode: {MODE}_{mode}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, Nreach: {d.sum()}, alpha: {SAC.alpha:.4f}")
        # save model
        if (ep_ret * Nrobot > max_ret):
            max_ret = ep_ret * Nrobot
            time_prefix = f"{int((time.time()-start_time)/3600)}h_{int((int(time.time()-start_time)%3600)/60)}min_{t}steps_{NN_update_count}updates"
            torch.save(SAC.Pi.state_dict(),
                       f'module_saves/max_{ep_ret:.2f}_{Nrobot}robots_{time_prefix}_policy.ptd')

        random_num = np.random.rand()
        if (t - PARAMs["random_steps"]) < PARAMs["max_ep_len"] * 500:
            MODE = 0
            if random_num < 0.3:
                mode = 0
            elif random_num < 0.6:
                mode = 1
            elif random_num < 0.8:
                mode = 2
            elif random_num < 0.9:
                mode = 3
            else:
                mode = 4
        elif (t - PARAMs["random_steps"]) < PARAMs["max_ep_len"] * 1500:
            MODE = 1
            if random_num < 0.2:
                mode = 0
            elif random_num < 0.4:
                mode = 1
            elif random_num < 0.6:
                mode = 2
            elif random_num < 0.8:
                mode = 3
            else:
                mode = 4
        else:
            MODE = 2
            if random_num < 0.2:
                mode = 0
            elif random_num < 0.4:
                mode = 1
            elif random_num < 0.6:
                mode = 2
            elif random_num < 0.8:
                mode = 3
            else:
                mode = 4

        Nrobot, robot_text, obs_text, obs, target_mode = SMLT.EC.env_create(
            MODE=MODE, mode=mode)
        pos_vel, observation, r, NNinput, d, dpre = SMLT.set_model(
            Nrobot, robot_text, obs_text, obs, target_mode)
        o = preNNinput(NNinput, PARAMs["obs_sur_dim"],
                       PARAMs["max_obs"], PARAMs["device"])
        ep_ret = 0
        ep_len = 0

    # Update handling
    if t >= PARAMs["update_after"] and t % PARAMs["update_every"] == 0:
        timebegin = time.time()
        update_num = int(PARAMs["update_every"] * (Nrobot / 4))
        losspi_log = np.zeros(update_num)
        lossq_log = np.zeros(update_num)
        alpha_log = np.zeros(update_num)

        for j in range(update_num):
            batch = replay_buffer.sample_batch(
                PARAMs["batch_size"], PARAMs["device"])
            losspi_log[j], lossq_log[j], alpha_log[j] = SAC.update(data=batch)

        timeend = time.time()
        time_for_NN_update += timeend - timebegin
        print(
            f"update {NN_update_count}~{NN_update_count+update_num}; step {t}; {Nrobot} robots:\n\
            \tmean losspi: {losspi_log.mean():.4f}; mean lossq: {lossq_log.mean():.4f}; mean alpha: {alpha_log.mean():.4f}\n\
            \ttime for NN update / total time: {time_for_NN_update / (timeend - start_time)*100:.4f} %\n\
            \ttime for step / total time: {time_for_step / (timeend - start_time)*100:.4f} %\n\
            \tstep per second {(t+1)/time_for_step:.4f},update per second {(NN_update_count+update_num)/time_for_NN_update:.4f}\n\
            \ttotal_time: {int((timeend-start_time)/3600)}h, {int((int(timeend-start_time)%3600)/60)}min;")
        NN_update_count += update_num

    # End of epoch handling
    if (t + 1) % PARAMs["steps_per_epoch"] == 0:
        epoch = (t + 1) // PARAMs["steps_per_epoch"]
        save_prefix = f"{int((time.time()-start_time)/3600)}h_{int((int(time.time()-start_time)%3600)/60)}min_{t}steps_{NN_update_count}updates"
        torch.save(SAC.Pi.state_dict(),
                   f'module_saves/{save_prefix}_policy.ptd')
