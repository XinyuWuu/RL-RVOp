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
import sac
import importlib

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

importlib.reload(sac)
importlib.reload(envCreator)
importlib.reload(contourGenerator)
importlib.reload(canvas)
importlib.reload(render)
importlib.reload(videoIO)
importlib.reload(simulator_cpp)

torch.set_num_threads(torch.get_num_threads())

# config environment
isdraw = False
isrender = False
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
SMLT.set_reward(vmax=vmax, rmax=rmax, a=4.0)

# cofig SAC
device = "cuda"
gamma = 0.99
polyak = 0.995
lr = 5e-4
alpha = 0.005
act_dim = 2
obs_dim = 14
act_limit = np.array([vmax, vmax], dtype=np.float32)
rnn_state_size = 512
rnn_layer = 1
bidir = True
SAC = sac.SAC(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, rnn_layer=rnn_layer, bidir=bidir,
              rnn_state_size=rnn_state_size, lr=lr, gamma=gamma, polyak=polyak, alpha=alpha, device=device)

# config replay buffer
replay_size = int(1e6)
max_obs = 17
replay_buffer = sac.core.ReplayBufferLite(
    obs_dim=obs_dim, act_dim=act_dim, max_obs=max_obs, max_size=replay_size)

# def get_action(o, deterministic=False):
#     with torch.no_grad():
#         a, _ = Pi(torch.as_tensor(o, dtype=torch.float32).to(
#             device), deterministic, False)
#         return a.cpu().detach().numpy()

# def test_agent():
#     total_ret, total_len, succ_rate = 0, 0, 0
#     for j in range(num_test_episodes):
#         o, d, ep_ret, ep_len, succ, = test_env.reset(), False, 0, 0, False
#         while not (d or (ep_len == max_ep_len)):
#             # Take deterministic actions at test time
#             a = get_action(o, True)
#             o, r, d, truncated, info = test_env.step(a)
#             ep_ret += r
#             ep_len += 1
#             total_ret += r
#             total_len += 1
#             if d == 1 and r > 0:
#                 succ = True
#                 succ_rate += 1
#         print(
#             f"test result: ret: {ep_ret:.2f}, len: {ep_len}, success: {succ}")

#     return total_ret / num_test_episodes, total_len / num_test_episodes, succ_rate / num_test_episodes


def preNNinput(NNinput: tuple, max_obs: int, device):
    # NNinput[0] Oself
    # NNinput[1] Osur
    for Nth in range(NNinput[0].__len__()):
        NNinput[1][Nth].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        total_len = NNinput[1][Nth].__len__()
        idxs = list(range(total_len))
        idxs.sort(key=lambda i: norm(NNinput[1][Nth][i][6:8]))
        NNinput[1][Nth] = tensor(NNinput[1][Nth], dtype=torch.float32)[
            idxs[0:min(total_len, max_obs)]]

    return [torch.hstack([torch.broadcast_to(tensor(NNinput[0][Nth], dtype=torch.float32), (NNinput[1][Nth].shape[0], NNinput[0][Nth].__len__())),
                          NNinput[1][Nth]]).to(device)
            for Nth in np.arange(NNinput[0].__len__())]


###########################################################
# init environment get initial observation
# init model
Nrobot = 4
robot_text = SMLT.EC.circle_robot(Nrobot)
obs_text1, obs1 = SMLT.EC.circle_obstacle(3, 'l')
obs_text2, obs2 = SMLT.EC.circle_obstacle(1, 's')
pos_vel, observation, r, NNinput, d = SMLT.set_model(Nrobot, robot_text, obs_text1 +
                                                     obs_text2, obs1 + obs2, "circle")
o = preNNinput(NNinput, max_obs, device)
ep_ret = 0
ep_len = 0

# config training process
max_simu_second = 30
max_ep_len = int(max_simu_second * framerate)
steps_per_epoch = 6000
epochs = 1000
batch_size = 512
random_steps = max_ep_len * 10
update_after = max_ep_len * 2
update_every = 50
num_test_episodes = 1


total_steps = steps_per_epoch * epochs
start_time = time.time()
max_ret, max_ret_time, max_ret_rel_time =  \
    -1e6, time.time(), (time.time() - start_time) / 3600
time_for_NN_update = 0
NN_update_count = 0
# Main loop: collect experience in env and update/log each epoch
for t in range(total_steps):

    if t > random_steps:
        # TODO only d == 0
        with torch.no_grad():
            a, logp = SAC.Pi(o, with_logprob=False)
            a = a.cpu().detach().numpy()
    else:
        a = (np.random.rand(Nrobot, act_dim) * 2 - 1) * act_limit

    # Step the env
    aglobal = a
    for Nth in range(SMLT.Nrobot):
        aglobal[Nth * 2: Nth * 2 + 2] = np.matmul(
            np.array([[np.cos(pos_vel[Nth][2]), -np.sin(pos_vel[Nth][2])],
                      [np.sin(pos_vel[Nth][2]), np.cos(pos_vel[Nth][2])]]),
            aglobal[Nth * 2: Nth * 2 + 2]
        )
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
    o2 = preNNinput(NNinput, max_obs, device)
    ep_ret += r.sum() / SMLT.Nrobot
    ep_len += 1

    # Store experience to replay buffer
    for Nth in np.arange(d.shape[0])[dpre == 0]:
        replay_buffer.store(o[Nth].cpu().detach().numpy(),
                            a[Nth], r[Nth], o2[Nth].cpu().detach().numpy(), d[Nth])

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if (d == 1).all() or (ep_len == max_ep_len):
        print(
            f"t: {t}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, alpha: {SAC.alpha:.4f}")
        # TODO change environment according to t
        if (t - random_steps) < max_ep_len * 20:
            Nrobot = 4
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(3, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(1, 's')
        elif (t - random_steps) < max_ep_len * 80:
            Nrobot = 8
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(6, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(2, 's')
        elif (t - random_steps) < max_ep_len * 160:
            Nrobot = 12
            robot_text = SMLT.EC.circle_robot(Nrobot)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(8, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(3, 's')
        else:
            Nrobot = 16 + 6
            robot_text = SMLT.EC.circle_robot(16)
            robot_text += SMLT.EC.circle_robot(6, 's', 16)
            obs_text1, obs1 = SMLT.EC.circle_obstacle(8, 'l')
            obs_text2, obs2 = SMLT.EC.circle_obstacle(3, 's')

        pos_vel, observation, r, NNinput, d = SMLT.set_model(Nrobot, robot_text, obs_text1 +
                                                             obs_text2, obs1 + obs2, "circle")
        o = preNNinput(NNinput, max_obs, device)
        ep_ret = 0
        ep_len = 0

    # Update handling
    if t >= update_after and t % update_every == 0:
        update_num = int(update_every * (Nrobot / 4))
        losspi_log = np.zeros(update_num)
        lossq_log = np.zeros(update_num)
        alpha_log = np.zeros(update_num)
        timebegin = time.time()
        for j in range(update_num):
            batch = replay_buffer.sample_batch(batch_size, device)
            losspi_log[j], lossq_log[j], alpha_log[j] = SAC.update(data=batch)
            # alpha = np.exp(log_alpha.cpu().detach().numpy())
        timeend = time.time()
        time_for_NN_update += timeend - timebegin

        print(
            f"update {NN_update_count}~{NN_update_count+update_num}; step {t}:\n\
            \tmean losspi: {losspi_log.mean():.4f}; mean lossq: {lossq_log.mean():.4f}; mean alpha: {alpha_log.mean():.4f}\n\
            \ttime for NN update / total time: {time_for_NN_update / (timeend - start_time)*100:.4f} %\n\
            \ttotal_time: {int((timeend-start_time)/3600)}h, {int((int(timeend-start_time)%3600)/60)}min; update per second {(NN_update_count+update_num)/(timeend - start_time):.4f}\n")
        NN_update_count += update_num
    # End of epoch handling
    if (t + 1) % steps_per_epoch == 0:
        epoch = (t + 1) // steps_per_epoch

        # Test the performance of the deterministic version of the agent.
        # print("!!!!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!")
        # ave_ret, ave_len, succ_rate = test_agent()
        # print(
        #     f"test result: ret: {ave_ret:.2f}, len: {ave_len:.2f}, time: {((time.time() - start_time) / 3600):.2f}, alpha: {alpha:.2f}")
        # if ave_ret > max_ret or succ_rate > 0.9:
        #     max_ret = ave_ret
        #     max_ret_time = time.time()
        #     max_ret_rel_time = (time.time() - start_time) / 3600
        save_prefix = f"{int((time.time()-start_time)/3600)}h_{int((int(time.time()-start_time)%3600)/60)}min_{t}steps"
        torch.save(SAC.Pi.state_dict(),
                   f'module_saves/{save_prefix}_policy.ptd')


# env=simulator.Simulator(2, show_figure=False),
# test_env=simulator.Simulator(2, show_figure=True),
# obs_dim=8,
# act_dim=2,
# act_limit=np.array([0.2, 3.63]),
# hidden_sizes=(256, 256, 256),
# activation=torch.nn.ReLU,
# device="cuda",
# seed=0,
# steps_per_epoch=6000,
# epochs=1000,
# replay_size=int(1e6),
# gamma=0.99,
# polyak=0.995,
# lr=1e-3,
# alpha=0.005,
# batch_size=256,
# random_steps=10000,
# update_after=1000,
# update_every=50,
# num_test_episodes=1,
# max_ep_len=3000

# a = torch.tensor(np.arange(100).reshape(-1, 2)).to('cuda')
# b = torch.tensor(np.array([2, 1])).to('cuda')
# a / b
# torch.broadcast_to(b, [a.shape[0], b.shape[0]])
# torch.hstack([a, torch.broadcast_to(b, [a.shape[0], b.shape[0]])])
