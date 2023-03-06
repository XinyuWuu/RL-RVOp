import itertools
import time
from copy import deepcopy
import numpy as np
import torch
from numpy.lib.npyio import save
from torch.optim import Adam
# import gym
import random
import importlib

import core
importlib.reload(core)


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

hidden_sizes = (256, 256, 256)
activation = torch.nn.ReLU
device = "cuda"
torch.set_num_threads(torch.get_num_threads())
steps_per_epoch = 6000
epochs = 1000
replay_size = int(1e6)
gamma = 0.99
polyak = 0.995
lr = 1e-3
alpha = 0.005
batch_size = 256
random_steps = 10000
update_after = 1000
update_every = 50
num_test_episodes = 1
max_ep_len = 3000


# env = gym.make(
#     'InvertedPendulum-v4',
#     new_step_api=True,
# )

# test_env = gym.make(
#     'InvertedPendulum-v4',
#     render_mode="human",
#     new_step_api=True,
# )

act_dim = 2
obs_dim = 10
obs_self_dim = 4
max_obs = 16
act_limit = np.array([1, 25])
rnn_state_size = 512
rnn_layer = 3
Pi = core.Policy(obs_dim, act_dim, act_limit, rnn_state_size, rnn_layer)
Qf1 = core.Qfunc(obs_dim, act_dim, rnn_state_size, rnn_layer)
Qf2 = core.Qfunc(obs_dim, act_dim, rnn_state_size, rnn_layer)
Qf1_targ = deepcopy(Qf1)
Qf2_targ = deepcopy(Qf2)
Pi.to(device)
Pi.act_limit = Pi.act_limit.to(device)
Qf1.to(device)
Qf2.to(device)
Qf1_targ.to(device)
Qf2_targ.to(device)

pi_optimizer = Adam(Pi.parameters(), lr=lr)
q_optimizer = Adam([{"params": Qf1.parameters()}, {
                   "params": Qf2.parameters()}], lr=lr)

log_alpha = torch.tensor(np.log(alpha), dtype=torch.float,
                         device=device, requires_grad=True)
alpha = np.exp(log_alpha.cpu().detach().numpy())
alpha_optim = torch.optim.Adam([log_alpha], lr=lr)

for p in Qf1_targ.parameters():
    p.requires_grad = False
for p in Qf2_targ.parameters():
    p.requires_grad = False


replay_buffer = core.ReplayBuffer(
    obs_dim=obs_dim, obs_self_dim=obs_self_dim, act_dim=act_dim, max_obs=max_obs, max_size=replay_size)


def compute_loss_q(data):
    q1 = Qf1(data["obs"], data['act'])
    q2 = Qf2(data["obs"], data['act'])
    with torch.no_grad():
        a2, logp_a2 = Pi(data["obs2"])

        q1_pi_targ = Qf1_targ(data["obs2"], a2)
        q2_pi_targ = Qf2_targ(data["obs2"], a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = data['rew'] + gamma * \
            (1 - data['done']) * (q_pi_targ - alpha * logp_a2)

    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    return loss_q, logp_a2


def compute_loss_pi(data):
    pi, logp_pi = Pi(data['obs'])
    q1_pi = Qf1(data['obs'], pi)
    q2_pi = Qf2(data['obs'], pi)
    q_pi = torch.min(q1_pi, q2_pi)
    loss_pi = (alpha * logp_pi - q_pi).mean()
    return loss_pi


def update(data):
    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, logp = compute_loss_q(data)
    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in Qf1.parameters():
        p.requires_grad = False
    for p in Qf2.parameters():
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi = compute_loss_pi(data)
    loss_pi.backward()
    pi_optimizer.step()

    # with torch.no_grad():
    #     a, logp = Pi(data['obs'])
    # print(logp.shape)
    loss_alpha = -(log_alpha.exp() * (logp.detach() - act_dim / 2)).mean()
    alpha_optim.zero_grad()
    loss_alpha.backward()
    alpha_optim.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in Qf1.parameters():
        p.requires_grad = True
    for p in Qf2.parameters():
        p.requires_grad = True

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(Qf1.parameters(), Qf1_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)
    with torch.no_grad():
        for p, p_targ in zip(Qf2.parameters(), Qf2_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)


def get_action(o, deterministic=False):
    with torch.no_grad():
        a, _ = Pi(torch.as_tensor(o, dtype=torch.float32).to(
            device), deterministic, False)
        return a.cpu().detach().numpy()


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


# Prepare for interaction with environment
total_steps = steps_per_epoch * epochs

# o, ep_ret, ep_len = env.reset(), 0, 0
###########################################################
# TODO init environment get initial observation
ep_ret = 0
ep_len = 0

start_time = time.time()
max_ret, max_ret_time, max_ret_rel_time =  \
    -1e6, time.time(), (time.time() - start_time) / 3600

##################### test#########################
# BUG
replay_buffer.size = 1000

# Main loop: collect experience in env and update/log each epoch
for t in range(total_steps):

    # if t > random_steps:
    #     a = get_action(o)
    # else:
    #     a = (np.random.rand(act_dim) * 2 - 1) * act_limit

    # Step the env
    # o2, r, d, truncated, info = env.step(a)
    # ep_ret += r
    # ep_len += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    # d = False if ep_len == max_ep_len else d

    # Store experience to replay buffer
    # replay_buffer.store(o, a, r, o2, d)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    # o = o2

    # End of trajectory handling
    # if d or (ep_len == max_ep_len):
    #     print(
    #         f"t: {t}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, last_r: {r:.2f}, alpha: {alpha:.4f}")
    #     o, ep_ret, ep_len = env.reset(), 0, 0

    # Update handling
    if t >= update_after and t % update_every == 0:
        for j in range(update_every):
            batch = replay_buffer.sample_batch(batch_size, device)
            # indexs = list(range(x_seqs.__len__()))
            # indexs = sorted(indexs, key=lambda x: x_seqs[x].size()[
            #                 0], reverse=True)
            # y = y[indexs]
            # x_pack = rnn_utils.pack_sequence([x_seqs[ind] for ind in indexs])
            update(data=batch)
            alpha = np.exp(log_alpha.cpu().detach().numpy())

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
        #     save_prefix = f"{time.ctime(max_ret_time)}_{max_ret_rel_time:.2f}h_{max_ret:.2f}_{ave_len:.0f}"
        # torch.save(ac.state_dict(),
        #            f'module_saves/rew/{save_prefix}_ac.ptd')
        # torch.save(ac_targ.state_dict(),
        #            f'module_saves/rew/{save_prefix}_ac_targ.ptd')


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
