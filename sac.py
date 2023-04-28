import itertools
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import importlib

import core;
importlib.reload(core)


class SAC():
    def __init__(self, obs_dim, act_dim, act_limit, rnn_state_size, rnn_layer, bidir, lr, gamma, polyak, alpha, device) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.rnn_state_size = rnn_state_size
        self.rnn_layer = rnn_layer
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.Pi = core.Policy(obs_dim, act_dim, act_limit,
                              rnn_state_size, rnn_layer, bidir)
        self.Qf1 = core.Qfunc(
            obs_dim, act_dim, rnn_state_size, rnn_layer, bidir)
        self.Qf2 = core.Qfunc(
            obs_dim, act_dim, rnn_state_size, rnn_layer, bidir)
        self.Qf1_targ = deepcopy(self.Qf1)
        self.Qf2_targ = deepcopy(self.Qf2)
        for p in self.Qf1_targ.parameters():
            p.requires_grad = False
        for p in self.Qf2_targ.parameters():
            p.requires_grad = False
        self.log_alpha = torch.tensor(
            np.log(alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = np.exp(self.log_alpha.cpu().detach().numpy())

        self.to(device)
        self.setOpti(lr)

    def to(self, device):
        self.Pi.to(device)
        self.Pi.act_limit = self.Pi.act_limit.to(device)
        self.Qf1.to(device)
        self.Qf2.to(device)
        self.Qf1_targ.to(device)
        self.Qf2_targ.to(device)

    def setOpti(self, lr):
        self.pi_optimizer = Adam(self.Pi.parameters(), lr=lr)
        self.q_optimizer = Adam([{"params": self. Qf1.parameters()}, {
            "params": self.Qf2.parameters()}], lr=lr)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    def compute_loss_q(self, data):
        q1 = self.Qf1(data["obs"], data['act'])
        q2 = self.Qf2(data["obs"], data['act'])
        with torch.no_grad():
            a2, logp_a2 = self.Pi(data["obs2"])
            q1_pi_targ = self.Qf1_targ(data["obs2"], a2)
            q2_pi_targ = self.Qf2_targ(data["obs2"], a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = data['rew'] + self.gamma * \
                (1 - data['done']) * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, logp_a2

    def compute_loss_pi(self, data):
        pi, logp_pi = self.Pi(data['obs'])
        q1_pi = self.Qf1(data['obs'], pi)
        q2_pi = self.Qf2(data['obs'], pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        return loss_pi

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, logp = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.Qf1.parameters():
            p.requires_grad = False
        for p in self.Qf2.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # with torch.no_grad():
        #     a, logp = Pi(data['obs'])
        # print(logp.shape)
        loss_alpha = -(self.log_alpha.exp() *
                       (logp.detach() - self.act_dim)).mean()
        self.alpha_optim.zero_grad()
        loss_alpha.backward()
        self.alpha_optim.step()
        self.alpha = np.exp(self.log_alpha.cpu().detach().numpy())

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.Qf1.parameters():
            p.requires_grad = True
        for p in self.Qf2.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.Qf1.parameters(), self.Qf1_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        with torch.no_grad():
            for p, p_targ in zip(self.Qf2.parameters(), self.Qf2_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_pi.cpu().detach().numpy(), loss_q.cpu().detach().numpy(), self.alpha
