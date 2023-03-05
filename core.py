import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.utils.rnn as rnn_utils


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Policy(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, rnn_state_size, rnn_layer):
        super().__init__()
        self.net0 = nn.Sequential(
            nn.Linear(obs_dim, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=rnn_state_size, hidden_size=rnn_state_size,
                          num_layers=rnn_layer, bidirectional=True, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(2 * rnn_state_size, 2 * rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(rnn_state_size, act_dim)
        self.log_std_layer = nn.Linear(rnn_state_size, act_dim)
        self.act_limit = torch.tensor(act_limit)
        self.act_limit_prod_log = np.log(self.act_limit.prod())

    def forward(self, obs, deterministic=False, with_logprob=True):
        x0 = self.net0(obs.data)
        x0 = rnn_utils.PackedSequence(
            data=x0, batch_sizes=obs.batch_sizes)
        h, (ht, ct) = self.rnn(x0)
        y = self.net(torch.concat([ht[-1], ht[-2]], dim=1))
        net_out = self.net(y)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action -
                        F.softplus(-2 * pi_action))).sum(axis=1) + self.act_limit_prod_log

        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action) * self.act_limit

        return pi_action, logp_pi


class Qfunc(nn.Module):

    def __init__(self, obs_dim, act_dim, rnn_state_size, rnn_layer):
        super().__init__()
        self.net0 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=rnn_state_size, hidden_size=rnn_state_size,
                          num_layers=rnn_layer, bidirectional=True, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(2 * rnn_state_size, 2 * rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, 1),
        )

    def forward(self, x):
        x0 = self.net0(x.data)
        x0 = rnn_utils.PackedSequence(
            data=x0, batch_sizes=x.batch_sizes)
        h, (ht, ct) = self.rnn(x0)
        y = self.net(torch.concat([ht[-1], ht[-2]], dim=1))
        q = self.net(y)
        # Critical to ensure q has right shape.
        return torch.squeeze(q, -1)


class ReplayBuffer:
    def __init__(self, obs_dim, obs_self_dim, act_dim, max_obs, max_size):
        self.obs_self_buf = np.zeros(
            (max_size, obs_dim), dtype=np.float32)
        self.obs_buf = np.zeros(
            (max_size * max_obs, obs_dim), dtype=np.float32)
        self.obs2_self_buf = np.zeros(
            (max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(
            (max_size * max_obs, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.obsbegin_buf = np.zeros(max_size, dtype=np.int64)
        self.obs2begin_buf = np.zeros(max_size, dtype=np.int64)
        self.obsend_buf = np.zeros(max_size, dtype=np.int64)
        self.obs2end_buf = np.zeros(max_size, dtype=np.int64)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size, self.max_obs, self.max_size = 0, 0, max_obs, max_size - 1

    def store(self, obs, obs_self, act, rew, next_obs, next_obs_self, done):
        self.obsend_buf[self.ptr] = obs.shape[0] + \
            self.obsbegin_buf[self.ptr]
        self.obs2end_buf[self.ptr] = next_obs.shape[0] + \
            self.obs2begin_buf[self.ptr]

        self.obsbegin_buf[self.ptr + 1] = self.obsend_buf[self.ptr]
        self.obs2begin_buf[self.ptr + 1] = self.obs2end_buf[self.ptr]

        self.obs_self_buf[self.ptr] = obs_self
        self.obs_buf[self.obsbegin_buf[self.ptr]                     :self.obsend_buf[self.ptr]] = obs

        self.obs2_self_buf[self.ptr] = next_obs_self
        self.obs2_buf[self.obs2begin_buf[self.ptr]                      :self.obs2end_buf[self.ptr]] = next_obs

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs_sur=[torch.as_tensor(
                self.obs_buf[self.obsbegin_buf[idx]                             :self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            obs2_sur=[torch.as_tensor(
                self.obs2_buf[self.obs2begin_buf[idx]                              :self.obs2end_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            obs_self=torch.as_tensor(
                self.obs_self_buf[idxs], dtype=torch.float32).to(device),
            obs2_self=torch.as_tensor(
                self.obs2_self_buf[idxs], dtype=torch.float32).to(device),
            act=torch.as_tensor(
                self.act_buf[idxs], dtype=torch.float32).to(device),
            rew=torch.as_tensor(
                self.rew_buf[idxs], dtype=torch.float32).to(device),
            done=torch.as_tensor(
                self.done_buf[idxs], dtype=torch.float32).to(device))

        # return dict(
        #     obs=torch.as_tensor(
        #         self.obs_buf[idxs], dtype=torch.float32).to(device),
        #     obs2=torch.as_tensor(
        #         self.obs2_buf[idxs], dtype=torch.float32).to(device),
        #     act=torch.as_tensor(
        #         self.act_buf[idxs], dtype=torch.float32).to(device),
        #     rew=torch.as_tensor(
        #         self.rew_buf[idxs], dtype=torch.float32).to(device),
        #     done=torch.as_tensor(
        #         self.done_buf[idxs], dtype=torch.float32).to(device))


a = np.arange(30).reshape((-1, 10))
b = np.array([2, 3, 4, 5])
np.broadcast_to(b, (a.shape[0], b.shape[0]))
np.hstack((a, np.broadcast_to(b, (a.shape[0], b.shape[0]))))
