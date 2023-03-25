import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_sequence, unpack_sequence


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Policy(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, rnn_state_size, rnn_layer, bidir):
        super().__init__()
        # self.net0 = nn.Sequential(
        #     nn.Linear(obs_dim, rnn_state_size),
        #     nn.ReLU(),
        #     nn.Linear(rnn_state_size, rnn_state_size),
        #     nn.ReLU(),
        # )
        self.rnn = nn.GRU(input_size=rnn_state_size, hidden_size=rnn_state_size,
                          num_layers=rnn_layer, bidirectional=bidir, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(2 * rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(rnn_state_size, act_dim)
        self.log_std_layer = nn.Linear(rnn_state_size, act_dim)
        self.act_limit = torch.tensor(act_limit)
        self.act_limit_prod_log = np.log(self.act_limit.prod())
        self.bidir = bidir

    def forward(self, obs, deterministic=False, with_logprob=True):
        x0 = rnn_utils.pack_sequence(obs, False)
        # x1 = rnn_utils.PackedSequence(
        #     data=self.net0(x0.data),
        #     batch_sizes=x0.batch_sizes,
        #     sorted_indices=x0.sorted_indices,
        #     unsorted_indices=x0.unsorted_indices)
        h, ht = self.rnn(x0)

        if self.bidir:
            y = self.net(torch.concat([ht[-1], ht[-2]], dim=1))
        else:
            y = self.net(ht[-1])

        mu = self.mu_layer(y)
        log_std = self.log_std_layer(y)

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

    def __init__(self, obs_dim, act_dim, rnn_state_size, rnn_layer, bidir):
        super().__init__()
        # self.net0 = nn.Sequential(
        #     nn.Linear(obs_dim + act_dim, rnn_state_size),
        #     nn.ReLU(),
        #     nn.Linear(rnn_state_size, rnn_state_size),
        #     nn.ReLU(),
        # )
        self.rnn = nn.GRU(input_size=rnn_state_size, hidden_size=rnn_state_size,
                          num_layers=rnn_layer, bidirectional=bidir, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(2 * rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, rnn_state_size),
            nn.ReLU(),
            nn.Linear(rnn_state_size, 1),
        )
        self.bidir = bidir

    def forward(self, obs, act):
        obsact = list(map(lambda i: torch.hstack([obs[i], torch.broadcast_to(
            act[i], (obs[i].shape[0], act[i].shape[0]))]), range(obs.__len__())))
        obsact = rnn_utils.pack_sequence(obsact, False)
        # x0 = rnn_utils.PackedSequence(
        #     data=self.net0(obsact.data),
        #     batch_sizes=obsact.batch_sizes,
        #     sorted_indices=obsact.sorted_indices,
        #     unsorted_indices=obsact.unsorted_indices)
        h, ht = self.rnn(obsact)
        if self.bidir:
            q = self.net(torch.concat([ht[-1], ht[-2]], dim=1))
        else:
            q = self.net(ht[-1])
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class ReplayBufferLite:
    def __init__(self, obs_dim, act_dim, max_obs, max_size):
        self.obsbegin_buf = np.zeros(max_size, dtype=np.int64)
        self.obs2begin_buf = np.zeros(max_size, dtype=np.int64)
        self.obsend_buf = np.zeros(max_size, dtype=np.int64)
        self.obs2end_buf = np.zeros(max_size, dtype=np.int64)

        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)

        self.policy_buf1 = np.zeros(
            (max_obs * max_size, obs_dim), dtype=np.float32)
        self.policy_buf2 = np.zeros(
            (max_obs * max_size, obs_dim), dtype=np.float32)
        self.ptr, self.size, self.max_obs, self.max_size = 0, 0, max_obs, max_size - 1

    def store(self, obs, act, rew, next_obs, done):
        self.obsend_buf[self.ptr] = obs.shape[0] + \
            self.obsbegin_buf[self.ptr]
        self.obs2end_buf[self.ptr] = next_obs.shape[0] + \
            self.obs2begin_buf[self.ptr]

        self.obsbegin_buf[self.ptr + 1] = self.obsend_buf[self.ptr]
        self.obs2begin_buf[self.ptr + 1] = self.obs2end_buf[self.ptr]

        self.policy_buf1[self.obsbegin_buf[self.ptr]                         :self.obsend_buf[self.ptr]] = obs
        self.policy_buf2[self.obs2begin_buf[self.ptr]                         :self.obs2end_buf[self.ptr]] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size, device):
        idxs = list(np.random.randint(0, self.size, size=batch_size))
        return dict(
            obs=[torch.as_tensor(
                self.policy_buf1[self.obsbegin_buf[idx]                                 :self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            obs2=[torch.as_tensor(
                self.policy_buf2[self.obsbegin_buf[idx]:self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            act=torch.as_tensor(
                self.act_buf[idxs], dtype=torch.float32).to(device),
            rew=torch.as_tensor(
                self.rew_buf[idxs], dtype=torch.float32).to(device),
            done=torch.as_tensor(
                self.done_buf[idxs], dtype=torch.float32).to(device),
        )


class ReplayBuffer:
    def __init__(self, obs_dim, obs_self_dim, act_dim, max_obs, max_size):
        self.obsbegin_buf = np.zeros(max_size, dtype=np.int64)
        self.obs2begin_buf = np.zeros(max_size, dtype=np.int64)
        self.obsend_buf = np.zeros(max_size, dtype=np.int64)
        self.obs2end_buf = np.zeros(max_size, dtype=np.int64)

        self.obs_self_buf = np.zeros(
            (max_size, obs_self_dim), dtype=np.float32)
        self.obs_buf = np.zeros(
            (max_size * max_obs, obs_dim), dtype=np.float32)
        self.obs2_self_buf = np.zeros(
            (max_size, obs_self_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(
            (max_size * max_obs, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)

        self.policy_buf1 = np.zeros(
            (max_obs * max_size, obs_dim + obs_self_dim), dtype=np.float32)
        self.policy_buf2 = np.zeros(
            (max_obs * max_size, obs_dim + obs_self_dim), dtype=np.float32)
        self.ptr, self.size, self.max_obs, self.max_size = 0, 0, max_obs, max_size - 1

    def store(self, obs, obs_self, act, rew, next_obs, next_obs_self, done):
        self.obsend_buf[self.ptr] = obs.shape[0] + \
            self.obsbegin_buf[self.ptr]
        self.obs2end_buf[self.ptr] = next_obs.shape[0] + \
            self.obs2begin_buf[self.ptr]

        self.obsbegin_buf[self.ptr + 1] = self.obsend_buf[self.ptr]
        self.obs2begin_buf[self.ptr + 1] = self.obs2end_buf[self.ptr]

        self.obs_self_buf[self.ptr] = obs_self
        self.obs_buf[self.obsbegin_buf[self.ptr]:self.obsend_buf[self.ptr]] = obs

        self.obs2_self_buf[self.ptr] = next_obs_self
        self.obs2_buf[self.obs2begin_buf[self.ptr]:self.obs2end_buf[self.ptr]] = next_obs

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.policy_buf1[self.obsbegin_buf[self.ptr]:self.obsend_buf[self.ptr]] = np.hstack(
            [obs, np.broadcast_to(obs_self, (obs.shape[0], obs_self.shape[0]))])
        self.policy_buf2[self.obs2begin_buf[self.ptr]:self.obs2end_buf[self.ptr]] = np.hstack(
            [next_obs, np.broadcast_to(next_obs_self, (next_obs.shape[0], next_obs_self.shape[0]))])

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size, device):
        idxs = list(np.random.randint(0, self.size, size=batch_size))

        return dict(
            obs_sur=[torch.as_tensor(
                self.obs_buf[self.obsbegin_buf[idx]:self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            obs2_sur=[torch.as_tensor(
                self.obs2_buf[self.obs2begin_buf[idx]:self.obs2end_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            obs_self=torch.as_tensor(
                self.obs_self_buf[idxs], dtype=torch.float32).to(device),
            obs2_self=torch.as_tensor(
                self.obs2_self_buf[idxs], dtype=torch.float32).to(device),
            obs=[torch.as_tensor(
                self.policy_buf1[self.obsbegin_buf[idx]
                    :self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            obs2=[torch.as_tensor(
                self.policy_buf2[self.obsbegin_buf[idx]
                    :self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            act=torch.as_tensor(
                self.act_buf[idxs], dtype=torch.float32).to(device),
            rew=torch.as_tensor(
                self.rew_buf[idxs], dtype=torch.float32).to(device),
            done=torch.as_tensor(
                self.done_buf[idxs], dtype=torch.float32).to(device))

    def sample_batch_lite(self, batch_size, device):
        idxs = list(np.random.randint(0, self.size, size=batch_size))
        # idxs_order1 = sorted(
        #     idxs, key=lambda x: self.obsend_buf[x] - self.obsbegin_buf[x])
        # idxs_order2 = sorted(
        #     idxs, key=lambda x: self.obs2end_buf[x] - self.obs2begin_buf[x])
        return dict(
            obs=[torch.as_tensor(
                self.policy_buf1[self.obsbegin_buf[idx]                                 :self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            obs2=[torch.as_tensor(
                self.policy_buf2[self.obsbegin_buf[idx]:self.obsend_buf[idx]], dtype=torch.float32
            ).to(device) for idx in idxs],
            act=torch.as_tensor(
                self.act_buf[idxs], dtype=torch.float32).to(device),
            rew=torch.as_tensor(
                self.rew_buf[idxs], dtype=torch.float32).to(device),
            done=torch.as_tensor(
                self.done_buf[idxs], dtype=torch.float32).to(device),
            # order1=idxs_order1,
            # order2=idxs_order2
        )


# a = np.arange(30).reshape((-1, 10))
# b = np.array([2, 3, 4, 5])
# np.broadcast_to(b, (a.shape[0], b.shape[0]))
# np.hstack((a, np.broadcast_to(b, (a.shape[0], b.shape[0]))))


# tliner = nn.Linear(5, 10)
# tGRU = nn.GRU(input_size=10, hidden_size=1, num_layers=3,
#               bidirectional=True, batch_first=True)
# t = [torch.arange(10, 20, dtype=torch.float32).reshape((-1, 5)),
#      torch.arange(20, 50, dtype=torch.float32).reshape((-1, 5)),
#      torch.arange(50, 70, dtype=torch.float32).reshape((-1, 5)),
#      torch.arange(90, 140, dtype=torch.float32).reshape((-1, 5)),
#      torch.arange(200, 210, dtype=torch.float32).reshape((-1, 5)),
#      torch.arange(40, 100, dtype=torch.float32).reshape((-1, 5)),
#      ]
# resort = [2, 3, 5, 1, 0, 4]
# t1 = [t[i] for i in resort]
# tp = rnn_utils.pack_sequence(t, enforce_sorted=False)
# tp1 = rnn_utils.pack_sequence(t1, enforce_sorted=False)

# tl = tliner(tp.data)
# tl1 = tliner(tp1.data)

# ltp = rnn_utils.PackedSequence(tl, batch_sizes=tp.batch_sizes,
#                                sorted_indices=tp.sorted_indices, unsorted_indices=tp.unsorted_indices)
# ltp1 = rnn_utils.PackedSequence(tl1, batch_sizes=tp1.batch_sizes,
#                                 sorted_indices=tp1.sorted_indices, unsorted_indices=tp1.unsorted_indices)

# ltp = rnn_utils.PackedSequence(tl, batch_sizes=tp.batch_sizes)
# ltp1 = rnn_utils.PackedSequence(tl1, batch_sizes=tp1.batch_sizes)

# h, ht = tGRU(ltp)
# ht.shape

# h1, ht1 = tGRU(ltp1)
# ht1.shape

# torch.hstack([ht[-1], ht[-2]])
# torch.hstack([ht1[-1], ht1[-2]])
# resort

# rebuf = ReplayBufferLite(3, 2, 2, 16, 1000)
# for i in range(100):
#     obs1length = np.random.randint(1, 16)
#     obs2length = np.random.randint(1, 16)

#     rebuf.store(np.random.rand(obs1length, 3),
#                 np.random.rand(2),
#                 np.random.rand(2),
#                 np.random.rand(1),
#                 np.random.rand(obs2length, 3),
#                 np.random.rand(2),
#                 0
#                 )
# batch = rebuf.sample_batch(4, 'cuda')

# qfunc = Qfunc(5, 2, 32, 3).to("cuda")
# policy = Policy(5, 2, [1, 25], 32, 3).to("cuda")
# policy.act_limit = policy.act_limit.to("cuda")

# policy.net0[0].weight.grad

# qfunc(batch["obs"], batch["act"])
# a, p = policy(batch["obs2"])
# t = qfunc(batch["obs2"], a)

# policy.net0[0].weight.grad
# t.sum().backward()
# policy.net0[0].weight.grad

# policy([batch["obs"][0], batch["obs"][1], batch["obs"][0]],True)
# qfunc([batch["obs"][0], batch["obs"][1], batch["obs"][0]],
#       [batch["act"][0],batch["act"][1],batch["act"][0]])
