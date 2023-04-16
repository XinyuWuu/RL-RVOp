# from core import Qfunc
from torch import tensor, rand, nn
import torch
from random import randint
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_sequence, unpack_sequence


class Qfunc(nn.Module):

    def __init__(self, obs_dim, act_dim, rnn_state_size, rnn_layer, bidir):
        super().__init__()
        self.bidir = bidir
        self.net0 = nn.Sequential(
            nn.Linear(act_dim, 2 * rnn_state_size),
            nn.ReLU(),
            nn.Linear(2 * rnn_state_size, 2 * rnn_state_size),
            nn.ReLU(),
            nn.Linear(2 * rnn_state_size,
                      (2 if self.bidir else 1) * rnn_state_size),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=rnn_state_size,
                          num_layers=rnn_layer, bidirectional=bidir, batch_first=True)

        self.net1 = nn.Sequential(
            nn.Linear((4 if self.bidir else 2) *
                      rnn_state_size, 2 * rnn_state_size),
            nn.ReLU(),
            nn.Linear(2 * rnn_state_size, 2 * rnn_state_size),
            nn.ReLU(),
            nn.Linear(2 * rnn_state_size, 2 * rnn_state_size),
            nn.ReLU(),
            nn.Linear(2 * rnn_state_size, 1),
        )

    def forward(self, obs, act):
        h, ht = self.rnn(obs)
        x1 = self.net0(act)
        if self.bidir:
            x1 = torch.concat([ht[-1], ht[-2], x1], dim=1)
        else:
            x1 = torch.concat([ht[-1], x1], dim=1)
        q = self.net1(x1)

        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


Q = Qfunc(10, 2, 512, 3, False).to("cuda")

obs = []
for i in range(1024):
    obs_len = randint(2, 16)
    obs.append(rand((obs_len, 10)))
act = rand((1024, 2))

obs, act = rnn_utils.pack_sequence(obs, False).to("cuda"), act.to("cuda")
start_t = time()
for i in range(128):
    print(Q(obs, act).size())
end_t = time()

time_per = (end_t - start_t) / 128
print(time_per)
