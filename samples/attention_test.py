import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

convo1 = nn.Conv1d(10 * 16, 128 * 16, 1, groups=16)
fc1 = nn.Linear(10, 128)
convo1(torch.rand((123, 10 * 16, 1))).size()
fc1(torch.rand((123, 10))).size()

nn.Conv1d(10, 16*128, 1)(torch.rand((123, 10, 1))).size()
torch.rand((123, 10)).reshape(123, 1, 10).broadcast_to((123, 16, 10))
