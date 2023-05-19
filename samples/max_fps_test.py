import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
import os
import mujoco as mj
from numpy.linalg import norm
from numpy import array, arctan2, flipud, zeros
import numpy as np
from time import sleep, time
from CppClass.CtrlConverter import CtrlConverter
import matplotlib.pyplot as plt
import importlib
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
PARAMs["hidden_sizes"] = [1024] * 4
model_file = "module_saves/nornn29/112h_23min_5639999steps_16625150updates_policy.ptd"
PARAMs["device"]="cpu"
torch.manual_seed(PARAMs["seed"])
np.random.seed(PARAMs["seed"])
random.seed(PARAMs["seed"])

torch.set_num_threads(torch.get_num_threads())

Pi = nornnsac.nornncore.Policy(obs_dim=PARAMs["obs_dim"], act_dim=PARAMs["act_dim"],
                               act_limit=PARAMs["act_limit"], hidden_sizes=PARAMs["hidden_sizes"])
Pi.load_state_dict(torch.load(
    model_file, map_location=torch.device(PARAMs["device"])))
Pi.to(device=PARAMs["device"])
Pi.act_limit = Pi.act_limit.to(device=PARAMs["device"])

for p in Pi.parameters():
    p.requires_grad = False

Pi.eval()

total_params = sum(p.numel() for p in Pi.parameters())
print(total_params)

data = torch.rand((1, 180), dtype=torch.float32, device=PARAMs["device"])

Pi(data[0], True, with_logprob=False)
test_time = 102400
start_t = time()
for d in range(test_time):
    Pi(data, True, with_logprob=False)
end_t = time()

print(test_time / (end_t - start_t))
