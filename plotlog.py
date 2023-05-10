import re
from numpy import array
from matplotlib import pyplot as plt
step_now = 0
eps_rets = []
eps_rets = []
pi_loss = []
q_loss = []

# log_file_url = "module_saves/data300g/module_saves/nornn32/0000aaaa.txt"
log_file_url = "module_saves/nornn32/0000aaaa.txt"
with open(log_file_url, "r") as fp:
    for line in fp.readlines():
        m = re.search(r"t: ([0-9]+).*?ep_ret: [-0-9.]*?:([-0-9.]*)", line)
        if m != None:
            eps_rets.append([int(m.groups()[0]), float(m.groups()[1])])
        m = re.search(r"step ([-0-9.]*);", line)
        if m != None:
            step_now = int(m.groups()[0])
        m = re.search(r"mean losspi: ([-0-9.]*)", line)
        if m != None:
            pi_loss.append([step_now, float(m.groups()[0])])
        m = re.search(r"mean lossq: ([-0-9.]*)", line)
        if m != None:
            q_loss.append([step_now, float(m.groups()[0])])
plt.plot(array(eps_rets)[:, 0], array(eps_rets)[:, 1], label="ave ret")
plt.plot(array(pi_loss)[:, 0], array(pi_loss)[:, 1], label="pi loss")
plt.plot(array(q_loss)[:, 0], array(q_loss)[:, 1], label="q loss")
plt.legend()
plt.show()
