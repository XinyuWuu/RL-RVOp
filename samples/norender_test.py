import os
import mujoco as mj
from mujoco import MjModel
from mujoco import MjData
import numpy as np
import matplotlib.pyplot as plt

model = MjModel.from_xml_path("assets/test.xml")
data = MjData(model)

total_sim_time = 12
step_size = 0.01
step_num = int(step_size / 0.002)
flag = False
# TODO set controller callback
nth = 0
omegal = -7.0
omegar = 7.0
data.ctrl[nth * 2:nth * 2 + 2] = np.array([omegal, omegar]) * 7

# data.ctrl[0:2]=np.array([5,5])

# mj.mj_step(model,data,1000)

# data.joint("robot_0").qvel
# data.joint("robot_1").qvel

# qvel=[data.joint("robot_0").qvel,data.joint("robot_1").qvel]


linv = np.zeros((int(np.ceil(total_sim_time / step_size)),))
rotv = np.zeros((int(np.ceil(total_sim_time / step_size)),))

step = 0
time_prev = data.time
while True:
    time_prev = data.time
    p0 = np.copy(data.joint(f"robot_{nth}").qpos[0:2])
    mj.mj_step(model, data, step_num)
    p1 = np.copy(data.joint(f"robot_{nth}").qpos[0:2])
    linv[step] = np.linalg.norm(p1 - p0) / (data.time - time_prev)
    rotv[step] = data.joint(f"robot_{nth}").qvel[5]

    step += 1
    if (data.time > total_sim_time):
        break

plt.plot(np.array(range(0, len(linv))) * step_size, linv, color='g')
plt.plot(np.array(range(0, len(rotv))) * step_size, rotv, color='b')
plt.savefig("assets/test.png")

plt.clf()

a = data.joint(f"left-wheel_{nth}").qvel[0]
b = omegal
b / a


a = data.joint(f"robot_{nth}").qvel[5]
b = (data.joint(f"right-wheel_{nth}").qvel[0] -
     data.joint(f"left-wheel_{nth}").qvel[0]) * 0.04 / 0.14 / 2
b / a

a = np.linalg.norm(data.joint(f"robot_{nth}").qvel[0:2])
b = (data.joint(f"right-wheel_{nth}").qvel[0] +
     data.joint(f"left-wheel_{nth}").qvel[0]) * 0.04 / 2
b / a
