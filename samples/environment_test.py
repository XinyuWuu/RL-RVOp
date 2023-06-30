# export LD_LIBRARY_PATH="CppClass/mujoco/lib;CppClass/glfw/src"
import sys

from torch import rand
if sys.path[0] != '':
    sys.path = [''] + sys.path
from CppClass.Environment import Environment
import envCreator
import contourGenerator
import numpy as np
from numpy import zeros
from PIL import Image
from matplotlib import pyplot as plt
import random

robot_r = 0.2
EC = envCreator.EnvCreator()
CG = contourGenerator.ContourGenrator(robot_r)
env = Environment()
env.setCtrl(1, 0.5, 0.04, 0.28, 7)

Nrobot = 0
robot_text = ""
obs_text = ""
obs = []
robot = []
target = []
ow, oh, ch, fovy, w, h = 1920, 1920, 17.5, 75, 22, 22

id = Nrobot
cs = EC.circle_points(7, 20)
for c in cs:
    robot_text += EC.robot(id=id, c=np.array(c),
                           theta=np.random.rand()*np.pi*2)
    robot.append(c)
    id += 1

target += EC.target_trans(cs, 3)
Nrobot += len(cs)

id = Nrobot
cs = EC.circle_points(5, 10)
for c in cs:
    robot_text += EC.robot(id=id, c=np.array(c),
                           theta=np.random.rand()*np.pi*2)
    robot.append(c)
    id += 1

target += EC.target_trans(cs, 1)
Nrobot += len(cs)

id = Nrobot
cs = EC.line_points(np.array([-10.0, 5]), np.array([-10.0, -5]), 10)
for c in cs:
    robot_text += EC.robot(id=id, c=np.array(c),
                           theta=np.random.rand()*np.pi*2)
    robot.append(c)
    id += 1

target += EC.target_trans(cs, 2)
Nrobot += len(cs)

cs = EC.circle_points(r=3.5, n=15)
for c in cs:
    obs_t, obs_ = EC.obs(np.array(c), 0.15, 1, 5)
    obs_text += obs_t
    obs += obs_

actuator_text = EC.actuator(Nrobot)

modelfile = EC.env_text(robot_text, obs_text,
                        actuator_text, ow, ow, ch, fovy)

contour = CG.generate_contour(obs)
ctrl = [[1.0, 0]]*Nrobot

env.setSim(modelfile, Nrobot, [list(t)
           for t in target], contour, True, ow, oh)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((ow, oh, 3))

env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels1 = posvels.copy()

env.stepVL(ctrl, 3000, 5)
env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels2 = posvels.copy()
print(posvels2-posvels1)


env.setSim(modelfile, Nrobot, [list(t)
           for t in target], contour, True, ow, oh)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((ow, oh, 3))

env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels1 = posvels.copy()

env.stepVL(ctrl, 3000, 5)
env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels2 = posvels.copy()
print(posvels2-posvels1)

env.CloseGLFW()
