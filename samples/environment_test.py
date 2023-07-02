# export LD_LIBRARY_PATH="CppClass/mujoco/lib;CppClass/glfw/src"
# export DISPLAY=:1
import sys
from base_config import PARAMs
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
dmax = 3.0
EC = envCreator.EnvCreator()
CG = contourGenerator.ContourGenrator(PARAMs["robot_r"])
env = Environment()
PARAMs["rmax"] = env.setCtrl(vmax=PARAMs["vmax"], tau=PARAMs["tau"],
                             wheel_d=PARAMs["wheel_d"], wheel_r=PARAMs["wheel_r"],
                             gain=PARAMs["gain"])
env.setRvop(dmax=PARAMs["dmax"], robot_r=PARAMs["robot_r"])
env.setRwd(robot_r=PARAMs["robot_r"], vmax=PARAMs["vmax"], rmax=PARAMs["rmax"], tolerance=PARAMs["tolerance"], dreach=PARAMs["dreach"], tb=PARAMs["tb"],
           a=PARAMs["a"], b=PARAMs["b"], c=PARAMs["c"], d=PARAMs["d"], e=PARAMs["e"],
           f=PARAMs["f"], g=PARAMs["g"], eta=PARAMs["eta"],
           h=PARAMs["h"], mu=PARAMs["mu"], rreach=PARAMs["rreach"],
           remix=PARAMs["remix"], rm_middle=PARAMs["rm_middle"], dmax=PARAMs["dmax"], w=PARAMs["w"])

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
NNinput1 = np.frombuffer(
    env.get_NNinput1(), dtype=np.float64
).reshape(Nrobot, 180)
reward = np.frombuffer(env.get_r(), dtype=np.float64)
reward_mix = np.frombuffer(env.get_rm(), dtype=np.float64)
death = np.frombuffer(env.get_d(), dtype=np.int32)

env.stepVL(ctrl, 1, 1)
env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels1 = posvels.copy()

env.stepVL(ctrl, 3000, 5)
env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels2 = posvels.copy()

env.cal_obs(True)
env.cal_NNinput1(6)
env.cal_reward()

print(NNinput1)
print(reward)
print(reward_mix)
print(death)


env.setSim(modelfile, Nrobot, [list(t)
           for t in target], contour, True, ow, oh)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((ow, oh, 3))
NNinput1 = np.frombuffer(
    env.get_NNinput1(), dtype=np.float64
).reshape(Nrobot, 180)
reward = env.get_r()
reward_mix = env.get_rm()
death = env.get_d()

env.stepVL(ctrl, 0, 5)
env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels1 = posvels.copy()

env.stepVL(ctrl, 3000, 5)
env.render()
img = Image.fromarray(img_arr, "RGB")
img.save("assets/test.png")
posvels2 = posvels.copy()

env.cal_obs(True)
env.cal_NNinput1(6)
env.cal_reward()

print(NNinput1)
print(reward)
print(reward_mix)
print(death)


env.CloseGLFW()
