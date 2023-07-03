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


EC = envCreator.EnvCreator(PARAMs["robot_r"])
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

modelfile, Nrobot, target, contour, ow, oh, w, h = EC.env_create4(0, 0)

ctrl = [[1.0, 0]]*Nrobot


env.setSim(modelfile, Nrobot, target, contour, True, ow, oh)
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
